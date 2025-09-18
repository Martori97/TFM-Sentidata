#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import argparse
import warnings
import inspect
import datetime
from dataclasses import asdict
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------- MLflow (opcional) ----------
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings("ignore")

# -------------------- Utilidades ----------------------- #

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
POSSIBLE_LABEL_COLS = ["label", "label3", "target"]

def repo_root_from_file(__file__: str) -> Path:
    # scripts/training/train_albert_from_parquets.py → raíz del repo dos niveles arriba
    return Path(__file__).resolve().parents[2]

def load_params(path: str) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) if yaml else json.load(f)
    return {}

def resolve_text_and_label_cols(df: pd.DataFrame, cfg_model: Dict[str, Any]) -> (str, str):
    # texto
    text_col = cfg_model.get("text_col", "review_text")
    if text_col not in df.columns:
        # fallback a review_text si pidieron review_text_clean y no existe
        if "review_text" in df.columns:
            text_col = "review_text"
        else:
            raise ValueError(f"No existe columna de texto '{text_col}' ni 'review_text' en el dataset.")
    # label
    label_col = cfg_model.get("label_col", None)
    if label_col and label_col in df.columns:
        return text_col, label_col
    # intenta alguna conocida
    for c in POSSIBLE_LABEL_COLS:
        if c in df.columns:
            return text_col, c
    # sino, rating
    if "rating" in df.columns:
        return text_col, "rating"
    raise ValueError("No encuentro columna de label (label/label3/target) ni 'rating'.")

def map_rating_to_label3(r) -> Optional[int]:
    try:
        r = int(r)
    except Exception:
        return None
    if r in (1, 2): return 0
    if r == 3:      return 1
    if r in (4, 5): return 2
    return None

def to_hf_dataset(df: pd.DataFrame, text_col: str, label_col: str) -> Dataset:
    # asegura labels int 0/1/2
    if label_col == "rating":
        labels = [map_rating_to_label3(x) for x in df["rating"].tolist()]
    else:
        if df[label_col].dtype == object:
            labels = [LABEL2ID.get(str(x).lower(), None) for x in df[label_col].tolist()]
        else:
            labels = df[label_col].tolist()
    labels = np.array(labels)
    m = pd.Series(labels).notna().to_numpy()
    labels = labels[m].astype(int)
    df2 = df.loc[m, [text_col]].copy()
    df2["labels"] = labels
    # id (opcional)
    if "review_id" in df.columns:
        df2["review_id"] = df.loc[m, "review_id"].values
    return Dataset.from_pandas(df2.reset_index(drop=True), preserve_index=False)

def read_parquet_any(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    # Soporta carpeta (dataset particionado) o archivo .parquet
    return pd.read_parquet(path, engine="pyarrow", columns=columns)

def compute_class_weights(y_train: np.ndarray, num_labels: int,
                          mode: str = "auto",
                          alpha: float = 1.0,
                          manual: Optional[List[float]] = None) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        return np.ones(num_labels, dtype=np.float32)

    if mode == "manual":
        arr = np.array(manual or [1.0] * num_labels, dtype=np.float32)
        if arr.size != num_labels:
            if arr.size < num_labels:
                arr = np.pad(arr, (0, num_labels - arr.size), constant_values=arr.mean() if arr.size > 0 else 1.0)
            else:
                arr = arr[:num_labels]
        # normaliza a media 1
        mu = float(arr.mean()) if float(arr.mean()) != 0.0 else 1.0
        return (arr / mu).astype(np.float32)

    # auto: inverso a la frecuencia con exponente alpha
    counts = np.bincount(y_train, minlength=num_labels).astype(np.float64)
    priors = counts / max(counts.sum(), 1.0)
    inv = 1.0 / np.maximum(priors, 1e-8)
    if alpha != 1.0:
        inv = inv ** float(alpha)
    mu = float(inv.mean()) if float(inv.mean()) != 0.0 else 1.0
    return (inv / mu).astype(np.float32)

# -------------------- Métricas ----------------------- #

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    y_pred = np.argmax(preds, axis=-1)
    acc = accuracy_score(labels, y_pred)
    p, r, f1m, _ = precision_recall_fscore_support(labels, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1m), "precision_macro": float(p), "recall_macro": float(r)}

# -------------------- Trainer con mejoras ----------------------- #

class WeightedTrainer(Trainer):
    """
    - class_weights (tensor) opcional
    - label_smoothing_factor (float) opcional
    - focal loss opcional (gamma, alpha por clase)
    - logit adjustment opcional (tau * log(prior))
    """
    def __init__(
        self, *args,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing_factor: float = 0.0,
        focal_enable: bool = False,
        focal_gamma: float = 1.5,
        focal_alpha: Optional[torch.Tensor] = None,
        enable_logit_adjustment: bool = False,
        logit_adjustment_tau: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.ls = float(label_smoothing_factor or 0.0)
        self.focal_enable = bool(focal_enable)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = focal_alpha  # tensor del tamaño [num_labels] o None
        self.enable_logit_adjustment = bool(enable_logit_adjustment)
        self.logit_adjustment_tau = float(logit_adjustment_tau)
        self._log_prior = None  # se calcula perezosamente

    def _maybe_build_log_prior(self, logits: torch.Tensor):
        if not self.enable_logit_adjustment or self._log_prior is not None:
            return
        # Priors desde el train_dataset de este trainer
        if not hasattr(self, "train_dataset") or self.train_dataset is None:
            return
        try:
            labels = np.array(self.train_dataset["labels"])
        except Exception:
            labels = np.array(list(self.train_dataset["labels"]))
        num_classes = logits.size(-1)
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64) + 1e-6
        priors = counts / counts.sum()
        lp = np.log(priors)
        self._log_prior = torch.tensor(lp, dtype=logits.dtype, device=logits.device)

    def _loss_ce(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.ls if self.ls > 0 else 0.0,
        )
        return loss_fct(logits, labels)

    def _loss_focal(self, logits, labels):
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        ce = F.cross_entropy(logits, labels, reduction="none")
        modulating = (1.0 - pt) ** self.focal_gamma
        if self.focal_alpha is not None:
            alpha_w = self.focal_alpha.to(logits.device).gather(0, labels)
        else:
            alpha_w = torch.ones_like(pt)
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device).gather(0, labels)
        else:
            cw = torch.ones_like(pt)
        loss = (alpha_w * cw * modulating * ce).mean()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        # Logit adjustment (solo si está activado)
        if self.enable_logit_adjustment:
            self._maybe_build_log_prior(logits)
            if self._log_prior is not None:
                logits = logits - self.logit_adjustment_tau * self._log_prior

        if self.focal_enable:
            loss = self._loss_focal(logits, labels)
        else:
            loss = self._loss_ce(logits, labels)

        return (loss, outputs) if return_outputs else loss

# -------------------- TrainingArguments compat ----------------------- #

def build_training_args_compat(kwargs: Dict[str, Any]) -> TrainingArguments:
    """
    Tu código original ya hace compat con transformers 4.5x: imprime eval/save efectivos
    y poda claves no soportadas. Mantengo la idea para que veas lo que realmente se usa.
    """
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    removed = {}

    # Alinea evaluation/save si uno es "no" y el otro no
    save_allowed = "save_strategy" in allowed
    eval_key = "evaluation_strategy" if "evaluation_strategy" in allowed else "eval_strategy"

    if "load_best_model_at_end" in kwargs and kwargs.get("load_best_model_at_end") and \
       (eval_key not in kwargs or str(kwargs.get(eval_key, "no")).lower() == "no"):
        # Si pidieron load_best pero no hay eval, desactiva para evitar warning HF (como viste en logs)
        kwargs["load_best_model_at_end"] = False
        if save_allowed:
            kwargs["save_strategy"] = "no"
    else:
        ev = str(kwargs.get(eval_key, "no")).lower()
        sv = str(kwargs.get("save_strategy", "no")).lower()
        if ev == "no" and sv in {"steps", "epoch"}:
            kwargs[eval_key] = sv
        elif sv == "no" and ev in {"steps", "epoch"}:
            kwargs["save_strategy"] = ev

    # Podado de claves no soportadas por tu versión
    for k in list(kwargs.keys()):
        if k not in allowed:
            removed[k] = kwargs.pop(k)

    if removed:
        warnings.warn(
            "TrainingArguments: ignoradas claves no soportadas por tu versión de transformers: "
            f"{sorted(removed.keys())}"
        )

    print(f"[hf] transformers={__import__('transformers').__version__}")
    print(f"[args] eval={kwargs.get(eval_key,'no')} save={kwargs.get('save_strategy','no')} "
          f"load_best={kwargs.get('load_best_model_at_end')}")

    return TrainingArguments(**kwargs)

# -------------------- Main ----------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Parquet (carpeta o archivo) con split train")
    ap.add_argument("--val", required=True, help="Parquet (carpeta o archivo) con split validation")
    ap.add_argument("--model-out", required=True, help="Carpeta para guardar el modelo")
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()

    cfg = load_params(args.params)
    m = (cfg.get("model") or {}).get("albert") or {}
    tr = (cfg.get("training") or {})
    ml = (cfg.get("mlflow") or {})

    # --------- MLflow: tracking_uri portable + run_name único --------- #
    run = bool(cfg.get("mlflow", {}).get("enable", True) and MLFLOW_AVAILABLE)
    if run:
        # 1) tracking_uri: si relativo (file:mlruns o file:./mlruns), normaliza a raíz del repo
        tracking_uri = ml.get("tracking_uri", "file:mlruns")
        if tracking_uri.startswith("file:"):
            uri_path = tracking_uri[len("file:"):]
            if uri_path.startswith("./") or not uri_path.startswith("/"):
                root = repo_root_from_file(__file__)
                final = f"file:{str((root / uri_path).resolve())}"
            else:
                final = tracking_uri
        else:
            final = tracking_uri
        try:
            mlflow.set_tracking_uri(final)
        except Exception:
            pass
        try:
            mlflow.set_experiment(ml.get("experiment", "default"))
        except Exception:
            pass
        # 2) run_name único
        base_name = ml.get("run_name", "run")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_run_name = f"{base_name}_{timestamp}"
        try:
            mlflow.start_run(run_name=unique_run_name, tags=ml.get("tags"))
            print(f"[mlflow] Started run: {unique_run_name}")
        except Exception:
            run = False

    # --------- Carga Parquet y resolución de columnas --------- #
    # Lee train/val (solo columnas necesarias)
    train_df = read_parquet_any(args.train)
    val_df   = read_parquet_any(args.val)

    text_col_tr, label_col_tr = resolve_text_and_label_cols(train_df, m)
    text_col_va, label_col_va = resolve_text_and_label_cols(val_df, m)

    # Construye Dataset HF (labels→int 0/1/2)
    ds_tr = to_hf_dataset(train_df, text_col_tr, label_col_tr)
    ds_va = to_hf_dataset(val_df, text_col_va, label_col_va)
    ds = DatasetDict(train=ds_tr, validation=ds_va)

    # --------- Tokenizer y tokenización --------- #
    model_name = m.get("model_name", "albert-base-v2")
    max_len = int(m.get("max_len", 192))
    num_proc = int(m.get("tokenize_num_proc", 6))

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def tok_fn(batch):
        return tok(batch[text_col_tr], truncation=True, padding=False, max_length=max_len)

    ds_tok = DatasetDict()
    ds_tok["train"] = ds["train"].map(tok_fn, batched=True, num_proc=num_proc, remove_columns=[text_col_tr])
    ds_tok["validation"] = ds["validation"].map(tok_fn, batched=True, num_proc=num_proc, remove_columns=[text_col_va])

    cols = ["input_ids", "attention_mask", "labels"]
    ds_tok["train"].set_format(type="torch", columns=cols)
    ds_tok["validation"].set_format(type="torch", columns=cols)

    # --------- Class Weights SOLO desde TRAIN --------- #
    cw_cfg = (tr.get("class_weights") or {})
    cw_mode = str(cw_cfg.get("mode", "none")).lower()
    cw_alpha = float(cw_cfg.get("alpha", 1.0))
    cw_manual = cw_cfg.get("manual", [1.0, 1.0, 1.0])

    y_train = np.array(ds_tok["train"]["labels"], dtype=np.int64)
    num_labels = int(m.get("num_labels", 3))
    class_w = compute_class_weights(y_train, num_labels, mode=cw_mode, alpha=cw_alpha, manual=cw_manual)
    print(f"[class_weights] {cw_mode}(alpha={cw_alpha}) -> {class_w.tolist()}")
    if run and MLFLOW_AVAILABLE:
        try:
            mlflow.log_dict({"class_weights": [float(x) for x in class_w.tolist()]}, "class_weights.json")
        except Exception:
            pass

    # --------- Modelo --------- #
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    # ajustes de dropout de la cabeza si vienen en params
    classifier_dropout_prob = m.get("classifier_dropout_prob", None)
    if classifier_dropout_prob is not None:
        try:
            if hasattr(model.config, "classifier_dropout_prob"):
                model.config.classifier_dropout_prob = float(classifier_dropout_prob)
            if hasattr(model, "classifier") and hasattr(model.classifier, "dropout"):
                model.classifier.dropout.p = float(classifier_dropout_prob)
            elif hasattr(model, "dropout"):
                model.dropout.p = float(classifier_dropout_prob)
            print(f"[dropout] classifier_dropout_prob={classifier_dropout_prob}")
        except Exception as e:
            print(f"[dropout] no pudo aplicarse ({e})")

    # Gradient checkpointing
    if bool(m.get("gradient_checkpointing", False)):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # torch.compile (opcional)
    torch_compile_flag = bool(m.get("torch_compile", False))
    if torch_compile_flag and hasattr(torch, "compile"):
        backend = m.get("torch_compile_backend", "inductor")
        mode = m.get("torch_compile_mode", "reduce-overhead")
        try:
            model = torch.compile(model, backend=str(backend), mode=str(mode), fullgraph=False)
            print(f"[torch.compile] enabled backend={backend} mode={mode}")
        except Exception as e:
            print(f"[torch.compile] disabled ({e})")

    # --------- TrainingArguments --------- #
    out_dir = args["model_out"] if isinstance(args, dict) else args.model_out
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    train_bs = int(m.get("batch", 64))
    eval_bs = int(m.get("eval_batch", train_bs))
    grad_accum = int(m.get("grad_accum", 1))
    epochs = float(m.get("epochs", 3))
    lr = float(m.get("lr", 2e-5))
    weight_decay = float(m.get("weight_decay", 0.01))
    warmup_ratio = float(m.get("warmup_ratio", 0.0))
    seed = int(m.get("seed", 42))
    optim = str(m.get("optim", "adamw_torch_fused"))
    eval_strategy = str(m.get("eval_strategy", "epoch")).lower()
    save_strategy = str(m.get("save_strategy", "epoch")).lower()
    logging_strategy = str(m.get("logging_strategy", "epoch")).lower()
    logging_steps = int(m.get("logging_steps", 50))
    save_total_limit = int(m.get("save_total_limit", 2))
    load_best_model_at_end = bool(m.get("load_best_model_at_end", True))
    metric_for_best_model = str(m.get("metric_for_best_model", "f1_macro"))
    greater_is_better = bool(m.get("greater_is_better", True))

    dl_num_workers = int(m.get("dataloader_num_workers", 8))
    dl_prefetch = int(m.get("dataloader_prefetch_factor", 2))
    dl_pin = bool(m.get("dataloader_pin_memory", True))
    dl_persist = bool(m.get("dataloader_persistent_workers", True))
    fp16 = bool(m.get("fp16", False))
    bf16 = bool(m.get("bf16", True))
    prefer_cuda = bool(m.get("prefer_cuda", True))
    max_grad_norm = m.get("max_grad_norm", None)

    # Collator
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    # arma kwargs y pásalos por compat layer
    targs_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        seed=seed,
        optim=optim,
        **({"evaluation_strategy": eval_strategy} if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters else {"eval_strategy": eval_strategy}),
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        dataloader_num_workers=dl_num_workers,
        dataloader_pin_memory=dl_pin,
        dataloader_persistent_workers=dl_persist,
        fp16=fp16,
        bf16=bf16,
        report_to=[] if not (run and MLFLOW_AVAILABLE) else ["mlflow"],
        disable_tqdm=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    if hasattr(TrainingArguments, "dataloader_prefetch_factor"):
        targs_kwargs["dataloader_prefetch_factor"] = dl_prefetch
    if max_grad_norm is not None and "max_grad_norm" in inspect.signature(TrainingArguments.__init__).parameters:
        targs_kwargs["max_grad_norm"] = float(max_grad_norm)

    targs = build_training_args_compat(targs_kwargs)

    # log args efectivos a MLflow
    if run and MLFLOW_AVAILABLE:
        try:
            mlflow.log_dict(asdict(targs), "training_args_effective.json")
        except Exception:
            try:
                mlflow.log_dict({k: getattr(targs, k) for k in targs.__dict__ if not k.startswith("_")},
                                "training_args_effective.json")
            except Exception:
                pass

    # --------- Instancia Trainer --------- #
    device = torch.device("cuda" if torch.cuda.is_available() and prefer_cuda else "cpu")
    class_w_tensor = torch.tensor(class_w, dtype=torch.float32, device=device)

    # focal / logit adjustment
    focal_cfg = (tr.get("focal") or {})
    focal_enable = bool(focal_cfg.get("enable", False))
    focal_gamma = float(focal_cfg.get("gamma", 1.5))
    focal_alpha_list = focal_cfg.get("alpha", None)
    focal_alpha_tensor = None
    if focal_enable and focal_alpha_list:
        if len(focal_alpha_list) != num_labels:
            raise ValueError(f"focal_alpha debe tener longitud {num_labels}.")
        focal_alpha_tensor = torch.tensor([float(x) for x in focal_alpha_list], dtype=torch.float32, device=device)

    laj_cfg = (tr.get("logit_adjustment") or {})
    enable_logit_adjustment = bool(laj_cfg.get("enable", False))
    logit_adjustment_tau = float(laj_cfg.get("tau", 1.0))

    label_smoothing_factor = float(m.get("label_smoothing_factor", 0.0))

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,  # deprec en v5; ok en 4.x (tu versión actual)
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_w_tensor,
        label_smoothing_factor=label_smoothing_factor,
        focal_enable=focal_enable,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha_tensor,
        enable_logit_adjustment=enable_logit_adjustment,
        logit_adjustment_tau=logit_adjustment_tau,
    )

    # Early stopping si patience > 0
    patience = int(m.get("patience", 0))
    if patience > 0:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    # --------- Entrenar --------- #
    train_result = trainer.train()
    trainer.save_model(out_dir)

    if run and MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifacts(out_dir, artifact_path="model")
        except Exception:
            pass

    # --------- Evaluación final --------- #
    eval_metrics = trainer.evaluate()
    print("[eval]", eval_metrics)
    if run and MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics({k: float(v) for k, v in eval_metrics.items() if np.isfinite(v)})
        except Exception:
            pass
        try:
            mlflow.end_run()
        except Exception:
            pass

if __name__ == "__main__":
    main()

