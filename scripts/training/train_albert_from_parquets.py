#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import time
import argparse
import inspect
import warnings
from dataclasses import asdict
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import yaml
import transformers
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# MLflow (opcional según params)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# -------------------- Utils de configuración -------------------- #

def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def select_value(d: dict, key: str, default=None, cast=None):
    if d is None:
        return default
    if key not in d or d[key] is None:
        return default
    v = d[key]
    if cast is None:
        return v
    try:
        return cast(v)
    except Exception:
        return default

def get_last_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    ckpts = [p for p in glob.glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(p)]
    if not ckpts:
        return None
    def step_of(p):
        try:
            return int(p.split("-")[-1])
        except Exception:
            return -1
    ckpts.sort(key=lambda p: step_of(p), reverse=True)
    return ckpts[0]

def setup_mlflow(cfg: dict):
    if not (cfg and cfg.get("enable", True) and MLFLOW_AVAILABLE):
        return None
    mlflow.set_tracking_uri(cfg.get("tracking_uri", "file:./mlruns"))
    if cfg.get("registry_uri"):
        mlflow.set_registry_uri(cfg["registry_uri"])
    mlflow.set_experiment(cfg.get("experiment", "default"))
    run_name = os.getenv("MLFLOW_RUN_NAME", cfg.get("run_name"))
    run = mlflow.start_run(run_name=run_name)
    for k, v in (cfg.get("tags") or {}).items():
        try:
            mlflow.set_tag(k, v)
        except Exception:
            pass
    return run

# ---------------- TrainingArguments compat ---------------------- #

def build_training_args_compat(targs_kwargs: dict) -> TrainingArguments:
    """
    - Filtra kwargs no soportados por tu versión de transformers
    - Remapea optimizadores legacy
    - Fuerza coherencia eval/save si se puede
    - Si tu versión NO soporta evaluation_strategy, desactiva load_best_model_at_end y pone save='no'
    """
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    kwargs = dict(targs_kwargs)
    removed = {}

    eval_allowed = "evaluation_strategy" in allowed
    save_allowed = "save_strategy" in allowed

    # Mapeo evaluation/logging/save para versiones viejas
    if "evaluation_strategy" in kwargs and not eval_allowed:
        strat = str(kwargs.pop("evaluation_strategy")).lower()
        if strat in {"steps", "epoch"}:
            if "evaluate_during_training" in allowed:
                kwargs.setdefault("evaluate_during_training", True)
            if "eval_steps" in allowed:
                kwargs.setdefault("eval_steps", kwargs.get("logging_steps", 500))

    if "save_strategy" in kwargs and not save_allowed:
        strat = str(kwargs.pop("save_strategy")).lower()
        if "save_steps" in allowed and strat == "steps":
            kwargs.setdefault("save_steps", kwargs.get("save_steps", kwargs.get("logging_steps", 500)))

    if "logging_strategy" in kwargs and "logging_strategy" not in allowed:
        kwargs.pop("logging_strategy", None)

    # Remapeo de optimizadores legacy
    if "optim" in kwargs:
        opt_in = str(kwargs["optim"]).lower()
        map_opt = {
            "adamw_hf": "adamw_torch",
            "adamw": "adamw_torch",
            "adamw_hf_fused": "adamw_torch_fused",
            "adamw_fused": "adamw_torch_fused",
        }
        new_opt = map_opt.get(opt_in, opt_in)
        try:
            from transformers.training_args import OptimizerNames as HFOpt
            valid_opts = {m.value for m in HFOpt}
        except Exception:
            valid_opts = set()
        if valid_opts and new_opt not in valid_opts:
            warnings.warn(f"optim='{opt_in}' no soportado; usando 'adamw_torch'")
            new_opt = "adamw_torch"
        kwargs["optim"] = new_opt

    # Coherencia con best model
    if kwargs.get("load_best_model_at_end", False):
        if not eval_allowed:
            warnings.warn(
                "load_best_model_at_end=True pero tu TrainingArguments no soporta 'evaluation_strategy'. "
                "Se desactiva load_best_model_at_end y se establece save_strategy='no'."
            )
            kwargs["load_best_model_at_end"] = False
            if save_allowed:
                kwargs["save_strategy"] = "no"
        else:
            ev = str(kwargs.get("evaluation_strategy", "no")).lower()
            sv = str(kwargs.get("save_strategy", "no")).lower()
            if ev == "no" and sv in {"steps", "epoch"}:
                kwargs["evaluation_strategy"] = sv
            elif sv == "no" and ev in {"steps", "epoch"}:
                kwargs["save_strategy"] = ev

    # Quitar flags no soportadas
    for k in [
        "metric_for_best_model", "greater_is_better",
        "lr_scheduler_type", "bf16", "torch_compile", "report_to",
        "gradient_checkpointing", "dataloader_prefetch_factor",
        "dataloader_persistent_workers", "dataloader_pin_memory",
        "dataloader_num_workers", "warmup_ratio", "disable_tqdm",
        "remove_unused_columns", "label_names", "max_grad_norm",
    ]:
        if k in kwargs and k not in allowed:
            removed[k] = kwargs.pop(k)

    # Filtrar el resto desconocido
    final = {}
    for k, v in kwargs.items():
        if k in allowed:
            final[k] = v
        else:
            removed[k] = v

    if removed:
        warnings.warn(
            "TrainingArguments: ignoradas claves no soportadas por tu versión de transformers: "
            f"{sorted(removed.keys())}"
        )

    print(f"[hf] transformers={transformers.__version__}")
    print(f"[args] eval={final.get('evaluation_strategy','no')} "
          f"save={final.get('save_strategy','no')} "
          f"load_best={final.get('load_best_model_at_end')}")

    try:
        return TrainingArguments(**final)
    except ValueError as e:
        if "OptimizerNames" in str(e) or "optim" in str(e).lower():
            final["optim"] = "adamw_torch"
            warnings.warn("Fallback: estableciendo optim='adamw_torch' por incompatibilidad.")
            return TrainingArguments(**final)
        raise

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
            # datasets Arrow
            labels = np.array(list(self.train_dataset["labels"]))
        num_classes = logits.size(-1)
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64) + 1e-6
        priors = counts / counts.sum()
        lp = np.log(priors)
        self._log_prior = torch.tensor(lp, dtype=logits.dtype, device=logits.device)

    def _loss_ce(self, logits, labels):
        # CrossEntropy con pesos + label_smoothing
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.ls if self.ls > 0 else 0.0,
        )
        return loss_fct(logits, labels)

    def _loss_focal(self, logits, labels):
        # Focal Loss multi-clase con gamma y alpha por clase (y class_weights como multiplicador)
        # p_t = softmax(logits)[range, y]; focal = (1-p_t)^gamma * CE
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        ce = F.cross_entropy(logits, labels, reduction="none")  # sin pesos dentro (los aplicamos fuera)
        modulating = (1.0 - pt) ** self.focal_gamma

        # alpha por clase (prioridad a minoritarias)
        if self.focal_alpha is not None:
            alpha_w = self.focal_alpha.to(logits.device).gather(0, labels)
        else:
            alpha_w = torch.ones_like(pt)

        # class_weights (si existen) por etiqueta
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

        # Logit adjustment (Menon et al.) -> reduce sesgo a mayoritaria
        if self.enable_logit_adjustment:
            self._maybe_build_log_prior(logits)
            if self._log_prior is not None:
                logits = logits - self.logit_adjustment_tau * self._log_prior

        if labels is None:
            # pérdida del modelo si existiera
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
        else:
            if self.focal_enable:
                loss = self._loss_focal(logits, labels)
            else:
                loss = self._loss_ce(logits, labels)

        return (loss, outputs) if return_outputs else loss

# -------------------- Dataset y tokenización ------------------- #

def resolve_columns(ds_sample, requested_text: str | None, requested_label: str | None, requested_rating: str | None):
    cols = set(ds_sample.features.keys())
    text_col = requested_text if requested_text in cols else (
        "review_text" if "review_text" in cols else next((c for c in cols if "text" in c.lower()), None)
    )
    if text_col is None:
        raise ValueError("No se encontró columna de texto (e.g., 'review_text').")
    label_col = None
    for cand in [requested_label, "labels", "label3", "label", "target"]:
        if cand and cand in cols:
            label_col = cand
            break
    rating_col = requested_rating if requested_rating in cols else ("rating" if "rating" in cols else None)
    return text_col, label_col, rating_col

def build_label_if_needed(ds: DatasetDict, label_col: str | None, rating_col: str | None, star_to_label: dict | None):
    if label_col is not None:
        return ds, label_col
    if rating_col is None or not star_to_label:
        raise ValueError("No existe columna de etiqueta ni rating con mapeo 'star_to_label' para construirla.")
    label_map = {"NEG": 0, "NEU": 1, "POS": 2}
    star_to_id = {int(k): label_map[v] for k, v in star_to_label.items()}
    def _add_label(example):
        r = example[rating_col]
        try:
            r = int(r)
        except Exception:
            r = int(float(r))
        example["labels"] = int(star_to_id.get(r))
        return example
    ds2 = ds.map(_add_label)
    return ds2, "labels"

def tokenize_fn_builder(tokenizer, text_col: str, max_len: int):
    def _tok(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_len,
            padding=False
        )
    return _tok

def infer_num_labels(ds: DatasetDict, label_col: str) -> int:
    unique = set()
    for split in ds:
        arr = ds[split][label_col]
        unique.update(set(int(x) for x in arr))
    if not unique:
        return 3
    return int(max(unique) + 1)

def compute_class_weights(train_labels: np.ndarray, num_labels: int, mode: str = "auto", alpha: float = 0.6,
                          manual: Optional[List[float]] = None):
    if mode == "manual" and manual:
        w = np.array(manual, dtype=np.float32)
        if len(w) != num_labels:
            raise ValueError(f"class_weights.manual debe tener longitud {num_labels}.")
        return w
    counts = np.bincount(train_labels, minlength=num_labels).astype(np.float64)
    total = float(counts.sum())
    counts[counts == 0] = 1.0
    freq = counts / total
    inv = 1.0 / freq
    inv_norm = inv / inv.mean()
    w = alpha * inv_norm + (1.0 - alpha) * np.ones_like(inv_norm)
    return w.astype(np.float32)

# ------------------------------ MAIN ----------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Directorio parquet de train (carpeta con *.parquet)")
    parser.add_argument("--val", required=True, help="Directorio parquet de validación (carpeta con *.parquet)")
    parser.add_argument("--model-out", required=True, help="Carpeta de salida del modelo")
    parser.add_argument("--params", default="params.yaml", help="Ruta de params.yaml")
    args = parser.parse_args()

    out_dir = args.model_out
    os.makedirs(out_dir, exist_ok=True)

    params_all = load_params(args.params)
    mdl = select_value(params_all, "model", {})
    p = (mdl or {}).get("albert", {})

    # MLflow
    mlf_cfg = (params_all or {}).get("mlflow", {})
    run = setup_mlflow(mlf_cfg)

    # Hiperparámetros base
    model_name = select_value(p, "model_name", "albert-base-v2", str)
    max_len = select_value(p, "max_len", 128, int)
    num_labels_yaml = select_value(p, "num_labels", 3, int)

    train_bs = select_value(p, "batch", 16, int)
    eval_bs = select_value(p, "eval_batch", train_bs, int)
    grad_accum = select_value(p, "grad_accum", 1, int)
    epochs = select_value(p, "epochs", 3, int)
    lr = float(select_value(p, "lr", 2e-5, float))
    weight_decay = float(select_value(p, "weight_decay", 0.0, float))
    warmup_ratio = float(select_value(p, "warmup_ratio", 0.0, float))
    seed = select_value(p, "seed", 42, int)
    optim = select_value(p, "optim", "adamw_torch", str)

    prefer_cuda = bool(select_value(p, "prefer_cuda", 1, int))
    bf16 = bool(select_value(p, "bf16", False))
    fp16 = bool(select_value(p, "fp16", False))
    grad_ckpt = bool(select_value(p, "gradient_checkpointing", False))
    torch_compile_flag = bool(select_value(p, "torch_compile", False))
    torch_compile_backend = select_value(p, "torch_compile_backend", "inductor", str)
    torch_compile_mode = select_value(p, "torch_compile_mode", "reduce-overhead", str)

    dl_num_workers = select_value(p, "dataloader_num_workers", 2, int)
    dl_prefetch = select_value(p, "dataloader_prefetch_factor", 2, int)
    dl_pin = bool(select_value(p, "dataloader_pin_memory", True))
    dl_persist = bool(select_value(p, "dataloader_persistent_workers", True))

    save_strategy = select_value(p, "save_strategy", "no", str)
    eval_strategy = select_value(p, "eval_strategy", "no", str)
    logging_strategy = select_value(p, "logging_strategy", "steps", str)
    logging_steps = select_value(p, "logging_steps", 200, int)
    save_total_limit = select_value(p, "save_total_limit", 2, int)
    patience = select_value(p, "patience", 0, int)
    load_best_model_at_end = bool(select_value(p, "load_best_model_at_end", False))
    greater_is_better = bool(select_value(p, "greater_is_better", True))
    metric_for_best_model = select_value(p, "metric_for_best_model", "eval_f1_macro", str)
    resume_from_last = bool(select_value(p, "resume_from_last", False))

    max_grad_norm = select_value(p, "max_grad_norm", None, float)
    classifier_dropout_prob = select_value(p, "classifier_dropout_prob", None, float)
    label_smoothing_factor = select_value(p, "label_smoothing_factor", 0.0, float)

    # Mejoras: logit adjustment / focal
    enable_logit_adjustment = bool(select_value(p, "enable_logit_adjustment", False))
    logit_adjustment_tau = float(select_value(p, "logit_adjustment_tau", 1.0, float))

    focal_enable = bool(select_value(p, "focal_enable", False))
    focal_gamma = float(select_value(p, "focal_gamma", 1.5, float))
    focal_alpha_list = select_value(p, "focal_alpha", None)  # list o None

    text_col_req = select_value(p, "text_col", "review_text", str)
    label_col_req = select_value(p, "label_col", "label3", str)
    rating_col_req = select_value(p, "rating_col", "rating", str)

    cw_cfg = (params_all or {}).get("training", {}).get("class_weights", {})
    cw_mode = select_value(cw_cfg, "mode", "auto", str)
    cw_alpha = float(select_value(cw_cfg, "alpha", 0.6, float))
    cw_manual = select_value(cw_cfg, "manual", None)

    # Log de hiperparámetros (sin colisiones con HF → prefijo requested_)
    if run and MLFLOW_AVAILABLE:
        try:
            mlflow.log_params({
                "model_name": model_name,
                "max_len": max_len,
                "num_labels_yaml": num_labels_yaml,
                "batch": train_bs,
                "eval_batch": eval_bs,
                "grad_accum": grad_accum,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "optim": optim,
                "requested_bf16": bf16,
                "requested_fp16": fp16,
                "requested_gradient_checkpointing": grad_ckpt,
                "requested_torch_compile": torch_compile_flag,
                "requested_torch_compile_backend": torch_compile_backend,
                "requested_torch_compile_mode": torch_compile_mode,
                "requested_logging_strategy": logging_strategy,
                "requested_logging_steps": logging_steps,
                "requested_save_total_limit": save_total_limit,
                "patience": patience,
                "resume_from_last": resume_from_last,
                "text_col_req": text_col_req,
                "label_col_req": label_col_req,
                "rating_col_req": rating_col_req,
                "cw_mode": cw_mode,
                "cw_alpha": cw_alpha,
                "classifier_dropout_prob": classifier_dropout_prob,
                "label_smoothing_factor": label_smoothing_factor,
                "enable_logit_adjustment": enable_logit_adjustment,
                "logit_adjustment_tau": logit_adjustment_tau,
                "focal_enable": focal_enable,
                "focal_gamma": focal_gamma,
                "max_grad_norm_req": max_grad_norm,
            })
        except Exception:
            pass

    # ----------------- Carga datasets parquet ----------------- #
    train_pattern = os.path.join(args.train, "**", "*.parquet")
    val_pattern = os.path.join(args.val, "**", "*.parquet")
    data_files = {"train": train_pattern, "validation": val_pattern}
    ds: DatasetDict = load_dataset("parquet", data_files=data_files)

    # Resolver columnas y construir label si falta
    sample_ref = ds["train"]
    text_col, label_col, rating_col = resolve_columns(sample_ref, text_col_req, label_col_req, rating_col_req)

    star_to_label = None
    if label_col is None:
        eda_cfg = (params_all or {}).get("eda", {})
        star_to_label = eda_cfg.get("star_to_label")

    ds, label_col = build_label_if_needed(ds, label_col, rating_col, star_to_label)

    # Tokenizador + mapeo
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

    tok_fn = tokenize_fn_builder(tok, text_col=text_col, max_len=max_len)
    ds_tok = ds.map(tok_fn, batched=True, num_proc=select_value(p, "tokenize_num_proc", 6, int))

    keep_cols = ["input_ids", "attention_mask", label_col]
    if "token_type_ids" in ds_tok["train"].features:
        keep_cols.append("token_type_ids")

    ds_tok = DatasetDict({
        "train": ds_tok["train"].remove_columns([c for c in ds_tok["train"].column_names if c not in keep_cols]).rename_column(label_col, "labels"),
        "validation": ds_tok["validation"].remove_columns([c for c in ds_tok["validation"].column_names if c not in keep_cols]).rename_column(label_col, "labels"),
    })

    num_labels = infer_num_labels(ds_tok, "labels")
    if run and MLFLOW_AVAILABLE:
        try:
            mlflow.log_param("num_labels_inferred", num_labels)
        except Exception:
            pass

    # Pesos de clase
    y_train = np.array(ds_tok["train"]["labels"], dtype=np.int64)
    class_w = compute_class_weights(y_train, num_labels, mode=cw_mode, alpha=cw_alpha, manual=cw_manual)
    print(f"[class_weights] {cw_mode}(alpha={cw_alpha}) -> {class_w.tolist()}")
    if run and MLFLOW_AVAILABLE:
        try:
            mlflow.log_dict({"class_weights": [float(x) for x in class_w.tolist()]}, "class_weights.json")
        except Exception:
            pass

    # Collator (pad múltiplo de 8 para Tensor Cores)
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.use_cache = False

    # Ajuste de dropout de la cabeza (si existe)
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
    if grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # torch.compile
    if torch_compile_flag and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, backend=torch_compile_backend, mode=torch_compile_mode, fullgraph=False)
            print(f"[torch.compile] enabled backend={torch_compile_backend} mode={torch_compile_mode}")
        except Exception as e:
            print(f"[torch.compile] disabled ({e})")

    # TrainingArguments (compat)  + FIX torch.compile: no podar columnas
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
        evaluation_strategy=eval_strategy,
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

    # Log de estrategias/compilación efectivas (sin chocar con HF)
    if run and MLFLOW_AVAILABLE:
        try:
            mlflow.log_dict(asdict(targs), "training_args_effective.json")
        except Exception:
            try:
                mlflow.log_dict({k: getattr(targs, k) for k in targs.__dict__ if not k.startswith("_")},
                                "training_args_effective.json")
            except Exception:
                pass

        eff_eval = getattr(targs, "evaluation_strategy", getattr(targs, "eval_strategy", "no"))
        eff_eval = getattr(eff_eval, "value", str(eff_eval)).lower()
        eff_save = getattr(targs, "save_strategy", "no")
        eff_save = getattr(eff_save, "value", str(eff_save)).lower()
        eff_tc = bool(getattr(targs, "torch_compile", False)) if hasattr(targs, "torch_compile") else False
        eff_tc_backend = getattr(targs, "torch_compile_backend", None)
        eff_tc_backend = getattr(eff_tc_backend, "value", eff_tc_backend)
        eff_tc_mode = getattr(targs, "torch_compile_mode", None)

        try:
            mlflow.log_params({
                "effective_eval_strategy": eff_eval,
                "effective_save_strategy": eff_save,
                "effective_load_best_model_at_end": bool(getattr(targs, "load_best_model_at_end", False)),
                "effective_torch_compile": eff_tc,
                "effective_torch_compile_backend": str(eff_tc_backend),
                "effective_torch_compile_mode": str(eff_tc_mode),
            })
        except Exception:
            pass

    # Trainer
    device = torch.device("cuda" if torch.cuda.is_available() and prefer_cuda else "cpu")
    class_w_tensor = torch.tensor(class_w, dtype=torch.float32, device=device)

    focal_alpha_tensor = None
    if focal_enable and focal_alpha_list:
        if len(focal_alpha_list) != num_labels:
            raise ValueError(f"focal_alpha debe tener longitud {num_labels}.")
        focal_alpha_tensor = torch.tensor([float(x) for x in focal_alpha_list], dtype=torch.float32, device=device)

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,  # (deprec. v5, OK en 4.x)
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_w_tensor,
        label_smoothing_factor=float(label_smoothing_factor or 0.0),
        focal_enable=focal_enable,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha_tensor,
        enable_logit_adjustment=enable_logit_adjustment,
        logit_adjustment_tau=logit_adjustment_tau,
    )

    # EarlyStopping solo si eval en steps/epoch
    ev_attr = getattr(targs, "evaluation_strategy", getattr(targs, "eval_strategy", "no"))
    ev_val = getattr(ev_attr, "value", str(ev_attr)).lower()
    if patience and ev_val in {"steps", "epoch"}:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=int(patience)))
    else:
        print("[early_stopping] Desactivado (evaluation_strategy=no o patience=0).")

    # Entrenamiento (resume opcional)
    resume_ckpt = get_last_checkpoint(out_dir) if resume_from_last else None
    if resume_ckpt:
        print(f"[resume] Reanudando desde {resume_ckpt}")
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except TypeError:
        trainer.train()

    # Eval final
    eval_metrics = trainer.evaluate()
    eval_logged = {
        "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
        "eval_accuracy": float(eval_metrics.get("eval_accuracy", float("nan"))),
        "eval_f1_macro": float(eval_metrics.get("eval_f1_macro", float("nan"))),
        "eval_precision_macro": float(eval_metrics.get("eval_precision_macro", float("nan"))),
        "eval_recall_macro": float(eval_metrics.get("eval_recall_macro", float("nan"))),
    }
    print(f"[eval] {json.dumps(eval_logged, indent=2)}")

    # ------------------ GUARDADO SEGURO (sin _orig_mod) ------------------ #
    to_save = trainer.model._orig_mod if hasattr(trainer.model, "_orig_mod") else trainer.model
    to_save.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # Metadatos
    with open(os.path.join(out_dir, "params_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(params_all, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_path": args.train,
            "val_path": args.val,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "num_labels": num_labels,
            "text_col": text_col,
            "label_col": label_col,
            "max_len": max_len,
            "batch": train_bs,
            "grad_accum": grad_accum,
        }, f, ensure_ascii=False, indent=2)

    # MLflow métricas + modelo
    if run and MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics({
                "val_loss": eval_logged["eval_loss"],
                "val_accuracy": eval_logged["eval_accuracy"],
                "val_f1_macro": eval_logged["eval_f1_macro"],
                "val_precision_macro": eval_logged["eval_precision_macro"],
                "val_recall_macro": eval_logged["eval_recall_macro"],
            })
        except Exception:
            pass
        try:
            from mlflow import transformers as mlflow_transformers
            mlflow_transformers.log_model(
                transformers_model={"model": to_save, "tokenizer": tok},
                artifact_path="model",
                task="text-classification",
            )
        except Exception:
            try:
                from mlflow import pytorch as mlflow_pytorch
                mlflow_pytorch.log_model(to_save, artifact_path="pytorch-model")
            except Exception:
                pass
        try:
            mlflow.end_run()
        except Exception:
            pass

    print(f"✅ Modelo guardado en: {out_dir}")

if __name__ == "__main__":
    main()