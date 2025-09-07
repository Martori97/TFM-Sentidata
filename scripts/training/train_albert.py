#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, warnings, argparse, inspect
from typing import Dict, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split

import torch
from torch import nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AlbertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# --- Spark / Delta (solo si --input_delta) ---
from pyspark.sql import SparkSession, functions as F
from delta import configure_spark_with_delta_pip

try:
    import yaml
except Exception:
    yaml = None


# =========================
# Utilidades
# =========================
def load_params(path: str = "params.yaml") -> Dict[str, Any]:
    if yaml and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

def select_device(prefer_cuda_idx: int = 0) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        idx = prefer_cuda_idx if 0 <= prefer_cuda_idx < n else 0
        print(f"[DEVICE] GPU {idx}: {torch.cuda.get_device_name(idx)}")
        return torch.device(f"cuda:{idx}")
    print("[DEVICE] CPU")
    return torch.device("cpu")

def rating_to_label3_int(r: int) -> int:
    # 1..2 -> 0 (neg), 3 -> 1 (neu), 4..5 -> 2 (pos)
    return 0 if r <= 2 else (1 if r == 3 else 2)

def to_hf(df: pd.DataFrame) -> Dataset:
    cols = ["review_id", "review_text", "label3"]
    return Dataset.from_pandas(df[cols].reset_index(drop=True), preserve_index=False)

def save_confusion_png(y_true: np.ndarray, y_pred: np.ndarray, path_png: str):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    labels = ["neg", "neu", "pos"]
    try:
        import seaborn as sns
        plt.figure(figsize=(5.2, 4.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
    except Exception:
        plt.figure(figsize=(5.2, 4.6))
        plt.imshow(cm, interpolation="nearest")
        plt.xticks(range(3), labels)
        plt.yticks(range(3), labels)
        for i in range(3):
            for j in range(3):
                plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.title("Confusion Matrix (val)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path_png, dpi=140)
    plt.close()


class WeightedTrainer(Trainer):
    """Trainer con CrossEntropy ponderada por clase (compatible HF 4.56)."""
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            loss = nn.functional.cross_entropy(logits, labels, weight=cw)
        else:
            loss = nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1m}


# =========================
# Carga de datos
# =========================
def load_from_delta(delta_path: str, min_len: int = 3) -> pd.DataFrame:
    builder = (
        SparkSession.builder
        .appName("ALBERT::train::load_trusted")
        .master("local[*]")
        .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read.format("delta").load(delta_path)
        .filter(F.col("review_text").isNotNull() & (F.length("review_text") > min_len))
        .withColumn("rating_int", F.col("rating").cast("int"))
        .filter(F.col("rating_int").isNotNull())
    )

    req = ["review_text", "rating"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        spark.stop()
        raise SystemExit(f"Faltan columnas en Delta: {miss}")

    if "review_id" not in df.columns:
        # fallback determinista
        df = df.withColumn("review_id", F.sha1(F.col("review_text").cast("string")))

    pdf = (
        df.select("review_id","review_text","rating_int")
          .withColumnRenamed("rating_int","rating")
          .dropDuplicates(["review_id"])
          .toPandas()
    )
    spark.stop()
    if pdf.empty:
        raise SystemExit(f"Sin datos válidos en {delta_path}")
    return pdf


def load_from_parquet_or_csv(path_like: str) -> pd.DataFrame:
    assert os.path.exists(path_like), f"No existe: {path_like}"
    if path_like.endswith(".parquet"):
        df = pd.read_parquet(path_like)
    elif path_like.endswith(".csv"):
        df = pd.read_csv(path_like)
    else:
        # directorio con *.parquet
        parts = [os.path.join(path_like, f) for f in os.listdir(path_like) if f.endswith(".parquet")]
        if not parts:
            raise SystemExit(f"No se hallaron parquet en {path_like}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return df


# =========================
# Main
# =========================
def main():
    # Silenciar telemetría
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_delta", type=str, default=None, help="Ruta Delta trusted (opcional).")
    ap.add_argument("--output_dir", type=str, default=None, help="Dónde guardar el modelo/preds.")
    ap.add_argument("--params", type=str, default="params.yaml", help="Ruta de params.yaml (opcional).")
    args = ap.parse_args()

    params = load_params(args.params)
    p = params.get("paths", {})
    m = params.get("model", {}).get("albert", {})

    # Semilla / device
    set_seed(int(m.get("seed", 42)))
    if hasattr(torch, "set_float32_matmul_precision"):
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    device = select_device(int(m.get("prefer_cuda", 0)))

    # -------------------------
    # Carga de datos
    # -------------------------
    if args.input_delta:
        df = load_from_delta(args.input_delta)
    else:
        sample_path = p.get("sample_sephora", "")
        df = load_from_parquet_or_csv(sample_path)

    # Limpiar / preparar
    assert {"review_text", "rating"}.issubset(df.columns), "Faltan review_text/rating"
    if "review_id" not in df.columns:
        df["review_id"] = pd.util.hash_pandas_object(
            df["review_text"].astype(str), index=False
        ).astype(str)

    df = df.dropna(subset=["review_text", "rating"]).copy()
    df["rating"] = df["rating"].astype(int).clip(1, 5)
    df["label3"] = df["rating"].apply(rating_to_label3_int).astype(int)

    # Split estratificado (85/15)
    y = df["label3"].values
    idx = np.arange(len(df))
    idx_tr, idx_val = train_test_split(
        idx, test_size=0.15, random_state=int(m.get("seed", 42)), stratify=y
    )
    df_tr = df.iloc[idx_tr].reset_index(drop=True)
    df_va = df.iloc[idx_val].reset_index(drop=True)
    print("TRAIN dist 0/1/2:", df_tr["label3"].value_counts().to_dict())
    print("VAL   dist 0/1/2:", df_va["label3"].value_counts().to_dict())

    # Datasets HF
    ds_tr = to_hf(df_tr)
    ds_va = to_hf(df_va)
    ds_all = to_hf(df)

    # ✅ guardar ids ANTES de tokenizar
    all_ids = ds_all["review_id"]

    # -------------------------
    # Tokenizador
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(m.get("model_name", "albert-base-v2"))
    max_len = int(m.get("max_len", 128))
    num_proc = int(m.get("tokenize_num_proc", max(1, (os.cpu_count() or 2)//2)))

    def tok_fn(batch):
        return tokenizer(
            batch["review_text"], truncation=True, max_length=max_len, padding=False
        )

    ds_tr = ds_tr.map(tok_fn, batched=True, num_proc=num_proc, remove_columns=["review_text"])
    ds_va = ds_va.map(tok_fn, batched=True, num_proc=num_proc, remove_columns=["review_text"])
    # ✅ eliminar también review_id para que no llegue al collator
    ds_all_tok = ds_all.map(tok_fn, batched=True, num_proc=num_proc, remove_columns=["review_text", "review_id"])

    # labels -> torch
    ds_tr = ds_tr.rename_column("label3", "labels")
    ds_va = ds_va.rename_column("label3", "labels")
    ds_all_tok = ds_all_tok.rename_column("label3", "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    ds_tr.set_format(type="torch", columns=cols)
    ds_va.set_format(type="torch", columns=cols)
    # ✅ fijar formato también en ds_all_tok
    ds_all_tok.set_format(type="torch", columns=cols)

    # -------------------------
    # Modelo
    # -------------------------
    model = AlbertForSequenceClassification.from_pretrained(
        m.get("model_name", "albert-base-v2"),
        num_labels=int(m.get("num_labels", 3))
    ).to(device)

    # Pesos por clase (inverso de frecuencia en train)
    ctr = df_tr["label3"].value_counts().to_dict()
    counts = np.array([ctr.get(0, 0), ctr.get(1, 0), ctr.get(2, 0)], dtype=np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    class_weights = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------------------------
    # TrainingArguments (compat 4.56)
    # -------------------------
    out_dir = args.output_dir or p.get("weights_dir", "models/albert")
    os.makedirs(out_dir, exist_ok=True)

    sig = inspect.signature(TrainingArguments.__init__).parameters
    kw = dict(
        output_dir=out_dir,
        per_device_train_batch_size=int(m.get("batch", 32)),
        per_device_eval_batch_size=int(m.get("batch", 32)),
        num_train_epochs=float(m.get("epochs", 3)),
        learning_rate=float(m.get("lr", 2e-5)),
        weight_decay=float(m.get("weight_decay", 0.01)),
        dataloader_num_workers=int(m.get("dataloader_num_workers", 0)),
        dataloader_prefetch_factor=int(m.get("dataloader_prefetch_factor", 2)),
        dataloader_pin_memory=bool(m.get("dataloader_pin_memory", True)),
        dataloader_persistent_workers=bool(m.get("dataloader_persistent_workers", False)),
        gradient_accumulation_steps=int(m.get("grad_accum", 1)),
        warmup_ratio=float(m.get("warmup_ratio", 0.0)),
        logging_strategy="steps",
        logging_steps=int(m.get("logging_steps", 200)),
        load_best_model_at_end=bool(m.get("load_best_model_at_end", True)),
        metric_for_best_model=str(m.get("metric_for_best_model", "eval_f1")),
        greater_is_better=bool(m.get("greater_is_better", True)),
        report_to="none",
        optim=str(m.get("optim", "adamw_torch_fused")),
        seed=int(m.get("seed", 42)),
        remove_unused_columns=False,   # evita batches vacíos
    )

    save_strategy = str(m.get("save_strategy", "epoch")).lower()
    eval_strategy = str(m.get("eval_strategy", save_strategy)).lower()
    if "eval_strategy" in sig:
        kw["eval_strategy"] = eval_strategy
    else:
        kw["evaluation_strategy"] = eval_strategy
    kw["save_strategy"] = save_strategy

    # steps solo si toca
    if save_strategy == "steps":
        st = int(m.get("save_steps", 1000))
        et = int(m.get("eval_steps", st))
        kw.update(save_steps=st, eval_steps=et)

    # precisión mixta
    if "bf16" in sig and bool(m.get("bf16", True)): kw["bf16"] = True
    elif "fp16" in sig and bool(m.get("fp16", False)): kw["fp16"] = True

    # compilación (opcional)
    if "torch_compile" in sig:
        kw["torch_compile"] = bool(m.get("torch_compile", False))
        if kw["torch_compile"] and "torch_compile_backend" in sig:
            kw["torch_compile_backend"] = str(m.get("torch_compile_backend", "inductor"))
        if kw["torch_compile"] and "torch_compile_mode" in sig:
            kw["torch_compile_mode"] = str(m.get("torch_compile_mode", "default"))

    args_tr = TrainingArguments(**kw)

    trainer = WeightedTrainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        data_collator=collator,
        tokenizer=tokenizer,
        class_weights=class_weights,
        compute_metrics=compute_metrics_fn,
    )

    patience = int(m.get("patience", 0))
    if patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    # Resume si hay checkpoint
    from transformers.trainer_utils import get_last_checkpoint
    last_ckpt = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    resume_from_last = bool(m.get("resume_from_last", True))
    if resume_from_last and last_ckpt is not None:
        print(f"[RESUME] Reanudando desde: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()

    # -------------------------
    # Evaluación + guardado
    # -------------------------
    eval_out = trainer.evaluate()
    preds = trainer.predict(ds_va)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    print("\n== Classification Report (val) ==")
    print(classification_report(y_true, y_pred, digits=3, target_names=["neg", "neu", "pos"]))

    metrics_path = os.path.join(out_dir, "metrics.json")
    eval_out = dict(eval_out)
    eval_out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    eval_out["accuracy"] = float(accuracy_score(y_true, y_pred))
    with open(metrics_path, "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"✅ Métricas JSON: {metrics_path}")

    confusion_png = os.path.join(out_dir, "confusion.png")
    save_confusion_png(y_true, y_pred, confusion_png)
    print(f"✅ Matriz de confusión: {confusion_png}")

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✅ Pesos guardados en: {out_dir}")

    # -------------------------
    # Inferencia sobre TODO el dataset (1..3)
    # -------------------------
    # all_preds = trainer.predict(ds_all_tok)
    # labels_0_2 = np.argmax(all_preds.predictions, axis=1)     # 0..2
    # pred_1_3 = (labels_0_2 + 1).astype(int)                   # 1..3

    # out_pred_path = os.path.join(out_dir, "predictions.parquet")
    # os.makedirs(os.path.dirname(out_pred_path), exist_ok=True)
    # import pyarrow as pa, pyarrow.parquet as pq
    # usar los IDs guardados (no leer de ds_all_tok)
    # pq.write_table(pa.Table.from_pandas(pd.DataFrame({
    #    "review_id": all_ids,
    #    "pred": pred_1_3
    # })), out_pred_path)
    # print(f" Predicciones (1..3) guardadas en: {out_pred_path}")


if __name__ == "__main__":
    main()
