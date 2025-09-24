#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from transformers import AutoTokenizer, AlbertForSequenceClassification

# === MLflow (opcional) ===
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

CLASSES = ["negative", "neutral", "positive"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, type=str)
    p.add_argument("--test_parquet", default=None, type=str,
                   help="Si no se pasa, usa <model_dir>/test.parquet")
    p.add_argument("--output_dir", default=None, type=str,
                   help="Si no se pasa, usa --model_dir")
    p.add_argument("--max_len", default=128, type=int)

    # --- MLflow ---
    p.add_argument("--mlflow", type=str, default="false",
                   help="true|false para activar MLflow tracking")
    p.add_argument("--experiment", type=str, default="Sentidata",
                   help="Nombre del experimento en MLflow")
    p.add_argument("--run_name", type=str, default=None,
                   help="Nombre del run en MLflow")
    return p.parse_args()


def tokenize_texts(tokenizer, texts: List[str], max_len: int):
    return tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=max_len, return_tensors="pt"
    )


def _to_label3_from_rating(sr: pd.Series) -> np.ndarray:
    r = sr.astype(int).clip(1, 5).to_numpy()
    return np.where(r <= 2, 0, np.where(r == 3, 1, 2))


def _coerce_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura df['label'] como strings negative/neutral/positive."""
    if "label" in df.columns:
        col = df["label"]
        if pd.api.types.is_numeric_dtype(col):
            return df.assign(label=[CLASSES[int(x)] for x in col.astype(int)])
        else:
            s = col.astype(str).str.strip().str.lower()
            mapping = {
                "neg": "negative", "negative": "negative", "0": "negative",
                "neu": "neutral",  "neutral":  "neutral",  "1": "neutral",
                "pos": "positive", "positive": "positive", "2": "positive",
            }
            vals = s.map(mapping)
            if vals.isna().any():
                raise ValueError("Valores no reconocidos en 'label'. Usa neg/neu/pos o negative/neutral/positive o 0/1/2.")
            return df.assign(label=vals.values)

    if "label3" in df.columns:
        lab3 = df["label3"].astype(int).to_numpy()
    elif "rating" in df.columns:
        lab3 = _to_label3_from_rating(df["rating"])
        df = df.assign(label3=lab3)
    else:
        raise ValueError("El test no tiene 'label', 'label3' ni 'rating' para derivarlo.")
    return df.assign(label=[CLASSES[int(x)] for x in lab3])


def _ensure_text_and_id(df: pd.DataFrame) -> pd.DataFrame:
    if "review_text" not in df.columns:
        if "text" in df.columns:
            df = df.assign(review_text=df["text"])
        else:
            raise ValueError("No encuentro columna de texto ('review_text' ni 'text').")

    if "review_id" not in df.columns:
        gen_id = pd.util.hash_pandas_object(
            df["review_text"].astype(str), index=False
        ).astype(str)
        df = df.assign(review_id=gen_id)
    return df


def _save_confusions(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    labels_short = ["neg", "neu", "pos"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1.0)

    # PNG (counts)
    fig1 = plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(3), labels_short)
    plt.yticks(range(3), labels_short)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.title("Confusion Matrix (test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    png_counts = os.path.join(out_dir, "confusion.png")
    plt.savefig(png_counts, dpi=140)
    plt.close(fig1)

    # PNG (normalized)
    fig2 = plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.xticks(range(3), labels_short)
    plt.yticks(range(3), labels_short)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    plt.title("Confusion Matrix (test) - normalized")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    png_norm = os.path.join(out_dir, "confusion_normalized.png")
    plt.savefig(png_norm, dpi=140)
    plt.close(fig2)

    # CSVs
    csv_counts = os.path.join(out_dir, "confusion.csv")
    csv_norm = os.path.join(out_dir, "confusion_normalized.csv")
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(csv_counts)
    pd.DataFrame(cm_norm, index=CLASSES, columns=CLASSES).to_csv(csv_norm)

    return {
        "png_counts": png_counts,
        "png_norm": png_norm,
        "csv_counts": csv_counts,
        "csv_norm": csv_norm,
    }


def _maybe_mlflow_start(enable: bool, experiment: str, run_name: str | None):
    if not enable or not _HAS_MLFLOW:
        return None
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


def _maybe_mlflow_log_params(enable: bool, params: Dict):
    if enable and _HAS_MLFLOW:
        mlflow.log_params(params)


def _maybe_mlflow_log_metrics(enable: bool, metrics: Dict):
    if enable and _HAS_MLFLOW:
        mlflow.log_metrics(metrics)


def _maybe_mlflow_log_artifact(enable: bool, path: str):
    if enable and _HAS_MLFLOW and os.path.exists(path):
        mlflow.log_artifact(path)


def main():
    args = parse_args()
    use_mlflow = str(args.mlflow).strip().lower() in {"1", "true", "yes", "y"}
    out_dir = args.output_dir or args.model_dir
    os.makedirs(out_dir, exist_ok=True)

    # Cargar test
    test_path = args.test_parquet or os.path.join(args.model_dir, "test.parquet")
    if not os.path.exists(test_path):
        raise SystemExit(f"No encuentro test.parquet en {test_path}.")
    if os.path.isdir(test_path):
        parts = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith(".parquet")]
        assert parts, f"Sin .parquet en {test_path}"
        test_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        test_df = pd.read_parquet(test_path)
    if test_df.empty:
        raise SystemExit("test.parquet está vacío")
    print(f"[eval] test rows={len(test_df)}")

    # Normalizar columnas esperadas
    test_df = _ensure_text_and_id(test_df)
    test_df = _coerce_label_column(test_df)

    # Modelo + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AlbertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    X_te = test_df["review_text"].astype(str).tolist()
    y_te_idx = np.array([LABEL2ID[s] for s in test_df["label"].astype(str)])
    id_te = test_df["review_id"].astype(str).tolist()

    # Evaluación (tokenización por lotes)
    batch = 256
    all_probs = []
    for i in range(0, len(X_te), batch):
        batch_texts = X_te[i:i+batch]
        enc = tokenize_texts(tokenizer, batch_texts, args.max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    probs = np.vstack(all_probs)
    preds = probs.argmax(axis=-1)

    # Métricas
    acc = accuracy_score(y_te_idx, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te_idx, preds, average="macro", zero_division=0
    )
    prc_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(
        y_te_idx, preds, average=None, zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "precision_negative": float(prc_cls[0]),
        "recall_negative": float(rec_cls[0]),
        "f1_negative": float(f1_cls[0]),
        "precision_neutral": float(prc_cls[1]),
        "recall_neutral": float(rec_cls[1]),
        "f1_neutral": float(f1_cls[1]),
        "precision_positive": float(prc_cls[2]),
        "recall_positive": float(rec_cls[2]),
        "f1_positive": float(f1_cls[2]),
    }

    # Guardar predicciones
    pred_path = os.path.join(out_dir, "predictions.parquet")
    pd.DataFrame({
        "review_id": id_te,
        "true_label": [CLASSES[i] for i in y_te_idx],
        "pred_3": [CLASSES[i] for i in preds],
        "p_neg": probs[:, LABEL2ID["negative"]],
        "p_neu": probs[:, LABEL2ID["neutral"]],
        "p_pos": probs[:, LABEL2ID["positive"]],
        "model": "albert",
    }).to_parquet(pred_path, index=False)

    # Guardar métricas JSON
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusiones (count + normalized) en PNG y CSV
    cm_paths = _save_confusions(y_te_idx, preds, out_dir)

    print(f"[out] metrics: {metrics}")
    print(f"[out] predictions -> {pred_path}")

    # === MLflow logging ===
    run = _maybe_mlflow_start(
        enable=use_mlflow,
        experiment=args.experiment,
        run_name=args.run_name or f"eval::{os.path.basename(args.model_dir)}",
    )
    try:
        _maybe_mlflow_log_params(use_mlflow, {
            "phase": "evaluation",
            "model_dir": args.model_dir,
            "max_len": args.max_len,
            "test_rows": int(len(test_df)),
            "device": device,
        })
        _maybe_mlflow_log_metrics(use_mlflow, metrics)
        # artefactos
        for p in [pred_path, metrics_path, cm_paths["png_counts"], cm_paths["png_norm"],
                  cm_paths["csv_counts"], cm_paths["csv_norm"]]:
            _maybe_mlflow_log_artifact(use_mlflow, p)
    finally:
        if run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()

