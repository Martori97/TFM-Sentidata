#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evalúa un modelo ALBERT ya entrenado usando el test guardado por train o derivado:
- Lee model_dir (modelo + tokenizer)
- Lee test.parquet (idealmente: review_id, review_text, label)
  * Si no hay 'label', la deriva de 'label3' o de 'rating'
  * Si no hay 'review_text', usa 'text'
  * Si no hay 'review_id', lo genera a partir del texto
- Guarda predictions.parquet, metrics.json y confusion.png en --output_dir (o model_dir)

Uso:
  python scripts/training/evaluate_albert.py \
    --model_dir models/albert_subset_0_250 \
    --test_parquet reports/albert_subset_0_250/test.parquet \
    --max_len 128 \
    --output_dir reports/albert_subset_0_250
"""
import os, sys, json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AlbertForSequenceClassification

CLASSES = ["negative", "neutral", "positive"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, type=str)
    p.add_argument("--test_parquet", default=None, type=str,
                   help="Si no se pasa, usa <model_dir>/test.parquet")
    p.add_argument("--output_dir", default=None, type=str,
                   help="Si no se pasa, usa --model_dir")
    p.add_argument("--max_len", default=128, type=int)
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
            # 0/1/2 -> strings
            return df.assign(label=[CLASSES[int(x)] for x in col.astype(int)])
        else:
            # strings: normaliza
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
    # Sin 'label': intenta label3 o rating
    if "label3" in df.columns:
        lab3 = df["label3"].astype(int).to_numpy()
    elif "rating" in df.columns:
        lab3 = _to_label3_from_rating(df["rating"])
        df = df.assign(label3=lab3)
    else:
        raise ValueError("El test no tiene 'label', 'label3' ni 'rating' para derivarlo.")
    return df.assign(label=[CLASSES[int(x)] for x in lab3])

def _ensure_text_and_id(df: pd.DataFrame) -> pd.DataFrame:
    # Texto
    if "review_text" not in df.columns:
        if "text" in df.columns:
            df = df.assign(review_text=df["text"])
        else:
            raise ValueError("No encuentro columna de texto ('review_text' ni 'text').")
    # ID
    if "review_id" not in df.columns:
        gen_id = pd.util.hash_pandas_object(df["review_text"].astype(str), index=False).astype(str)
        df = df.assign(review_id=gen_id)
    return df

def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, path_png: str):
    labels = ["neg", "neu", "pos"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.title("Confusion Matrix (test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path_png, dpi=140)
    plt.close()

def main():
    args = parse_args()
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

    # Tokenizar por lotes para no gastar RAM
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
    prec, rec, f1, _ = precision_recall_fscore_support(y_te_idx, preds, average="macro", zero_division=0)
    f1_per_class = precision_recall_fscore_support(y_te_idx, preds, average=None, zero_division=0)[2]
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
    }
    for i, cls in enumerate(CLASSES):
        metrics[f"f1_{cls}"] = float(f1_per_class[i])

    # Guardar outputs
    pred_path = os.path.join(out_dir, "predictions.parquet")
    pd.DataFrame({
        "review_id": id_te,
        "true_label": [CLASSES[i] for i in y_te_idx],
        "pred_3": [CLASSES[i] for i in preds],
        "p_neg": probs[:, LABEL2ID["negative"]],
        "p_neu": probs[:, LABEL2ID["neutral"]],
        "p_pos": probs[:, LABEL2ID["positive"]],
        "model": "albert"
    }).to_parquet(pred_path, index=False)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Matriz de confusión
    _save_confusion(y_te_idx, preds, os.path.join(out_dir, "confusion.png"))

    print(f"[out] metrics: {metrics}")
    print(f"[out] predictions -> {pred_path}")

if __name__ == "__main__":
    main()
