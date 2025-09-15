#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json, warnings
from typing import Optional, Tuple, List
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# HF / Torch
import torch
from transformers import AutoTokenizer, AlbertForSequenceClassification

# Métricas / plots
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt

# YAML params
try:
    import yaml
except Exception:
    yaml = None


# ---------- Utilidades ----------
DEF_TEXT_COL = "review_text"
DEF_ID_COL   = "review_id"
POSSIBLE_LABEL_COLS = ["label", "label3", "target"]

CLASS_ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}

def load_params(path: str = "params.yaml") -> dict:
    if yaml and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.splitext(p)[1] else os.makedirs(p, exist_ok=True)

def map_rating_to_label3(r: Optional[float]) -> Optional[int]:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return None
    try:
        r = int(r)
    except Exception:
        return None
    if r in (1, 2): return 0   # negative
    if r == 3:     return 1    # neutral
    if r in (4, 5):return 2    # positive
    return None

def pick_label_series(df: pd.DataFrame) -> Tuple[Optional[pd.Series], str]:
    # intenta encontrar label directo
    for c in POSSIBLE_LABEL_COLS:
        if c in df.columns:
            return df[c], c
    # si no, intenta derivar desde rating
    if "rating" in df.columns:
        return df["rating"].map(map_rating_to_label3), "rating→label3"
    return None, ""

def plot_confusion(cm: np.ndarray, labels: List[str], out_png: str, normalize: bool = False):
    fig = plt.figure(figsize=(6, 5), dpi=140)
    data = cm.astype('float')
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        data = data / row_sums
    im = plt.imshow(data, interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=35)
    plt.yticks(tick_marks, labels)
    fmt = ".2f" if normalize else "d"
    thresh = data.max() / 2.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, format(data[i, j], fmt),
                     ha="center", va="center",
                     color="white" if data[i, j] > thresh else "black")
    title = "Confusion Matrix (normalized)" if normalize else "Confusion Matrix"
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    ensure_dir(out_png)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


# ---------- Inferencia + evaluación opcional ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directorio del modelo ALBERT guardado")
    ap.add_argument("--input_parquet", required=True, help="Ruta a parquet o carpeta con parquets")
    ap.add_argument("--output_parquet", required=True, help="Predicciones parquet")
    ap.add_argument("--text_col", default=DEF_TEXT_COL)
    ap.add_argument("--id_col", default=DEF_ID_COL)
    ap.add_argument("--max_len", type=int, default=None, help="Override de params.yaml")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--eval_output_dir", default=None, help="Si se pasa, escribe métricas/confusión aquí. Por defecto usa carpeta de output_parquet/infer_eval")
    ap.add_argument("--params_yaml", default="params.yaml")
    args = ap.parse_args()

    # Lee params.yaml (para max_len si no se pasó)
    params = load_params(args.params_yaml)
    if args.max_len is None:
        args.max_len = (
            params.get("model", {})
                  .get("albert", {})
                  .get("max_len", 192)
        )

    # Carga datos
    # - si es carpeta, lee todos los parquet dentro (partitioned dataset)
    if os.path.isdir(args.input_parquet):
        df = pd.read_parquet(args.input_parquet, engine="pyarrow")
    else:
        df = pd.read_parquet(args.input_parquet, engine="pyarrow")

    # Limpia NaNs en texto
    if args.text_col not in df.columns:
        raise ValueError(f"No existe la columna de texto '{args.text_col}' en el input.")
    texts = df[args.text_col].astype(str).fillna("").tolist()

    # Dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carga modelo/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AlbertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Inferencia en batches
    preds = []
    probs = []

    def iter_batches(seq, bs):
        for i in range(0, len(seq), bs):
            yield seq[i:i+bs]

    with torch.no_grad():
        for batch_texts in iter_batches(texts, args.batch_size):
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_preds = batch_probs.argmax(axis=1)
            preds.extend(batch_preds.tolist())
            probs.extend(batch_probs.tolist())

    # Construye dataframe de salida
    out = pd.DataFrame({
        args.id_col: df[args.id_col].values if args.id_col in df.columns else np.arange(len(df)),
        "pred_label_id": preds,
        "pred_label": [CLASS_ID_TO_NAME.get(i, f"class_{i}") for i in preds],
        "prob_neg": [p[0] for p in probs],
        "prob_neu": [p[1] for p in probs],
        "prob_pos": [p[2] for p in probs],
    })

    # (opcional) añade rating o labels originales si existían
    for c in [*POSSIBLE_LABEL_COLS, "rating"]:
        if c in df.columns and c not in out.columns:
            out[c] = df[c].values

    # Guarda predicciones
    ensure_dir(args.output_parquet)
    out.to_parquet(args.output_parquet, index=False)

    # ===== Evaluación si hay labels / rating =====
    # Busca columna de verdad-terreno o deriva desde rating
    y_true_series, src = pick_label_series(df)
    if y_true_series is not None:
        y_true = y_true_series.values
        # Deriva label3 desde rating si venía de ahí
        if src == "rating→label3":
            # ya viene mapeado en pick_label_series
            pass

        # Asegura enteros 0/1/2
        y_true = pd.Series(y_true).astype("float").round().astype("Int64")
        mask = y_true.notna()
        y_true = y_true[mask].astype(int)
        y_pred = out.loc[mask.index[mask], "pred_label_id"].astype(int)

        labels = [0, 1, 2]
        label_names = [CLASS_ID_TO_NAME[i] for i in labels]

        # Métricas
        acc = accuracy_score(y_true, y_pred)
        pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)

        cls_rep = classification_report(
            y_true, y_pred, labels=labels, target_names=label_names, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Salidas
        eval_dir = (
            args.eval_output_dir
            if args.eval_output_dir
            else os.path.join(os.path.dirname(args.output_parquet), "infer_eval")
        )
        ensure_dir(eval_dir)

        # metrics.json / csv
        metrics = {
            "source_of_truth": src if src else "explicit_label",
            "accuracy": float(acc),
            "precision_weighted": float(pr_w),
            "recall_weighted": float(rc_w),
            "f1_weighted": float(f1_w),
            "precision_macro": float(pr_m),
            "recall_macro": float(rc_m),
            "f1_macro": float(f1_m),
            "n_samples_eval": int(len(y_true)),
        }
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame([metrics]).to_csv(os.path.join(eval_dir, "metrics.csv"), index=False)

        # classification_report.csv
        pd.DataFrame(cls_rep).T.to_csv(os.path.join(eval_dir, "classification_report.csv"))

        # confusion matrices (png) + csv
        plot_confusion(cm, label_names, os.path.join(eval_dir, "confusion.png"), normalize=False)
        plot_confusion(cm, label_names, os.path.join(eval_dir, "confusion_normalized.png"), normalize=True)
        # también guardo los números
        pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(os.path.join(eval_dir, "confusion.csv"))

        print(f"[infer+eval] Done. Acc={acc:.4f} | eval_dir={eval_dir} | truth={src or 'label'}")
    else:
        print("[infer] No se encontraron columnas de verdad-terreno (label/label3/target ni rating). Solo se guardaron predicciones.")


if __name__ == "__main__":
    main()
