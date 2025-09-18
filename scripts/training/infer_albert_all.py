#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")  # evitar problemas de display en servidores/CLI
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

# --- YAML (params.yaml) ---
try:
    import yaml
except Exception:
    yaml = None

# ---------- Config por defecto ----------
DEF_TEXT_COL = "review_text"
DEF_ID_COL   = "review_id"
POSSIBLE_LABEL_COLS = ["label", "label3", "target"]
CLASS_ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}

DEFAULTS = {
    "max_len": 192,
    "batch_size": 256,      # inferencia → batch grande por defecto
    "stream_batch": 20000,  # tamaño de lectura en filas (CPU)
    "no_probs": False,
}

def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def map_rating_to_label3(r):
    try:
        r = int(r)
    except Exception:
        return None
    if r in (1, 2): return 0
    if r == 3:      return 1
    if r in (4, 5): return 2
    return None

def plot_confusion(cm: np.ndarray, labels, out_png: str, normalize: bool = False):
    fig = plt.figure(figsize=(6,5), dpi=140)
    if normalize:
        data = cm.astype(float)
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        data = data / row_sums
        fmt = ".2f"
    else:
        data = cm.astype(int)
        fmt = "d"

    im = plt.imshow(data, interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=35)
    plt.yticks(ticks, labels)
    thresh = (np.nanmax(data) / 2.0) if data.size else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            plt.text(j, i, format(v, fmt),
                     ha="center", va="center",
                     color="white" if v > thresh else "black")
    plt.title("Confusion Matrix (normalized)" if normalize else "Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    ensure_dir_for_file(out_png)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

def load_params_yaml(path="params.yaml"):
    if not yaml or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception:
            return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_parquet", required=True, help="Parquet file o carpeta (dataset) de PyArrow")
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--eval_output_dir", default=None)
    ap.add_argument("--text_col", default=DEF_TEXT_COL)
    ap.add_argument("--id_col", default=DEF_ID_COL)
    # None → para aplicar precedencia CLI > params.yaml > defaults
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None, help="Batch GPU (forward). 256 recomendado en inferencia.")
    ap.add_argument("--stream_batch", type=int, default=None, help="Tamaño de lote de lectura/stream (CPU).")
    ap.add_argument("--no_probs", action="store_true", help="No calcular probabilidades (softmax). Más rápido.")
    args = ap.parse_args()

    # ---------- Cargar params.yaml (si existe) ----------
    params = load_params_yaml("params.yaml")
    infer_cfg = (params.get("infer") or {})

    # ---------- Resolver hiperparámetros con precedencia ----------
    # CLI > params.yaml (infer.* o model.albert.*) > DEFAULTS
    max_len = args.max_len if args.max_len is not None else \
              infer_cfg.get("max_len", params.get("model", {}).get("albert", {}).get("max_len", DEFAULTS["max_len"]))
    batch_size = args.batch_size if args.batch_size is not None else \
                 infer_cfg.get("batch_size", DEFAULTS["batch_size"])
    stream_batch = args.stream_batch if args.stream_batch is not None else \
                   infer_cfg.get("stream_batch", DEFAULTS["stream_batch"])
    # no_probs: si viene flag CLI, manda; si no, params; si no, default
    no_probs = args.no_probs or bool(infer_cfg.get("no_probs", DEFAULTS["no_probs"]))

    # ---------- Modelo / Tokenizer ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device).eval()

    # ---------- Dataset Arrow (streaming) ----------
    dataset = ds.dataset(args.input_parquet, format="parquet")
    total_rows = dataset.count_rows()

    # columnas existentes en el schema
    schema_cols = set(dataset.schema.names)

    # columnas a leer desde disco (solo las que EXISTEN)
    read_cols = []
    # texto
    if args.text_col in schema_cols:
        read_cols.append(args.text_col)
    else:
        raise ValueError(f"No existe la columna de texto '{args.text_col}' en el input.")
    # id (si existe; si no, se sintetiza al vuelo)
    if args.id_col in schema_cols:
        read_cols.append(args.id_col)
    # verdad-terreno (solo si existen)
    for c in ["rating", *POSSIBLE_LABEL_COLS]:
        if c in schema_cols:
            read_cols.append(c)

    # ---------- Writer Parquet (append) ----------
    ensure_dir_for_file(args.output_parquet)
    writer = None  # pq.ParquetWriter
    wrote_rows = 0

    # ---------- Métricas incrementales (confusión) ----------
    have_truth_any = False
    labels_order = [0,1,2]
    cm_sum = np.zeros((3,3), dtype=np.int64)
    support_sum = np.zeros(3, dtype=np.int64)

    # util: escribir bloque al parquet de salida
    def write_block(df_out: pd.DataFrame):
        nonlocal writer, wrote_rows
        table = pa.Table.from_pandas(df_out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.output_parquet, table.schema)
        writer.write_table(table)
        wrote_rows += len(df_out)

    # util: avanzar GPU en mini-batches
    def predict_batch_texts(texts):
        preds_ids = []
        probs = [] if not no_probs else None
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tok(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            preds_ids.extend(pred.tolist())
            if not no_probs:
                p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                probs.extend(p.tolist())
        return preds_ids, probs

    # ---------- Stream principal ----------
    scanner = dataset.scanner(columns=read_cols, batch_size=stream_batch)
    pbar = tqdm(total=total_rows, desc="Infer (stream)", unit="rows")

    for record_batch in scanner.to_batches():
        # a pandas
        df = record_batch.to_pandas(types_mapper=pd.ArrowDtype)
        # sanea texto
        texts = df[args.text_col].astype(str).fillna("").tolist()

        # inferencia GPU
        pred_ids, prob_list = predict_batch_texts(texts)

        # construir salida del bloque
        out = {
            args.id_col: df[args.id_col].values if args.id_col in df.columns else np.arange(wrote_rows, wrote_rows+len(df)),
            "pred_label_id": np.array(pred_ids, dtype=np.int64),
            "pred_label": np.array([CLASS_ID_TO_NAME.get(i, f"class_{i}") for i in pred_ids], dtype=object),
        }
        if not no_probs and prob_list is not None:
            prob_arr = np.array(prob_list, dtype=np.float32)
            out["prob_neg"] = prob_arr[:,0]
            out["prob_neu"] = prob_arr[:,1]
            out["prob_pos"] = prob_arr[:,2]

        # añade rating/labels originales si existen (útil para auditoría)
        for c in [*POSSIBLE_LABEL_COLS, "rating"]:
            if c in df.columns:
                out[c] = df[c].values

        out_df = pd.DataFrame(out)
        write_block(out_df)

        # actualizar confusión incremental si hay verdad
        truth = None
        for c in POSSIBLE_LABEL_COLS:
            if c in df.columns:
                truth = df[c].to_numpy()
                break
        if truth is None and "rating" in df.columns:
            truth = np.array([map_rating_to_label3(x) for x in df["rating"].tolist()])

        if truth is not None:
            m = pd.Series(truth).notna().to_numpy()
            t = truth[m].astype(int)
            y = np.array(pred_ids, dtype=int)[m]
            cm_blk = confusion_matrix(t, y, labels=labels_order)
            cm_sum += cm_blk
            for i in labels_order:
                support_sum[i] += (t == i).sum()
            have_truth_any = True

        pbar.update(len(df))

    pbar.close()
    if writer is not None:
        writer.close()

    # ---------- Métricas y confusión (si hubo verdad) ----------
    if have_truth_any:
        eval_dir = args.eval_output_dir or os.path.join(os.path.dirname(args.output_parquet), "infer_eval")
        os.makedirs(eval_dir, exist_ok=True)

        tp = np.diag(cm_sum).astype(float)
        per_class_recall = np.divide(tp, cm_sum.sum(axis=1, keepdims=False), out=np.zeros_like(tp), where=cm_sum.sum(axis=1)!=0)
        per_class_precision = np.divide(tp, cm_sum.sum(axis=0, keepdims=False), out=np.zeros_like(tp), where=cm_sum.sum(axis=0)!=0)
        per_class_f1 = np.divide(2*per_class_precision*per_class_recall,
                                 per_class_precision+per_class_recall,
                                 out=np.zeros_like(tp),
                                 where=(per_class_precision+per_class_recall)!=0)

        macro_precision = float(np.nanmean(per_class_precision))
        macro_recall    = float(np.nanmean(per_class_recall))
        macro_f1        = float(np.nanmean(per_class_f1))
        accuracy        = float(tp.sum() / cm_sum.sum()) if cm_sum.sum() > 0 else 0.0

        weights = support_sum / support_sum.sum() if support_sum.sum() > 0 else np.array([0,0,0], dtype=float)
        weighted_precision = float(np.nansum(per_class_precision * weights))
        weighted_recall    = float(np.nansum(per_class_recall * weights))
        weighted_f1        = float(np.nansum(per_class_f1 * weights))

        label_names = [CLASS_ID_TO_NAME[i] for i in labels_order]

        metrics = {
            "accuracy": accuracy,
            "precision_macro": macro_precision,
            "recall_macro": macro_recall,
            "f1_macro": macro_f1,
            "precision_negative": float(per_class_precision[0]),
            "recall_negative": float(per_class_recall[0]),
            "f1_negative": float(per_class_f1[0]),
            "precision_neutral":  float(per_class_precision[1]),
            "recall_neutral":    float(per_class_recall[1]),
            "f1_neutral":        float(per_class_f1[1]),
            "precision_positive": float(per_class_precision[2]),
            "recall_positive":    float(per_class_recall[2]),
            "f1_positive":        float(per_class_f1[2]),
            "support_negative":   int(support_sum[0]),
            "support_neutral":    int(support_sum[1]),
            "support_positive":   int(support_sum[2]),
            "n_samples_eval":     int(support_sum.sum()),
            "effective_params": {
                "max_len": int(max_len),
                "batch_size": int(batch_size),
                "stream_batch": int(stream_batch),
                "no_probs": bool(no_probs),
                "text_col": args.text_col,
                "id_col": args.id_col,
            }
        }
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame([metrics]).to_csv(os.path.join(eval_dir, "metrics.csv"), index=False)

        pd.DataFrame(cm_sum, index=label_names, columns=label_names).to_csv(os.path.join(eval_dir, "confusion.csv"))
        plot_confusion(cm_sum, label_names, os.path.join(eval_dir, "confusion.png"), normalize=False)
        plot_confusion(cm_sum, label_names, os.path.join(eval_dir, "confusion_normalized.png"), normalize=True)

        print(f"[infer+eval] Done: wrote={wrote_rows} rows | acc={accuracy:.4f} | eval_dir={eval_dir}")
    else:
        print(f"[infer] Done: wrote={wrote_rows} rows (sin labels/rating -> no métricas)")

if __name__ == "__main__":
    main()



