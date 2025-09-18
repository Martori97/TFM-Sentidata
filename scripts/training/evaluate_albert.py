#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_recall_fscore_support)

import pyarrow.dataset as ds
from tqdm import tqdm

try:
    import yaml
except Exception:
    yaml = None

CLASS_ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}
POSSIBLE_LABEL_COLS = ["label3", "label", "target"]

DEFAULTS = {
    "text_col": "review_text_clean",
    "max_len": 192,
    "batch_size": 256,
    "stream_batch": 20000,
    "pad_to_multiple_of": 8,
    "bf16": True,
    "fp16": False,
    "enable_sdp_flash": True,
}

def load_params_yaml(path="params.yaml"):
    if not yaml or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception:
            return {}

def map_rating_to_label3(r):
    try:
        r = int(r)
    except Exception:
        return None
    if r in (1,2): return 0
    if r == 3: return 1
    if r in (4,5): return 2
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_parquet", required=True, help="Directorio/archivo Parquet con el split de test")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    # ---- params
    params = load_params_yaml("params.yaml")
    infer_cfg = (params.get("infer") or {})
    model_cfg = (params.get("model") or {}).get("albert", {})

    text_col = infer_cfg.get("text_col", model_cfg.get("text_col", DEFAULTS["text_col"]))
    max_len = int(infer_cfg.get("max_len", DEFAULTS["max_len"]))
    batch_size = int(infer_cfg.get("batch_size", model_cfg.get("eval_batch", DEFAULTS["batch_size"])))
    stream_batch = int(infer_cfg.get("stream_batch", DEFAULTS["stream_batch"]))
    pad_to_multiple_of = infer_cfg.get("pad_to_multiple_of", DEFAULTS["pad_to_multiple_of"])
    bf16 = bool(infer_cfg.get("bf16", DEFAULTS["bf16"]))
    fp16 = bool(infer_cfg.get("fp16", DEFAULTS["fp16"])) if not bf16 else False
    enable_sdp_flash = bool(infer_cfg.get("enable_sdp_flash", DEFAULTS["enable_sdp_flash"]))

    # ---- dataset
    if not os.path.exists(args.test_parquet):
        raise SystemExit(f"[error] No existe el path de test: {args.test_parquet}")

    dataset = ds.dataset(args.test_parquet, format="parquet")
    total_rows = dataset.count_rows()
    schema_cols = set(dataset.schema.names)

    print(f"[info] test rows={total_rows} | cols={sorted(schema_cols)}")
    if total_rows == 0:
        raise SystemExit("[error] El split de test tiene 0 filas. Revisa el stage de split o el filtro aplicado.")

    if text_col not in schema_cols:
        raise SystemExit(f"[error] No existe la columna de texto '{text_col}'. "
                         f"Columnas disponibles: {sorted(schema_cols)}")

    # label o rating
    label_col = None
    for c in POSSIBLE_LABEL_COLS:
        if c in schema_cols:
            label_col = c
            break
    if label_col is None and "rating" not in schema_cols:
        raise SystemExit("[error] No encuentro label (label3/label/target) ni 'rating' en el test.")

    # ---- modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    if enable_sdp_flash and device == "cuda":
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except Exception:
            pass

    # ---- eval
    os.makedirs(args.output_dir, exist_ok=True)
    y_true_all, y_pred_all = [], []

    def predict_batch(texts):
        preds = []
        ac_dtype = torch.bfloat16 if (device=="cuda" and bf16) else (torch.float16 if (device=="cuda" and fp16) else None)
        for i in range(0, len(texts), batch_size):
            enc = tok(
                texts[i:i+batch_size],
                padding=True, truncation=True, max_length=max_len,
                pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt"
            )
            enc = {k: v.to(device, non_blocking=True) for k,v in enc.items()}
            ctx = torch.autocast("cuda", dtype=ac_dtype) if ac_dtype is not None else torch.inference_mode()
            with torch.inference_mode(), ctx:
                logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).detach().cpu().numpy().tolist())
        return preds

    # tqdm con total de filas
    processed = 0
    scanner = dataset.scanner(columns=[col for col in [text_col, label_col, "rating"] if col is not None],
                              batch_size=stream_batch)
    with tqdm(total=total_rows, desc="Evaluate (rows)", unit="rows") as pbar:
        for record_batch in scanner.to_batches():
            df = record_batch.to_pandas()
            texts = df[text_col].astype(str).fillna("").tolist()
            preds = predict_batch(texts)

            # verdad
            if label_col is not None:
                truth = df[label_col].to_numpy()
            else:
                truth = np.array([map_rating_to_label3(x) for x in df["rating"].tolist()])

            m = pd.Series(truth).notna().to_numpy()
            y_true_all.extend(truth[m].astype(int).tolist())
            y_pred_all.extend(np.array(preds, dtype=int)[m].tolist())

            processed += len(df)
            pbar.update(len(df))

    if len(y_true_all) == 0:
        raise SystemExit("[error] No se encontraron etiquetas válidas en el test tras filtrar NAs.")

    # métricas
    y_true = np.array(y_true_all, dtype=int)
    y_pred = np.array(y_pred_all, dtype=int)

    labels_order = [0,1,2]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels_order, average=None, zero_division=0)

    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "per_class": {
            str(i): {
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int((y_true == i).sum())
            } for i in range(3)
        },
        "effective_params": {
            "text_col": text_col, "max_len": int(max_len),
            "batch_size": int(batch_size), "stream_batch": int(stream_batch),
            "pad_to_multiple_of": int(pad_to_multiple_of),
            "bf16": bool(bf16), "fp16": bool(fp16),
            "enable_sdp_flash": bool(enable_sdp_flash),
        }
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(cm, index=["negative","neutral","positive"],
                    columns=["negative","neutral","positive"]).to_csv(
        os.path.join(args.output_dir, "confusion.csv"), index=True
    )

    print(f"[evaluate] Done: n={len(y_true)} | acc={acc:.4f} | out={args.output_dir}")

if __name__ == "__main__":
    main()


