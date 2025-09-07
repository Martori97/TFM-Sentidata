#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia masiva para ALBERT (3 clases). Lee un parquet o carpeta con parquet,
predice en batches y guarda un parquet con:
- id (review_id o generado)
- pred_index (0=neg,1=neu,2=pos)
- pred_label (si el checkpoint tiene id2label)
- p_<clase> (probabilidades)
- true_label (0/1/2) si hay rating o label3 en los datos

Uso típico (como en DVC):
  python scripts/training/infer_albert_all.py \
    --model_dir models/albert_subset_0_250 \
    --input_parquet data/trusted/sephora_clean/reviews_0_250 \
    --output_parquet reports/albert_subset_all/reviews_0_250_preds.parquet \
    --text_col review_text --id_col review_id --max_len 128 --batch 256
"""
import os, argparse, glob, warnings
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# Silenciar avisos ruidosos
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=str)
    ap.add_argument("--input_parquet", required=True, type=str, help="Archivo .parquet o carpeta con .parquet")
    ap.add_argument("--output_parquet", required=True, type=str)
    ap.add_argument("--text_col", default="review_text", type=str)
    ap.add_argument("--id_col", default="review_id", type=str)
    ap.add_argument("--max_len", default=128, type=int)
    ap.add_argument("--batch", default=256, type=int)
    ap.add_argument("--num_workers", default=0, type=int)  # compatibilidad (no se usa aquí)
    return ap.parse_args()


def load_parquet_or_dir(path_like: str) -> pd.DataFrame:
    assert os.path.exists(path_like), f"No existe: {path_like}"
    if os.path.isdir(path_like):
        parts = sorted(glob.glob(os.path.join(path_like, "*.parquet")))
        assert parts, f"No se hallaron .parquet en {path_like}"
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return pd.read_parquet(path_like)


def ensure_cols(df: pd.DataFrame, text_col: str, id_col: str):
    # Texto
    if text_col not in df.columns:
        if "text" in df.columns:
            text_col = "text"
        else:
            raise ValueError(f"No encuentro columna de texto: '{text_col}' ni 'text'")
    # ID
    if id_col not in df.columns:
        gen_id = pd.util.hash_pandas_object(df[text_col].astype(str), index=False).astype(str)
        df = df.copy()
        df["gen_id"] = gen_id
        id_col = "gen_id"
    return df, text_col, id_col


def derive_true_label(df: pd.DataFrame):
    """Devuelve np.int32 (0/1/2) o None si no se puede derivar."""
    if "label3" in df.columns:
        return df["label3"].astype("int32").to_numpy()
    if "rating" in df.columns:
        r = df["rating"].astype(int).clip(1, 5).to_numpy()
        lab3 = np.where(r <= 2, 0, np.where(r == 3, 1, 2))
        return lab3.astype("int32")
    return None


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

    # 1) Cargar datos (archivo o carpeta)
    df = load_parquet_or_dir(args.input_parquet)
    if df.empty:
        raise SystemExit("Input vacío")

    # 2) Asegurar columnas de texto e id
    df, text_col, id_col = ensure_cols(df, args.text_col, args.id_col)

    # 3) Etiqueta verdadera (opcional) en 0/1/2
    true_label = derive_true_label(df)  # np.int32 o None

    # 4) Modelo + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"[DEVICE] {device}")

    # Nombres de clases/probabilidades
    if getattr(model.config, "id2label", None):
        try:
            id2label = {int(k): v for k, v in model.config.id2label.items()}
        except Exception:
            id2label = model.config.id2label
        class_names = [id2label[i] for i in range(len(id2label))]
    else:
        ncls = getattr(model.config, "num_labels", 3)
        class_names = ["neg", "neu", "pos"] if ncls == 3 else [f"class_{i}" for i in range(ncls)]

    # 5) Textos/ids
    texts = df[text_col].astype(str).tolist()
    ids   = df[id_col].astype(str).tolist()
    N = len(texts)

    # 6) Inferencia batcheada
    probs_list = []
    for i in tqdm(range(0, N, args.batch)):
        batch_texts = texts[i:i+args.batch]
        enc = tokenizer(
            batch_texts,
            truncation=True, padding="max_length",
            max_length=args.max_len, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).detach().cpu().to(torch.float32).numpy()
        probs_list.append(p)

    probs = np.vstack(probs_list).astype("float32")
    pred_index = probs.argmax(axis=1).astype("int32")

    # 7) DataFrame de salida con tipos "planos"
    out = {
        "id": [str(x) for x in ids],
        "pred_index": pred_index,
    }

    # Etiqueta legible (si el checkpoint tiene nombres)
    if len(class_names) == probs.shape[1]:
        out["pred_label"] = [class_names[int(i)] for i in pred_index]

    # Probabilidades por clase
    for j, cname in enumerate(class_names):
        safe = cname.lower().replace(" ", "_")
        out[f"p_{safe}"] = probs[:, j]

    # Etiqueta verdadera si la tenemos
    if true_label is not None:
        out["true_label"] = true_label  # ya es np.int32

    out_df = pd.DataFrame(out)
    out_df.to_parquet(args.output_parquet, index=False, engine="pyarrow")
    print(f"✅ Guardado: {args.output_parquet}  (filas={len(out_df)})")


if __name__ == "__main__":
    main()
