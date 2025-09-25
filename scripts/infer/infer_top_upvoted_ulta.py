#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia ALBERT sobre Top-N reseñas por upvotes desde Delta (Ulta)
- Lee Delta: data/trusted/ulta_clean/Ulta Skincare Reviews
- Selecciona Top-N por upvotes (>0)
- Usa Review_Title + Review_Text como entrada (par de secuencias)
- Mantiene 'neutral' SOLO si p_neutral >= min_neutral_prob (por defecto 0.90)
- Escribe:
  * Parquet con predicciones
  * CSV con distribución de pred_labels
  * CSV con Top-N reseñas (marca, producto, upvotes, título, texto y probabilidades)
"""

import argparse
import os
import json
import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyspark.sql import SparkSession, functions as F, types as T

# Delta
from delta import configure_spark_with_delta_pip


def pick_device(verbose: bool = True):
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        if verbose:
            print(f"[device] Using {dev}: {name}")
    else:
        dev = torch.device("cpu")
        if verbose:
            print("[device] Using CPU")
    return dev


def softmax(logits, axis: int = -1):
    e = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def ensure_review_id(df, id_col: str):
    """Si no hay id_col, genera un hash determinista de título/texto+producto+marca."""
    if id_col in df.columns:
        return df

    def _mk_hash(*vals):
        base = "||".join([str(v) if v is not None else "" for v in vals])
        return hashlib.md5(base.encode("utf-8")).hexdigest()

    mk_hash = F.udf(_mk_hash, T.StringType())
    cols = [c for c in ["Review_Text", "Review_Title", "Product", "Brand"] if c in df.columns]
    if not cols:
        cols = [df.columns[0]]
    return df.withColumn(id_col, mk_hash(*map(F.col, cols)))


def build_spark(app_name: str = "infer_top_upvoted_ulta") -> SparkSession:
    """SparkSession configurada para Delta."""
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_delta",
                    default="data/trusted/ulta_clean/Ulta Skincare Reviews",
                    help="Ruta Delta de Ulta (con espacios).")
    ap.add_argument("--text_col", default="Review_Text",
                    help="Columna con el texto de la reseña.")
    ap.add_argument("--title_col", default="Review_Title",
                    help="Columna con el título de la reseña.")
    ap.add_argument("--use_title", type=int, default=1,
                    help="1=usar título+texto (par), 0=solo texto.")
    ap.add_argument("--upvotes_col", default="Review_Upvotes")
    ap.add_argument("--downvotes_col", default="Review_Downvotes")
    ap.add_argument("--id_col", default="review_id")
    ap.add_argument("--model_dir", default="models/albert_sample_30pct")
    ap.add_argument("--label_names", default='["negative","neutral","positive"]')
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output_parquet",
                    default="reports/albert_sample_30pct/infer_top50_upvoted_ulta/predictions.parquet")
    ap.add_argument("--coalesce", type=int, default=1)

    # Mantener 'neutral' solo si p_neutral >= min_neutral_prob
    ap.add_argument("--demote_neutral", type=int, default=1,
                    help="1=si p_neutral < min_neutral_prob, reasigna a neg/pos según mayor prob.")
    ap.add_argument("--min_neutral_prob", type=float, default=0.90,
                    help="Umbral mínimo de probabilidad para aceptar 'neutral' (por defecto 0.90).")

    args = ap.parse_args()

    # Spark con Delta
    spark = build_spark("infer_top_upvoted_ulta")

    # Leer Delta
    sdf = spark.read.format("delta").load(args.input_delta)

    # Tipos
    if args.upvotes_col not in sdf.columns:
        raise ValueError(f"No existe columna {args.upvotes_col} en el Delta de Ulta.")
    sdf = sdf.withColumn(args.upvotes_col, F.col(args.upvotes_col).cast("int"))
    if args.downvotes_col in sdf.columns:
        sdf = sdf.withColumn(args.downvotes_col, F.col(args.downvotes_col).cast("int"))

    # Asegurar ID
    sdf = ensure_review_id(sdf, args.id_col)

    # Selección Top-N por upvotes (>0)
    keep_extra = [c for c in ["Product", "Brand", "Verified_Buyer", "Review_Date"] if c in sdf.columns]

    # Columnas a extraer: id, upvotes, extras y textos
    select_cols = [args.id_col, args.upvotes_col] + keep_extra
    select_for_infer = select_cols.copy()
    if args.use_title and (args.title_col in sdf.columns):
        select_for_infer.append(args.title_col)
    select_for_infer.append(args.text_col)

    base = (
        sdf.filter(F.col(args.upvotes_col) > 0)
           .select(*select_for_infer)
           .orderBy(F.col(args.upvotes_col).desc())
           .limit(args.top_n)
    )

    pdf = base.toPandas()
    if pdf.empty:
        print("[warn] No hay reseñas con upvotes > 0 en Ulta. Nada que inferir.")
        spark.stop()
        return

    # Modelo
    device = pick_device()
    label_names = json.loads(args.label_names)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Entradas
    texts = pdf[args.text_col].fillna("").tolist()
    use_title = bool(args.use_title and (args.title_col in pdf.columns))
    if use_title:
        titles = pdf[args.title_col].fillna("").tolist()

    # Inferencia por lotes
    logits_all = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            if use_title:
                batch_titles = titles[i:i + args.batch_size]
                enc = tokenizer(
                    batch_titles,  # 1ª secuencia: título
                    batch_texts,   # 2ª secuencia: texto
                    padding=True,
                    truncation=True,
                    max_length=args.max_len,
                    return_tensors="pt"
                )
            else:
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_len,
                    return_tensors="pt"
                )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits_all.append(out.logits.detach().cpu().numpy())

    logits = np.vstack(logits_all)
    probs = softmax(logits)
    idx_neg, idx_neu, idx_pos = 0, 1, 2
    p_neg, p_neu, p_pos = probs[:, idx_neg], probs[:, idx_neu], probs[:, idx_pos]

    # Predicción base por argmax
    pred_idx = probs.argmax(axis=1)
    pred_label = np.array([label_names[i] for i in pred_idx], dtype=object)

    # Mantener 'neutral' SOLO si p_neutral ≥ min_neutral_prob
    if args.demote_neutral:
        mask_neu = (pred_label == "neutral") & (p_neu < args.min_neutral_prob)
        better_is_neg = p_neg >= p_pos
        pred_label[mask_neu & better_is_neg] = "negative"
        pred_label[mask_neu & (~better_is_neg)] = "positive"

    # Salida principal (Parquet)
    out = pdf.copy()
    out["pred_label"] = pred_label
    for j, name in enumerate(label_names):
        out[f"prob_{name}"] = probs[:, j]
        out[f"logit_{name}"] = logits[:, j]

    # Asegurar presencia de columnas de texto en la salida
    if args.title_col not in out.columns:
        out[args.title_col] = None
    if args.text_col not in out.columns:
        out[args.text_col] = None

    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    out_sdf = spark.createDataFrame(out).coalesce(args.coalesce)
    out_sdf.write.mode("overwrite").parquet(args.output_parquet)

    # Distribución y export CSV (pred_distribution_csv)
    print("[pred distribution]")
    dist = (spark.read.parquet(args.output_parquet)
            .groupBy("pred_label")
            .count()
            .orderBy("count", ascending=False))
    dist.show(truncate=False)

    dist_dir = os.path.dirname(args.output_parquet) + "/pred_distribution_csv"
    (dist.coalesce(1)
         .write.mode("overwrite")
         .option("header", True)
         .csv(dist_dir))

    # Export Top-N a CSV plano con TÍTULO y TEXTO SIEMPRE
    out_full = spark.read.parquet(args.output_parquet)

    # Renombrados robustos (independientes del nombre original)
    renamed = out_full
    if args.upvotes_col != "Review_Upvotes" and args.upvotes_col in renamed.columns:
        renamed = renamed.withColumnRenamed(args.upvotes_col, "Review_Upvotes")
    if args.title_col != "Review_Title" and args.title_col in renamed.columns:
        renamed = renamed.withColumnRenamed(args.title_col, "Review_Title")
    if args.text_col != "Review_Text" and args.text_col in renamed.columns:
        renamed = renamed.withColumnRenamed(args.text_col, "Review_Text")

    # Asegura existencia de columnas opcionales
    for c in ["Brand", "Product", "Review_Upvotes", "Review_Title", "Review_Text"]:
        if c not in renamed.columns:
            renamed = renamed.withColumn(c, F.lit(None))

    select_cols_csv = [
        "Brand", "Product", "Review_Upvotes", "pred_label",
        "prob_negative", "prob_neutral", "prob_positive",
        "Review_Title", "Review_Text"
    ]

    top_csv_dir = os.path.dirname(args.output_parquet) + "/top50_reviews_csv"
    (renamed.orderBy(F.col("Review_Upvotes").desc())
            .select(*[c for c in select_cols_csv if c in renamed.columns])
            .coalesce(1)
            .write.mode("overwrite")
            .option("header", True)
            .csv(top_csv_dir))

    print(f"[ok] Predicciones Parquet: {args.output_parquet}")
    print(f"[ok] Distribución CSV:     {dist_dir}")
    print(f"[ok] Top-N reviews CSV:    {top_csv_dir}")

    spark.stop()


if __name__ == "__main__":
    main()





