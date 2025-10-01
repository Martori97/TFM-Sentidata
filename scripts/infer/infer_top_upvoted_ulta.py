#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia sobre Top-N reseñas (Ulta) leyendo Delta.
Backends: --backend {albert, rf}

- Selecciona Top-N por upvotes (>0)
- Texto de entrada: título + texto (si --use_title=1) o solo texto
- Mantiene 'neutral' solo si p_neutral >= --min_neutral_prob
- Salidas:
  * Parquet con predicciones (pred_label, prob_negative/neutral/positive)
  * CSV con distribución (pred_distribution_csv/)
  * CSV con Top-N (top50_reviews_csv/)
"""

import argparse
import os
import json
import hashlib
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, functions as F, types as T
from delta import configure_spark_with_delta_pip

# === ALBERT ===
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------
# Spark + Delta
# -----------------------
def build_spark(app_name: str = "infer_top_upvoted_ulta") -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# -----------------------
# Utilidades
# -----------------------
def pick_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print(f"[device] Using cuda:0: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[device] Using CPU")
    return dev


def softmax(logits, axis: int = -1):
    e = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def ensure_review_id(sdf, id_col: str, cols_for_hash=("Review_Text", "Review_Title", "Product", "Brand")):
    """Si no existe id_col, genera un hash determinista a partir de título/texto+producto+marca."""
    if id_col in sdf.columns:
        return sdf
    def _mk_hash(*vals):
        base = "||".join([str(v) if v is not None else "" for v in vals])
        return hashlib.md5(base.encode("utf-8")).hexdigest()
    mk_hash = F.udf(_mk_hash, T.StringType())
    cols = [c for c in cols_for_hash if c in sdf.columns] or [sdf.columns[0]]
    return sdf.withColumn(id_col, mk_hash(*map(F.col, cols)))


# -----------------------
# RF loader (compatible con tus nombres)
# -----------------------
def _load_rf_pipeline(model_dir):
    import joblib
    from pathlib import Path
    model_dir = Path(model_dir)

    # vectorizer
    vec = None
    for name in ("tfidf_vectorizer.joblib", "vectorizer.joblib", "tfidf.joblib", "vectorizer.pkl"):
        p = model_dir / name
        if p.exists():
            try:
                vec = joblib.load(p)
            except Exception:
                try:
                    vec = pd.read_pickle(p)
                except Exception:
                    pass
            if vec is not None:
                break

    # modelo
    rf = None
    for name in ("rf_model.joblib", "modelo_random_forest.joblib", "rf.joblib", "model.joblib", "rf.pkl", "model.pkl"):
        p = model_dir / name
        if p.exists():
            try:
                rf = joblib.load(p)
            except Exception:
                try:
                    rf = pd.read_pickle(p)
                except Exception:
                    pass
            if rf is not None:
                break

    pre_cfg = {}
    pcfg = model_dir / "preprocess_config.json"
    if pcfg.exists():
        pre_cfg = json.loads(pcfg.read_text())

    if vec is None or rf is None:
        raise SystemExit(f"[model] No pude cargar vectorizer/rf desde {model_dir}")
    return vec, rf, pre_cfg


# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["albert", "rf"], default="albert")

    # Input Delta
    ap.add_argument("--input_delta", default="data/trusted/ulta_clean/Ulta Skincare Reviews")
    ap.add_argument("--text_col", default="Review_Text")
    ap.add_argument("--title_col", default="Review_Title")
    ap.add_argument("--use_title", type=int, default=1)
    ap.add_argument("--upvotes_col", default="Review_Upvotes")
    ap.add_argument("--id_col", default="review_id")

    # Modelo (ambos usan --model_dir)
    ap.add_argument("--model_dir", default="models/albert_sample_30pct")
    ap.add_argument("--label_names", default='["negative","neutral","positive"]')

    # Batching/Top
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=192)   # ALBERT
    ap.add_argument("--batch_size", type=int, default=64) # ALBERT

    # Output
    ap.add_argument("--output_parquet", default="reports/albert_sample_30pct/ulta_top50/predictions.parquet")
    ap.add_argument("--coalesce", type=int, default=1)

    # Neutrales
    ap.add_argument("--demote_neutral", type=int, default=1)
    ap.add_argument("--min_neutral_prob", type=float, default=0.90)

    # RF extra
    ap.add_argument("--use_extra", type=int, default=1)

    return ap.parse_args()


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    spark = build_spark()

    # Leer Delta y asegurar tipos
    sdf = spark.read.format("delta").load(args.input_delta)
    if args.upvotes_col not in sdf.columns:
        raise ValueError(f"No existe columna {args.upvotes_col} en {args.input_delta}")
    sdf = sdf.withColumn(args.upvotes_col, F.col(args.upvotes_col).cast("int"))

    # Asegurar ID
    sdf = ensure_review_id(sdf, args.id_col)

    # Seleccionar Top-N por upvotes (>0) y columnas útiles
    keep_extra = [c for c in ["Brand", "Product", "Verified_Buyer", "Review_Date"] if c in sdf.columns]
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
        print("[warn] No hay reseñas con upvotes > 0. Nada que inferir.")
        spark.stop()
        return

    label_names = json.loads(args.label_names)

    # =========================
    # Backend: ALBERT
    # =========================
    if args.backend == "albert":
        device = pick_device()
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

        use_title = bool(args.use_title and (args.title_col in pdf.columns))
        texts = pdf[args.text_col].fillna("").tolist()
        if use_title:
            titles = pdf[args.title_col].fillna("").tolist()

        logits_all = []
        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                batch_texts = texts[i:i+args.batch_size]
                if use_title:
                    batch_titles = titles[i:i+args.batch_size]
                    enc = tokenizer(batch_titles, batch_texts, padding=True, truncation=True,
                                    max_length=args.max_len, return_tensors="pt")
                else:
                    enc = tokenizer(batch_texts, padding=True, truncation=True,
                                    max_length=args.max_len, return_tensors="pt")
                enc = {k: v.to(device) for k,v in enc.items()}
                out = model(**enc)
                logits_all.append(out.logits.detach().cpu().numpy())

        probs = softmax(np.vstack(logits_all))
        p_neg, p_neu, p_pos = probs[:,0], probs[:,1], probs[:,2]
        pred_idx = probs.argmax(axis=1)

    # =========================
    # Backend: RF
    # =========================
    else:
        # Cargar piezas RF
        vec, rf, pre_cfg = _load_rf_pipeline(args.model_dir)

        # Preprocesado del RF (usa utilidades del training)
        import sys as _sys
        from pathlib import Path as _Path
        TRAIN_DIR = _Path(__file__).resolve().parents[1] / "training"
        if str(TRAIN_DIR) not in _sys.path:
            _sys.path.insert(0, str(TRAIN_DIR))
        from train_random_forest import TextCleaner, ExtraFeatures  # noqa

        # Texto (título + texto o sólo texto)
        if args.use_title and (args.title_col in pdf.columns):
            texts = (pdf[args.title_col].fillna("").astype(str) + ". " +
                     pdf[args.text_col].fillna("").astype(str)).str.strip().tolist()
        else:
            texts = pdf[args.text_col].fillna("").astype(str).tolist()

        cleaner = TextCleaner(use_lemma=pre_cfg.get("use_lemma", True),
                              lang=pre_cfg.get("language", "en")).fit([])

        # ⬇️ El cleaner espera Series (usa .astype)
        texts_series = pd.Series(texts, dtype=str)
        clean_txt = cleaner.transform(texts_series)

        # Vectorización
        X = vec.transform(clean_txt)

        # Extra features opcionales
        if int(args.use_extra) == 1:
            xf = ExtraFeatures().transform(pd.Series(clean_txt, dtype=str))
            from scipy.sparse import hstack
            X = hstack([X, xf], format="csr")

        # Probabilidades y alineación a [neg, neu, pos]
        proba = rf.predict_proba(X)
        classes = list(rf.classes_)
        aligned = np.zeros((proba.shape[0], 3), dtype=float)

        def to_idx(c):
            s = str(c).lower()
            if s in ("0","neg","negative"): return 0
            if s in ("1","neu","neutral"):  return 1
            if s in ("2","pos","positive"): return 2
            try:
                return int(c)
            except Exception:
                raise ValueError(f"Clase desconocida en RF: {c}")

        for j, c in enumerate(classes):
            i = to_idx(c)
            if i in (0,1,2):
                aligned[:, i] = proba[:, j]

        p_neg, p_neu, p_pos = aligned[:,0], aligned[:,1], aligned[:,2]
        pred_idx = np.argmax(aligned, axis=1)

    # Argmax → etiquetas
    pred_label = np.array([label_names[i] for i in pred_idx], dtype=object)

    # Demote neutral si p_neu < umbral
    if args.demote_neutral:
        mask = (pred_label == "neutral") & (p_neu < float(args.min_neutral_prob))
        better_is_neg = p_neg >= p_pos
        pred_label[mask &  better_is_neg] = "negative"
        pred_label[mask & ~better_is_neg] = "positive"

    # -----------------------
    # Salvar resultados
    # -----------------------
    out = pdf.copy()
    out["pred_label"] = pred_label
    out["prob_negative"] = p_neg
    out["prob_neutral"]  = p_neu
    out["prob_positive"] = p_pos

    # Asegura columnas de texto para export
    if args.title_col not in out.columns: out[args.title_col] = None
    if args.text_col  not in out.columns: out[args.text_col]  = None

    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    (spark.createDataFrame(out)
          .coalesce(args.coalesce)
          .write.mode("overwrite").parquet(args.output_parquet))

    # Distribución
    print("[pred distribution]")
    dist = (spark.read.parquet(args.output_parquet)
            .groupBy("pred_label").count().orderBy("count", ascending=False))
    dist.show(truncate=False)
    dist_dir = os.path.dirname(args.output_parquet) + "/pred_distribution_csv"
    (dist.coalesce(1).write.mode("overwrite").option("header", True).csv(dist_dir))

    # Top-N CSV (ordenado por upvotes desc, con texto y título)
    out_full = spark.read.parquet(args.output_parquet)
    renamed = out_full
    if args.upvotes_col != "Review_Upvotes" and args.upvotes_col in renamed.columns:
        renamed = renamed.withColumnRenamed(args.upvotes_col, "Review_Upvotes")
    if args.title_col != "Review_Title" and args.title_col in renamed.columns:
        renamed = renamed.withColumnRenamed(args.title_col, "Review_Title")
    if args.text_col != "Review_Text" and args.text_col in renamed.columns:
        renamed = renamed.withColumnRenamed(args.text_col, "Review_Text")

    for c in ["Brand", "Product", "Review_Upvotes", "Review_Title", "Review_Text"]:
        if c not in renamed.columns:
            renamed = renamed.withColumn(c, F.lit(None))

    select_cols_csv = ["Brand","Product","Review_Upvotes","pred_label",
                       "prob_negative","prob_neutral","prob_positive",
                       "Review_Title","Review_Text"]

    top_csv_dir = os.path.dirname(args.output_parquet) + "/top50_reviews_csv"
    (renamed.orderBy(F.col("Review_Upvotes").desc())
            .select(*[c for c in select_cols_csv if c in renamed.columns])
            .coalesce(1).write.mode("overwrite").option("header", True).csv(top_csv_dir))

    print(f"[ok] Predicciones Parquet: {args.output_parquet}")
    print(f"[ok] Distribución CSV:     {dist_dir}")
    print(f"[ok] Top-N reviews CSV:    {top_csv_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
