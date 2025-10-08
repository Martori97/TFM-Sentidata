#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_preview_fused.py
----------------------

Genera un dataset unificado con:
- reviews integradas (Delta) normalizadas -> brand, category, product_name, rating, review_text_clean
- spans ATE mapeados a ontología -> aspect_norm, aspect_span, aspect_score, aspect_prob
- sentimiento (opcional) -> pred_3, p_neg, p_neu, p_pos

Salida (por defecto):
  reports/absa/final_all/preview_fused.parquet

Uso rápido:
  python scripts/absa/build_preview_fused.py \
    --reviews_delta data/trusted/reviews_product_info_clean_full \
    --ate_mapped_parquet reports/absa/ate_spans_mapped.parquet \
    --sentiment_parquet reports/albert_sample_30pct/infer_full_pandas/predictions.parquet \
    --output_parquet reports/absa/final_all/preview_fused.parquet
"""

import os
import argparse

from pyspark.sql import SparkSession, functions as F

# --------------------------------------------------------------------
# Spark con Delta
# --------------------------------------------------------------------
def build_spark(app_name: str = "absa-fuse-preview") -> SparkSession:
    try:
        from delta import configure_spark_with_delta_pip
    except Exception as e:
        raise RuntimeError(
            "No se pudo importar delta. Instala: pip install delta-spark==3.1.0"
        ) from e

    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# --------------------------------------------------------------------
# Args
# --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Construye dataset unificado para ABSA/visualización.")
    p.add_argument("--reviews_delta", type=str, required=True,
                   help="Ruta Delta de reviews integradas (p.ej., data/trusted/reviews_product_info_clean_full)")
    p.add_argument("--ate_mapped_parquet", type=str, default="reports/absa/ate_spans_mapped.parquet",
                   help="Parquet de spans mapeados a ontología (por defecto existe en reports/absa/)")
    p.add_argument("--ate_spans_parquet", type=str, default="reports/absa/ate_spans.parquet",
                   help="Parquet de spans sin mapear (fallback si no existe el mapeado)")
    p.add_argument("--sentiment_parquet", type=str, default="reports/albert_sample_30pct/infer_full_pandas/predictions.parquet",
                   help="Parquet con sentimiento por review_id (opcional)")
    p.add_argument("--output_parquet", type=str, default="reports/absa/final_all/preview_fused.parquet",
                   help="Ruta de salida del parquet unificado")
    p.add_argument("--coalesce", type=int, default=1, help="Número de ficheros de salida (coalesce)")
    p.add_argument("--limit_show", type=int, default=10, help="Cuántas filas mostrar en la previsualización")
    return p.parse_args()

# --------------------------------------------------------------------
# Lecturas y normalización
# --------------------------------------------------------------------
def read_reviews_normalized(spark: SparkSession, reviews_delta: str):
    df = spark.read.format("delta").load(reviews_delta)
    reviews = df.select(
        "review_id",
        "product_id",
        F.coalesce(F.col("pinfo_brand_name"), F.col("brand_name")).alias("brand"),
        F.coalesce(
            F.col("pinfo_primary_category"),
            F.col("pinfo_secondary_category"),
            F.col("pinfo_tertiary_category")
        ).alias("category"),
        "rating",
        F.coalesce(F.col("pinfo_product_name"), F.col("product_name")).alias("product_name"),
        "review_text_clean"
    ).dropna(subset=["review_id"])
    return reviews

def read_ate_mapped_or_spans(spark: SparkSession, mapped_path: str, spans_path: str):
    """
    Devuelve columnas estandarizadas:
      review_id, aspect_span, aspect_norm, aspect_score, aspect_prob
    """
    if os.path.exists(mapped_path):
        mapped = spark.read.parquet(mapped_path)
        # Columnas esperadas (según tu head): review_id, span_text, aspect_norm, score, prob_span
        cols = mapped.columns
        if not {"review_id", "aspect_norm"}.issubset(set(cols)):
            raise ValueError(f"El parquet mapeado no contiene columnas mínimas: {mapped_path}")
        out = mapped.select(
            "review_id",
            F.col("span_text").alias("aspect_span") if "span_text" in cols else F.lit(None).alias("aspect_span"),
            "aspect_norm",
            F.col("score").alias("aspect_score") if "score" in cols else F.lit(None).alias("aspect_score"),
            F.col("prob_span").alias("aspect_prob") if "prob_span" in cols else F.lit(None).alias("aspect_prob"),
        )
        return out

    # Fallback a spans sin mapear
    if not os.path.exists(spans_path):
        raise FileNotFoundError(
            f"No existe ni el mapeado ({mapped_path}) ni el spans base ({spans_path})."
        )
    spans = spark.read.parquet(spans_path)
    cols = spans.columns
    out = spans.select(
        "review_id" if "review_id" in cols else F.lit(None).alias("review_id"),
        F.col("aspect_span") if "aspect_span" in cols else F.lit(None).alias("aspect_span"),
        F.lit(None).alias("aspect_norm"),
        F.lit(None).alias("aspect_score"),
        F.col("prob").alias("aspect_prob") if "prob" in cols else F.lit(None).alias("aspect_prob"),
    )
    return out

def read_sentiment_optional(spark: SparkSession, sent_path: str):
    if not os.path.exists(sent_path):
        return None, []
    sent = spark.read.parquet(sent_path)
    cand = [c for c in ["review_id", "pred_3", "p_neg", "p_neu", "p_pos"] if c in sent.columns]
    if "review_id" not in cand:
        # no hay review_id usable
        return None, []
    # Dejar 1 registro por review_id
    sent = sent.select(*cand).dropna(subset=["review_id"]).dropDuplicates(["review_id"])
    return sent, cand

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    a = parse_args()
    spark = build_spark()

    print("[INFO] Leyendo reviews Delta normalizadas…")
    reviews = read_reviews_normalized(spark, a.reviews_delta)
    n_reviews = reviews.select("review_id").distinct().count()
    print(f"[OK] Reviews únicas: {n_reviews:,}")

    print("[INFO] Leyendo ATE (mapeado o spans base)…")
    ate = read_ate_mapped_or_spans(spark, a.ate_mapped_parquet, a.ate_spans_parquet)
    n_spans = ate.count()
    print(f"[OK] Spans ATE filas: {n_spans:,}")

    # Cobertura ATE (% de reviews con >=1 span)
    coverage = (
        reviews.select("review_id").distinct()
        .join(ate.groupBy("review_id").agg(F.count("*").alias("n_spans")), on="review_id", how="left")
        .withColumn("has_span", F.when(F.col("n_spans").isNull() | (F.col("n_spans") == 0), 0).otherwise(1))
    )
    total = coverage.count()
    with_span = coverage.agg(F.sum("has_span")).first()[0]
    ratio = (with_span / total) if total else 0.0
    print(f"[Cobertura ATE] reviews totales={total:,} | con spans={with_span:,} | ratio={ratio:.2%}")

    # Sentimiento (opcional)
    print("[INFO] Intentando leer sentimiento…")
    sent, sent_cols = read_sentiment_optional(spark, a.sentiment_parquet)
    if sent is None:
        print("[INFO] Sentimiento NO encontrado. Continuo sin sentimiento.")
    else:
        print(f"[OK] Sentimiento columnas: {sent_cols}")

    # Fusionar: 1 fila por span (join por review_id para añadir brand/category/rating/…)
    print("[INFO] Fusionando datasets…")
    fused = (
        ate.join(reviews, on="review_id", how="left")
    )
    if sent is not None:
        fused = fused.join(sent, on="review_id", how="left")

    # Guardar
    out_dir = os.path.dirname(a.output_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    (
        fused.coalesce(a.coalesce)
             .write.mode("overwrite")
             .parquet(a.output_parquet)
    )
    print(f"[OK] Guardado: {a.output_parquet}")

    # Muestra rápida
    cols_show = [c for c in [
        "review_id", "brand", "category", "product_name", "rating",
        "aspect_norm", "aspect_span", "aspect_score", "aspect_prob",
        "pred_3", "p_neg", "p_neu", "p_pos"
    ] if c in fused.columns]

    print("\n[PREVIEW]")
    fused.select(*cols_show).show(a.limit_show, truncate=False)

    print("\n[Top brand×aspect por nº spans]")
    (
        fused.groupBy("brand", "aspect_norm")
             .agg(F.count("*").alias("n_spans"))
             .orderBy(F.desc("n_spans"))
             .show(10, truncate=False)
    )

if __name__ == "__main__":
    main()
