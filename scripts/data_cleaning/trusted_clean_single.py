# -*- coding: utf-8 -*-
# Limpia UNA tabla Sephora (Delta -> Delta) SIN spaCy

import sys, os
from pyspark.sql import SparkSession, functions as F, types as T
from delta import configure_spark_with_delta_pip

builder = (
    SparkSession.builder
    .appName("TrustedCleanSingle-Sephora")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "8")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Hacer disponible utils_text en los executors
spark.sparkContext.addPyFile("scripts/data_cleaning/utils_text.py")
from utils_text import clean_text

clean_udf = F.udf(clean_text, T.StringType())

TEXT_CANDIDATES = ["review_text", "review_body", "text", "review", "content", "reviewText", "body", "comments"]
RATING_CANDIDATES = ["rating", "stars", "score", "overall", "star_rating"]

def _ensure_columns(df):
    text_col = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
    if text_col is None:
        return None
    if text_col != "review_text":
        df = df.withColumnRenamed(text_col, "review_text")
    rating_col = next((c for c in RATING_CANDIDATES if c in df.columns), None)
    if rating_col and rating_col != "rating":
        df = df.withColumnRenamed(rating_col, "rating")
    if "rating" in df.columns:
        df = df.withColumn("rating", F.col("rating").cast("double"))
    return df

def clean_table(input_delta, output_delta):
    df = spark.read.format("delta").load(input_delta)

    if "review_id" not in df.columns:
        df = df.withColumn("review_id", F.monotonically_increasing_id())

    df = _ensure_columns(df)
    if df is None:
        print(f"[SKIP] {input_delta} sin columna de texto; no se genera trusted.")
        return

    if "review_text_clean" not in df.columns:
        df = df.withColumn("review_text_clean", clean_udf(F.col("review_text")))
    df = df.withColumn("text_len", F.length(F.col("review_text_clean")))

    df.repartition(4).write.format("delta").mode("overwrite").save(output_delta)
    print(f"[OK] Trusted (clean+len) -> {output_delta}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: trusted_clean_single.py <input_delta_dir> <output_delta_dir>")
        sys.exit(1)
    clean_table(sys.argv[1], sys.argv[2])
