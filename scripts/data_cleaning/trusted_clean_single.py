#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pyspark.sql import SparkSession, functions as F, types as T
from delta import configure_spark_with_delta_pip

if len(sys.argv) != 2 and len(sys.argv) != 3:
    print("Uso: python trusted_clean_single.py <dataset> <tabla>")
    sys.exit(1)

DATASET = sys.argv[1]
TABLA   = sys.argv[2]

INPUT_PATH  = f"data/landing/{DATASET}/delta/{TABLA}"
OUTPUT_PATH = f"data/trusted/{DATASET}_clean/{TABLA}"

builder = (
    SparkSession.builder
    .appName(f"trusted_clean_single::{DATASET}/{TABLA}")
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

def nz(colname: str, default: str = ""):
    return F.coalesce(F.col(colname).cast(T.StringType()), F.lit(default))

try:
    print(f"[INFO] Leyendo LANDING/DELTA: {INPUT_PATH}")
    df = spark.read.format("delta").load(INPUT_PATH)

    # Drop filas totalmente nulas y duplicados exactos
    df = df.dropna(how="all").dropDuplicates()

    # rating → int y filtrado 1–5
    if "rating" in df.columns:
        df = df.withColumn("rating", F.col("rating").cast(T.IntegerType()))
        df = df.filter(F.col("rating").isNotNull() & F.col("rating").between(1, 5))

    # review_text → normalizado (trim + lower + quitar tabs/nuevas líneas)
    if "review_text" in df.columns:
        df = df.withColumn("review_text", F.col("review_text").cast(T.StringType()))
        df = df.withColumn("review_text", F.regexp_replace(F.col("review_text"), r"[\r\n\t]+", " "))
        df = df.withColumn("review_text", F.lower(F.trim(F.col("review_text"))))
        df = df.filter(F.col("review_text").isNotNull() & (F.length(F.col("review_text")) > 0))

        # review_id reproducible
        df = df.withColumn(
            "review_id",
            F.sha1(
                F.concat_ws(
                    "||",
                    nz("product_id"),
                    nz("author_id"),
                    nz("submission_time"),
                    nz("review_title"),
                    nz("review_text")
                )
            )
        )
        df = df.dropDuplicates(["review_id"])

    # Quality report
    print("\n=== TRUSTED | QUALITY REPORT ===")
    total = df.count()
    print(f"Rows (trusted): {total}")

    if "rating" in df.columns:
        print("Distribución de ratings:")
        df.groupBy("rating").count().orderBy("rating").show()

    if "review_text" in df.columns:
        null_text = df.filter((F.col("review_text").isNull()) | (F.length("review_text") == 0)).count()
        print(f"review_text vacíos: {null_text}")

    if "review_id" in df.columns:
        dupl = df.groupBy("review_id").count().filter(F.col("count") > 1).count()
        print(f"Duplicados por review_id: {dupl}")

    # Guardar en TRUSTED conservando todas las columnas originales + review_id
    print(f"\n[INFO] Guardando TRUSTED/DELTA en: {OUTPUT_PATH}")
    df.write.format("delta").mode("overwrite").save(OUTPUT_PATH)

    print(f"[OK] Limpieza completada para {DATASET}/{TABLA} → {OUTPUT_PATH}")

except Exception as e:
    print(f"[ERROR] Procesando {DATASET}/{TABLA}: {e}")
    sys.exit(2)
finally:
    spark.stop()


