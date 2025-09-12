# -*- coding: utf-8 -*-
# Lee explotación (Delta Sephora) y saca métricas en JSON.

from pyspark.sql import SparkSession, functions as F
from delta import configure_spark_with_delta_pip

INPUT_PATH  = "data/exploitation/modelos_input/reviews_base"   # Delta
OUTPUT_PATH = "data/exploitation/modelos_input/hera_report"    # JSON

builder = (
    SparkSession.builder
    .appName("HERA")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

def run_hera():
    df = spark.read.format("delta").load(INPUT_PATH)
    resumen = df.select(
        F.count("*").alias("num_reviews"),
        F.avg(F.col("rating").cast("double")).alias("rating_promedio"),
        F.avg(F.col("text_len").cast("double")).alias("longitud_promedio"),
    )
    resumen.write.mode("overwrite").json(OUTPUT_PATH)
    resumen.show(truncate=False)
    print(f"[OK] Reporte HERA -> {OUTPUT_PATH}")

if __name__ == "__main__":
    run_hera()
