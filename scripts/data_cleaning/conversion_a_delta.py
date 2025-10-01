# -*- coding: utf-8 -*-
# Convierte CSV -> (Sephora: Delta) | (Ulta: Parquet)

import os
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

builder = (
    SparkSession.builder
    .appName("CSV->Delta/Parquet")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

BASE = "data/landing"
SEPH_RAW = os.path.join(BASE, "sephora/raw")
SEPH_DEL = os.path.join(BASE, "sephora/delta")
ULTA_RAW = os.path.join(BASE, "ulta/raw")
ULTA_PAR = os.path.join(BASE, "ulta/parquet")

os.makedirs(SEPH_DEL, exist_ok=True)
os.makedirs(ULTA_PAR, exist_ok=True)

def _csvs(path): return [f for f in os.listdir(path) if f.lower().endswith(".csv")]

def convert():
    # Sephora -> Delta
    for f in _csvs(SEPH_RAW):
        name = os.path.splitext(f)[0]  # p.ej. reviews_0-250
        src = os.path.join(SEPH_RAW, f)
        dst = os.path.join(SEPH_DEL, name)
        df = spark.read.option("header", True).csv(src)
        df.write.format("delta").mode("overwrite").save(dst)
        print(f"[OK] Sephora {f} -> DELTA -> {dst}")

    # Ulta -> Parquet (no se usa después)
    if os.path.isdir(ULTA_RAW):
        for f in _csvs(ULTA_RAW):
            name = os.path.splitext(f)[0]
            src = os.path.join(ULTA_RAW, f)
            dst = os.path.join(ULTA_PAR, name)
            df = spark.read.option("header", True).csv(src)
            df.write.mode("overwrite").parquet(dst)
            print(f"[OK] Ulta {f} -> PARQUET -> {dst}")

if __name__ == "__main__":
    convert()
