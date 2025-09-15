#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
from pyspark.sql import SparkSession

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Carpeta con los trozos trusted (reviews_*)")
    p.add_argument("--output", required=True, help="Ruta de salida final (ej. data/trusted/reviews_full)")
    p.add_argument("--format", default="parquet", choices=["parquet", "delta"], help="Formato de salida")
    p.add_argument("--coalesce", type=int, default=8, help="Nº de particiones de salida")
    return p.parse_args()

def main():
    args = parse_args()

    # Construye la lista de rutas a partir de --input-dir
    # Tomamos reviews_* pero excluimos cualquier reviews_full previo si lo hubiera
    paths = sorted(
        p for p in glob.glob(os.path.join(args.input_dir, "reviews_*"))
        if os.path.basename(p) != "reviews_full"
    )
    if not paths:
        raise SystemExit(f"[merge] No se han encontrado particiones en {args.input_dir}/reviews_*")

    spark = (
        SparkSession.builder
        .appName("merge_trusted_reviews")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.files.maxRecordsPerFile", "500000")
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))  # 64MB
        .getOrCreate()
    )

    df = spark.read.parquet(*paths).coalesce(int(args.coalesce))

    writer = df.write.mode("overwrite").option("compression", "snappy").option("maxRecordsPerFile", 500000)
    if args.format == "delta":
        (writer.format("delta").save(args.output))
    else:
        (writer.parquet(args.output))

    print(f"✅ reviews_full → {args.output}")
    spark.stop()

if __name__ == "__main__":
    main()

