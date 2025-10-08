#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys
from pyspark.sql import SparkSession, functions as F

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_delta", required=True)
    p.add_argument("--id-col", default="product_id")
    p.add_argument("--rating-col", default="rating")
    p.add_argument("--text-col", default="review_text_clean")
    p.add_argument("--out", default="reports/absa/product_rating_stats.parquet")
    return p.parse_args()

def get_spark():
    """
    Crea una SparkSession con Delta Lake habilitado.
    Si existe delta-spark (pip), la usa para inyectar los jars y extensiones.
    """
    builder = (
        SparkSession.builder.appName("rating_stats")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )

    try:
        # Requiere: pip install delta-spark
        from delta import configure_spark_with_delta_pip
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
    except Exception as e:
        # Fallback: al menos deja las extensiones puestas (si no hay jars, fallará al leer Delta)
        print(f"[warn] delta-spark no disponible o no se pudo cargar ({e}). "
              f"Intento continuar con builder base...", file=sys.stderr)
        spark = builder.getOrCreate()
    return spark

def main():
    a = parse_args()
    spark = get_spark()

    # Lee Delta
    df = spark.read.format("delta").load(a.input_delta)

    # Señales para conteos
    has_rating = F.col(a.rating_col).isNotNull()
    has_review = F.col(a.text_col).isNotNull() & (F.length(F.col(a.text_col)) > 0)

    stats = (
        df.groupBy(a.id_col)
          .agg(
              F.count(F.when(has_rating, True)).alias("ratings_count"),
              F.count(F.when(has_review, True)).alias("reviews_count"),
              F.avg(F.col(a.rating_col)).alias("rating_avg"),
              F.round(F.avg(F.col(a.rating_col)), 2).alias("rating_avg_2d"),
          )
    )

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    stats.write.mode("overwrite").parquet(a.out)
    print(f"[ok] stats -> {a.out}")

if __name__ == "__main__":
    main()

