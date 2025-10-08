#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta un snapshot Parquet de productos a partir del Delta principal.
Lee rutas de absa_paths (params.yaml).

Salida:
  data/exploitation/product_info_snapshot.parquet
"""

import argparse
import os
import yaml
from pyspark.sql import SparkSession, functions as F


def build_spark():
    return (
        SparkSession.builder
        .appName("ExportProductInfoSnapshot")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.params, "r", encoding="utf-8"))
    p = cfg.get("absa_paths", cfg["paths"])  # prioridad a absa_paths
    src_delta = p["delta_product_reviews"]
    out_parquet = p["product_snapshot"]

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.format("delta").load(src_delta)

    # Campos derivados: product_name2, brand_name2, category2
    product_name2 = F.coalesce(F.col("pinfo_product_name"), F.col("product_name")).alias("product_name2")
    brand_name2 = F.coalesce(F.col("pinfo_brand_name"), F.col("brand_name")).alias("brand_name2")
    category2 = F.coalesce(F.col("pinfo_secondary_category"), F.col("pinfo_primary_category")).alias("category2")

    snap = (
        df.select(
            "review_id", "product_id",
            "product_name", "pinfo_product_name",
            "brand_name", "pinfo_brand_name",
            "rating", "pinfo_rating", "pinfo_reviews", "pinfo_price_usd",
            "pinfo_loves_count", "pinfo_secondary_category", "pinfo_primary_category"
        )
        .withColumn("product_name2", product_name2)
        .withColumn("brand_name2", brand_name2)
        .withColumn("category2", category2)
        .select(
            "review_id", "product_id", "product_name2", "brand_name2", "category2",
            "pinfo_loves_count", "pinfo_rating", "pinfo_reviews", "pinfo_price_usd", "rating"
        )
        .dropDuplicates(["review_id"])
    )

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    snap.write.mode("overwrite").parquet(out_parquet)
    print("[ok] exportado:", out_parquet, "| rows:", snap.count())


if __name__ == "__main__":
    main()



