#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge: reviews_full + product_info por product_id (Trusted en Delta)
- Limpia product_id (trim + cast + vacíos→NULL) y elimina filas con product_id NULL en ambos datasets.
- Dedup de product_info por product_id.
- Prefijo en TODAS las columnas de product_info (excepto product_id), por defecto 'pinfo_'.
- JOIN: **INNER** (solo conserva reviews con product_info).
- Anti-OOM: AQE, skew join, opciones de memoria y splits, broadcast opcional.
- Escritura en Delta o Parquet.

Variables opcionales de entorno:
  export SPARK_DRIVER_MEMORY=8g
  export SPARK_EXECUTOR_MEMORY=4g

Uso ejemplo:
python scripts/data_cleaning/merge_reviews_product_info.py \
  --reviews_path data/trusted/reviews_full \
  --product_info_path data/trusted/sephora_clean/product_info \
  --output_path data/trusted/reviews_product_info_clean_full \
  --input_format auto --output_format delta \
  --repartition 0 --coalesce 64 \
  --product_prefix pinfo_ --broadcast_product_info true
"""

import argparse
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark_with_delta(app_name: str = "merge_reviews_product_info") -> SparkSession:
    """
    SparkSession con Delta y ajustes anti-OOM.
    """
    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "8g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", "4g")

    builder = (
        SparkSession.builder
        .appName(app_name)
        # Delta
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Memoria
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.3")
        # AQE + skew
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Shuffle fino
        .config("spark.sql.shuffle.partitions", "400")
        # Broadcast generoso (si cabe pinfo)
        .config("spark.sql.autoBroadcastJoinThreshold", str(256 * 1024 * 1024))  # 256MB
        # Splits pequeños (texto largo)
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))  # 64MB
        # Varios
        .config("spark.sql.files.ignoreCorruptFiles", "true")
    )

    try:
        from delta import configure_spark_with_delta_pip
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        return spark
    except Exception as e:
        sys.stderr.write(
            "[warn] No se pudo inicializar delta-spark via pip. "
            "Instala 'delta-spark' o añade los JARs de Delta a Spark.\n"
            f"[warn] Detalle: {repr(e)}\n"
        )
        return builder.getOrCreate()


def is_delta_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, "_delta_log"))


def read_any(spark: SparkSession, path: str, fmt: str = "auto"):
    if fmt == "auto":
        fmt = "delta" if is_delta_dir(path) else "parquet"
    if fmt == "delta":
        return spark.read.format("delta").load(path)
    elif fmt == "parquet":
        return spark.read.parquet(path)
    else:
        raise ValueError("--input_format debe ser 'auto', 'delta' o 'parquet'")


def write_any(df, path: str, fmt: str, mode: str = "overwrite"):
    if fmt == "delta":
        df.write.format("delta").mode(mode).save(path)
    elif fmt == "parquet":
        df.write.mode(mode).parquet(path)
    else:
        raise ValueError("--output_format debe ser 'delta' o 'parquet'")


def normalize_and_drop_null_product_id(df, dataset_name: str):
    """
    Normaliza product_id (trim + cast string + vacíos→NULL) y elimina filas con product_id NULL.
    Devuelve (df_limpio, total, dropped, kept)
    """
    df = df.withColumn("product_id", F.trim(F.col("product_id").cast("string")))
    df = df.withColumn("product_id", F.when(F.col("product_id") == "", F.lit(None)).otherwise(F.col("product_id")))
    total = df.count()
    df_non_null = df.filter(F.col("product_id").isNotNull())
    kept = df_non_null.count()
    dropped = total - kept
    print(f"[clean] {dataset_name}: total={total} | dropped_null_product_id={dropped} | kept={kept}")
    return df_non_null, total, dropped, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews_path", required=True)
    ap.add_argument("--product_info_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--input_format", default="auto", choices=["auto", "delta", "parquet"])
    ap.add_argument("--output_format", default="delta", choices=["delta", "parquet"])
    ap.add_argument("--coalesce", type=int, default=64, help=">0 coalesce(N) (NO shuffle)")
    ap.add_argument("--repartition", type=int, default=0, help=">0 repartition(N) (sí shuffle)")
    ap.add_argument("--broadcast_product_info", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--product_prefix", default="pinfo_")
    args = ap.parse_args()

    spark = build_spark_with_delta()

    reviews_raw = read_any(spark, args.reviews_path, args.input_format)
    pinfo_raw = read_any(spark, args.product_info_path, args.input_format)

    if "product_id" not in reviews_raw.columns:
        raise ValueError("reviews_full no contiene 'product_id'.")
    if "product_id" not in pinfo_raw.columns:
        raise ValueError("product_info no contiene 'product_id'.")

    # Limpieza: normaliza y elimina product_id NULL
    reviews, rev_total, rev_dropped, rev_kept = normalize_and_drop_null_product_id(reviews_raw, "reviews_full")
    pinfo, pin_total, pin_dropped, pin_kept = normalize_and_drop_null_product_id(pinfo_raw, "product_info")

    # Dedup product_info
    pinfo = pinfo.dropDuplicates(["product_id"])

    # Prefijo en columnas de product_info
    prefix = args.product_prefix or ""
    renamed_cols = [F.col("product_id")]
    for c in pinfo.columns:
        if c != "product_id":
            renamed_cols.append(F.col(c).alias((prefix + c) if prefix else c))
    pinfo_prefixed = pinfo.select(*renamed_cols)

    # Broadcast opcional
    do_broadcast = (args.broadcast_product_info.lower() == "true")
    if do_broadcast:
        pinfo_prefixed = F.broadcast(pinfo_prefixed)

    # === JOIN INNER === solo reviews con match en product_info
    joined = reviews.alias("r").join(pinfo_prefixed.alias("p"), on="product_id", how="inner")

    # Particiones de salida
    out_df = joined
    if do_broadcast:
        if args.repartition > 0:
            print("[info] Ignorando --repartition porque hay broadcast; usando coalesce para evitar shuffle.")
        parts = args.coalesce if args.coalesce and args.coalesce > 0 else 64
        out_df = out_df.coalesce(parts)
    else:
        if args.repartition > 0:
            out_df = out_df.repartition(args.repartition)
        elif args.coalesce > 0:
            out_df = out_df.coalesce(args.coalesce)

    # Escritura
    write_any(out_df, args.output_path, args.output_format, mode="overwrite")

    # Métricas
    n_joined = out_df.count()
    print(
        f"[stats] reviews_in={rev_total} | reviews_clean={rev_kept} (dropped_null_id={rev_dropped}) | "
        f"product_info_in={pin_total} | product_info_clean={pin_kept} (dropped_null_id={pin_dropped}) | "
        f"joined_rows={n_joined} (INNER coverage vs clean reviews = {n_joined/max(1,rev_kept):.1%})"
    )

    spark.stop()


if __name__ == "__main__":
    main()

