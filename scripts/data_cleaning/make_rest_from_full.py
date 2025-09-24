#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_rest_from_full.py

Crea un conjunto "resto" con todas las reviews de FULL que NO están en
train/val/test del sample actual (anti-join por review_id).

Si los Parquet de train/val/test no se pueden leer (p.ej. TIMESTAMP(NANOS)),
usa --sample-delta (Delta del sample) para extraer los IDs.

Uso:
  python scripts/data_cleaning/make_rest_from_full.py \
    --full data/trusted/reviews_full \
    --sample-root data/exploitation/modelos_input/sample_tvt \
    --sample-delta data/trusted/reviews_sample_30pct \
    --out data/exploitation/modelos_input/rest_from_full \
    --format parquet \
    --coalesce 64
"""

import os
import argparse
from pyspark.sql import SparkSession, DataFrame, functions as F

# Delta opcional (si está instalado)
try:
    from delta import configure_spark_with_delta_pip
    HAS_DELTA = True
except Exception:
    HAS_DELTA = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full", required=True, help="Ruta FULL (Delta/Parquet), p.ej. data/trusted/reviews_full")
    p.add_argument("--sample-root", required=True, help="Raíz del split actual (carpeta con train/ val/ test/)")
    p.add_argument("--out", required=True, help="Salida para el resto (Delta/Parquet)")
    p.add_argument("--format", choices=["delta", "parquet"], default="parquet", help="Formato de salida")
    p.add_argument("--coalesce", type=int, default=64)
    # NUEVO: sample Delta (ej. data/trusted/reviews_sample_30pct) para extraer IDs si parquet falla
    p.add_argument("--sample-delta", default=None, help="Ruta del sample en Delta para fallback de IDs")
    return p.parse_args()


def build_spark():
    builder = (
        SparkSession.builder
        .appName("make_rest_from_full")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.files.maxRecordsPerFile", "500000")
    )
    if HAS_DELTA:
        builder = (
            builder
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
    else:
        spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def is_delta(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "_delta_log"))


def read_any(spark: SparkSession, path: str) -> DataFrame:
    if is_delta(path):
        return spark.read.format("delta").load(path)
    return spark.read.parquet(path)


def read_ids_from_splits_or_delta(spark: SparkSession, sample_root: str, sample_delta: str | None) -> DataFrame:
    """Intenta leer review_id de train/val/test; si falla por Parquet TIMESTAMP(NANOS), usa sample_delta (Delta)."""
    train_p = os.path.join(sample_root, "train")
    val_p   = os.path.join(sample_root, "val")
    test_p  = os.path.join(sample_root, "test")

    for p in [train_p, val_p, test_p]:
        if not os.path.isdir(p):
            raise SystemExit(f"[rest] No existe la carpeta esperada: {p}")

    try:
        df_train = read_any(spark, train_p).select("review_id").distinct()
        df_val   = read_any(spark, val_p).select("review_id").distinct()
        df_test  = read_any(spark, test_p).select("review_id").distinct()
        return df_train.unionByName(df_val).unionByName(df_test).distinct()
    except Exception as e:
        msg = str(e)
        print(f"[rest][WARN] No se pudieron leer los Parquet de splits ({type(e).__name__}: {msg[:140]}...)")
        if sample_delta and os.path.isdir(sample_delta) and is_delta(sample_delta):
            print(f"[rest][INFO] Usando fallback: IDs desde sample Delta -> {sample_delta}")
            return spark.read.format("delta").load(sample_delta).select("review_id").distinct()
        raise SystemExit("[rest][ERROR] Falla lectura de Parquet y no se proporcionó --sample-delta válido. "
                         "Pasa --sample-delta data/trusted/reviews_sample_30pct")


def main():
    args = parse_args()
    spark = build_spark()

    # 1) FULL
    df_full = read_any(spark, args.full)
    if "review_id" not in df_full.columns:
        raise SystemExit("[rest] 'review_id' no está en FULL; no se puede cruzar.")

    # 2) IDs usados en el sample actual (preferimos splits; si parquet falla, usamos sample_delta)
    df_ids = read_ids_from_splits_or_delta(spark, args.sample_root, args.sample_delta)

    # 3) Anti-join → resto (no usado)
    df_rest = df_full.join(df_ids, on="review_id", how="left_anti")

    # 4) Métricas
    tot_full = df_full.count()
    tot_ids  = df_ids.count()
    tot_rest = df_rest.count()
    print(f"[rest] full={tot_full} | ids_train+val+test={tot_ids} | rest={tot_rest}")

    # 5) Escribir
    if args.coalesce and args.coalesce > 0:
        df_rest = df_rest.coalesce(args.coalesce)

    writer = (
        df_rest.write
        .mode("overwrite")
        .option("compression", "snappy")
        .option("maxRecordsPerFile", 500000)
    )
    if args.format == "delta":
        writer.format("delta").save(args.out)
    else:
        writer.parquet(args.out)

    print(f"✅ rest guardado en {args.out} [{args.format}]")
    spark.stop()


if __name__ == "__main__":
    main()

