#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window as W

# Delta opcional (igual que tu base)
try:
    from delta import configure_spark_with_delta_pip
    HAS_DELTA = True
except Exception:
    HAS_DELTA = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Carpeta con los trozos trusted (reviews_*)")
    p.add_argument("--output", required=True, help="Ruta de salida final (ej. data/trusted/reviews_full)")
    p.add_argument("--format", default="delta", choices=["parquet", "delta"], help="Formato de salida (por defecto: delta)")
    p.add_argument("--coalesce", type=int, default=8, help="Nº de particiones de salida")
    p.add_argument("--input-format", default="auto", choices=["auto", "parquet", "delta"],
                   help="Forzar lectura como parquet/delta o autodetectar por _delta_log")
    return p.parse_args()


def build_spark(app_name="merge_trusted_reviews"):
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.files.maxRecordsPerFile", "500000")
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
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


def detect_is_delta(path: str) -> bool:
    return os.path.exists(os.path.join(path, "_delta_log"))


def read_many(spark: SparkSession, paths, input_format: str):
    dfs = []
    for p in paths:
        if input_format == "delta" or (input_format == "auto" and detect_is_delta(p)):
            dfs.append(spark.read.format("delta").load(p))
        else:
            dfs.append(spark.read.parquet(p))
    if len(dfs) == 1:
        return dfs[0]
    return reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)


def ensure_columns_for_tie_break(df):
    """
    Normaliza columnas usadas en el criterio de desempate de la deduplicación.
    - updated_at (o date si no existe; si ninguna, null)
    - review_text_clean (o review_text; si ninguna, "")
    """
    if "updated_at" not in df.columns:
        if "date" in df.columns:
            df = df.withColumn("updated_at", F.col("date"))
        else:
            df = df.withColumn("updated_at", F.lit(None).cast("timestamp"))
    if "review_text_clean" not in df.columns:
        if "review_text" in df.columns:
            df = df.withColumn("review_text_clean", F.col("review_text"))
        else:
            df = df.withColumn("review_text_clean", F.lit(""))
    return df


def dedup_by_review_id(df):
    """
    DEDUP determinista por review_id (hash ya creado en trusted_clean_single.py).
    Regla: más reciente (updated_at DESC) y, a igualdad, texto más largo.
    """
    if "review_id" not in df.columns:
        raise SystemExit("[merge] Falta columna 'review_id' en el dataframe unido.")
    df = ensure_columns_for_tie_break(df)

    w = W.partitionBy("review_id").orderBy(
        F.col("updated_at").desc_nulls_last(),
        F.length(F.col("review_text_clean")).desc_nulls_last()
    )
    out = (df.withColumn("rn", F.row_number().over(w))
             .filter(F.col("rn") == 1)
             .drop("rn"))
    return out


def main():
    args = parse_args()

    # Toma reviews_* bajo --input-dir (tu estructura de sephora_clean encaja aquí)
    paths = sorted(
        p for p in glob.glob(os.path.join(args.input_dir, "reviews_*"))
        if os.path.basename(p) != "reviews_full"
    )
    if not paths:
        raise SystemExit(f"[merge] No se han encontrado particiones en {args.input_dir}/reviews_*")

    spark = build_spark()

    df = read_many(spark, paths, args.input_format)

    # Métricas previas (útiles para ver cuántas quitamos)
    total_in = df.count()
    uniq_in = df.select(F.countDistinct("review_id")).first()[0]
    print(f"[merge] filas_in={total_in} | review_id_unicos_in={uniq_in}")

    # >>>>> DEDUP POR review_id (ÚNICO CAMBIO IMPORTANTE) <<<<<
    df = dedup_by_review_id(df)

    # Métricas y guardas
    total_out = df.count()
    uniq_out = df.select(F.countDistinct("review_id")).first()[0]
    print(f"[dedup] filas_out={total_out} | review_id_unicos_out={uniq_out} | eliminadas={total_in - total_out}")
    if total_out != uniq_out:
        raise SystemExit(f"[dedup][ERROR] review_id sigue duplicado: filas={total_out} | únicos={uniq_out}")

    # Coalesce al final (como hacías)
    df = df.coalesce(int(args.coalesce))

    writer = (
        df.write
        .mode("overwrite")
        .option("compression", "snappy")
        .option("maxRecordsPerFile", 500000)
    )
    if args.format == "delta":
        writer.format("delta").save(args.output)
    else:
        writer.parquet(args.output)

    print(f"✅ reviews_full ({args.format}) → {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()

