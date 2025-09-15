#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pyspark.sql import SparkSession, functions as F

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Ruta de entrada (parquet o delta)")
    p.add_argument("--output", required=True, help="Ruta de salida")
    p.add_argument("--frac",   type=float, default=0.30, help="Fracción total a muestrear")
    p.add_argument("--by",     type=str, default=None, help="Columna para estratificar (p.ej. labels)")
    p.add_argument("--seed",   type=int, default=42, help="Semilla")
    p.add_argument("--format", choices=["parquet","delta"], default="parquet", help="Formato de salida")
    p.add_argument("--coalesce", type=int, default=8, help="Particiones de salida")
    return p.parse_args()

def read_any(spark, path):
    # Detecta si hay _delta_log para leer delta
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "_delta_log")):
        return spark.read.format("delta").load(path)
    return spark.read.parquet(path)

def ensure_labels(df, by_col):
    cols = set(df.columns)
    target = by_col

    if by_col is None:
        return df, None

    if by_col in cols:
        return df, by_col

    # Normaliza nombres candidatos
    cand = [c for c in ["label", "label3", "labels"] if c in cols]
    if cand:
        c = cand[0]
        # Si es numérico 0/1/2, convierte a texto negative/neutral/positive
        if dict(df.dtypes)[c] in ("int", "bigint", "smallint", "tinyint", "double", "float"):
            df = df.withColumn(
                "labels",
                F.when(F.col(c) <= 0, F.lit("negative"))
                 .when(F.col(c) == 1, F.lit("neutral"))
                 .otherwise(F.lit("positive"))
            )
            return df, "labels"
        else:
            # asume ya es string usable
            return df.withColumnRenamed(c, "labels"), "labels"

    # Derivar de rating si existe
    if "rating" in cols:
        df = df.withColumn(
            "labels",
            F.when(F.col("rating") <= 2, F.lit("negative"))
             .when(F.col("rating") == 3, F.lit("neutral"))
             .otherwise(F.lit("positive"))
        )
        return df, "labels"

    # Si no podemos derivar, error claro
    raise ValueError(f"La columna '{by_col}' no existe y no se pudo derivar "
                     f"desde ['label','label3','rating']. Columnas disponibles: {sorted(cols)}")

def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .appName("sample_reviews")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    df = read_any(spark, args.input)

    # Asegurar columna de estratificación si se pide
    by_col = args.by
    if by_col:
        df, by_col = ensure_labels(df, by_col)
        # classes presentes
        classes = [r[0] for r in df.select(by_col).distinct().collect()]
        # Fracciones por clase (misma fracción global para cada clase)
        fractions = {c: float(args.frac) for c in classes}
        sampled = df.sampleBy(by_col, fractions=fractions, seed=args.seed)
    else:
        sampled = df.sample(withReplacement=False, fraction=float(args.frac), seed=args.seed)

    # Coalesce para menos archivos
    sampled = sampled.coalesce(int(args.coalesce))

    writer = sampled.write.mode("overwrite")
    if args.format == "delta":
        (writer.format("delta").save(args.output))
    else:
        (writer.parquet(args.output))

    print(f"✅ sample guardado en {args.output}")
    spark.stop()

if __name__ == "__main__":
    main()

