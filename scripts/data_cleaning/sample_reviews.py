#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sample_reviews.py

- Modo normal: muestreo aleatorio o estratificado por --by y --frac.
- Modo balance (--balance): NEG=TODOS, NEU=TODOS, POS=tamaño de NEG
    * Si hay más POS que NEG -> downsample
    * Si hay menos POS que NEG -> upsample con reposición

Notas:
- --balance tiene prioridad y **ignora** --frac y --by.
- Si la columna de etiqueta no contiene 'negative|neutral|positive' pero existe 'rating',
  se deriva una etiqueta temporal de 3 clases a partir de 'rating' (<=2, ==3, >=4).
"""

import argparse
import os
import random

from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window
from pyspark import StorageLevel

# --- Delta opcional (si está instalado) ---
try:
    from delta import configure_spark_with_delta_pip
    HAS_DELTA = True
except Exception:
    HAS_DELTA = False

ALT_LABEL_CANDIDATES = ["label", "sentiment_category", "sentiment", "stars", "rating"]


# ---------- util parse ----------
def str2bool(v):
    """
    Acepta: true/false, yes/no, 1/0
    """
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    return v in ("true", "1", "yes", "y", "t", "si", "sí")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Ruta de entrada (carpeta Delta/Parquet)")
    p.add_argument("--output", required=True, help="Ruta de salida (carpeta)")
    p.add_argument("--frac", type=float, default=0.3, help="Fracción de muestreo (modo normal)")
    p.add_argument("--by", type=str, default=None, help="Columna de estratificación")
    p.add_argument("--format", type=str, default="delta", choices=["delta", "parquet"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coalesce", type=int, default=64, help="#particiones de salida (usamos repartition)")
    p.add_argument("--balance", type=str2bool, default=False, help="Modo balanceado 3 clases")
    p.add_argument("--text_col", type=str, default="review_text_clean", help="Columna de texto (si aplica)")
    p.add_argument("--id_col", type=str, default="review_id", help="Columna id (si aplica)")
    return p.parse_args()


# ---------- spark ----------
def build_spark(app_name="sample_reviews", driver_mem="6g", executor_mem="6g"):
    """
    SparkSession con Delta y parámetros para reducir presión de memoria.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        # Memoria
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        # Shuffle/particiones más finas
        .config("spark.sql.shuffle.partitions", "128")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        # Tuning de memoria
        .config("spark.memory.fraction", "0.6")
        .config("spark.storage.memoryFraction", "0.3")
        # Parquet seguro para heap
        .config("parquet.enable.dictionary", "false")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
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

    # Menos verbosidad
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------- io ----------
def read_input(spark: SparkSession, path: str, fmt: str):
    if fmt == "delta":
        return spark.read.format("delta").load(path)
    elif fmt == "parquet":
        return spark.read.parquet(path)
    else:
        raise ValueError(f"Formato no soportado: {fmt}")


def write_output(df, out_path, fmt="delta", n_out_partitions=64):
    """
    Escribe con repartition + persist(DISK_ONLY) + materialización previa
    para reducir picos de memoria en el write.
    """
    # Reparticiona uniformemente (mejor que coalesce para evitar particiones desbalanceadas)
    df = df.repartition(int(max(1, n_out_partitions)))
    # Materializa a disco antes del write
    df = df.persist(StorageLevel.DISK_ONLY)
    _ = df.count()  # fuerza evaluación

    writer = df.write.mode("overwrite")
    if fmt == "delta":
        writer.format("delta").save(out_path)
    elif fmt == "parquet":
        writer.parquet(out_path)
    else:
        raise ValueError(f"Formato no soportado: {fmt}")


# ---------- helpers de etiquetas ----------
def ensure_three_class_label(df, label_col: str):
    """
    Garantiza que label_col tenga valores en {'negative','neutral','positive'}.
    Si label_col == 'rating' (1..5), deriva 3 clases:
      <=2 -> negative, ==3 -> neutral, >=4 -> positive
    """
    # Si ya contiene strings 3 clases, devolvemos tal cual
    sample_vals = [r[label_col] for r in df.select(label_col).dropna().limit(1000).collect()]
    sample_vals_str = set(map(lambda x: str(x).lower(), sample_vals))

    triples = {"negative", "neutral", "positive"}
    if any(v in triples for v in sample_vals_str):
        # Asumimos ya correcto
        return df, label_col

    # Si es rating/numérico, derivamos
    if label_col.lower() == "rating":
        df2 = df.withColumn(
            "__label_bal",
            F.when(F.col("rating") <= F.lit(2), F.lit("negative"))
             .when(F.col("rating") == F.lit(3), F.lit("neutral"))
             .otherwise(F.lit("positive"))
        )
        return df2, "__label_bal"

    # Si es stars/numérico de 1..5
    if label_col.lower() == "stars":
        df2 = df.withColumn(
            "__label_bal",
            F.when(F.col("stars") <= F.lit(2), F.lit("negative"))
             .when(F.col("stars") == F.lit(3), F.lit("neutral"))
             .otherwise(F.lit("positive"))
        )
        return df2, "__label_bal"

    # En último término: intenta mapear strings tipo 'neg','neu','pos'
    df2 = df.withColumn(
        "__label_bal",
        F.when(F.lower(F.col(label_col)).startswith("neg"), F.lit("negative"))
         .when(F.lower(F.col(label_col)).startswith("neu"), F.lit("neutral"))
         .when(F.lower(F.col(label_col)).startswith("pos"), F.lit("positive"))
         .otherwise(F.lit(None))
    )
    return df2, "__label_bal"


def pick_label_column(df):
    cols = df.columns
    # prioridad conocida
    for c in ["sentiment_category", "label", "sentiment"]:
        if c in cols:
            return c
    # fallback: stars/rating
    for c in ["rating", "stars"]:
        if c in cols:
            return c
    # último recurso: primera entre candidatos
    for c in ALT_LABEL_CANDIDATES:
        if c in cols:
            return c
    raise ValueError("No se encontró ninguna columna de etiqueta candidata.")


# ---------- sampling ----------
def sample_balanced_three_class(df, label_col, seed=42):
    """
    NEG = todos, NEU = todos, POS = igual a NEG (downsample o upsample con reposición).
    """
    neg = df.filter(F.col(label_col) == "negative")
    neu = df.filter(F.col(label_col) == "neutral")
    pos = df.filter(F.col(label_col) == "positive")

    n_neg = neg.count()
    n_neu = neu.count()
    n_pos = pos.count()

    if n_neg == 0 or n_neu == 0 or n_pos == 0:
        raise RuntimeError(f"[balance] Alguna clase está vacía: neg={n_neg}, neu={n_neu}, pos={n_pos}")

    # POS → tamaño de NEG
    if n_pos >= n_neg:
        frac = n_neg / n_pos
        pos_bal = pos.sample(withReplacement=False, fraction=frac, seed=seed)
    else:
        # upsample con reposición
        reps = int(n_neg // max(1, n_pos))
        resto = (n_neg % max(1, n_pos)) / max(1, n_pos)
        parts = [pos] * max(1, reps)
        if resto > 0:
            parts.append(pos.sample(withReplacement=True, fraction=resto, seed=seed))
        pos_bal = parts[0]
        for p in parts[1:]:
            pos_bal = pos_bal.unionByName(p)

    sampled = neg.unionByName(neu).unionByName(pos_bal)
    return sampled


def sample_stratified(df, by_col, frac, seed=42):
    """
    Muestreo estratificado por columna 'by_col' con fracción 'frac'.
    """
    # usa sampleBy si cardinalidad razonable; si no, fallback por Window+row_number
    distinct_vals = [r[by_col] for r in df.select(by_col).distinct().limit(10000).collect()]
    if len(distinct_vals) <= 1000:
        fractions = {v: frac for v in distinct_vals}
        return df.sampleBy(by_col, fractions=fractions, seed=seed)
    else:
        w = Window.partitionBy(by_col).orderBy(F.rand(seed))
        return (
            df.withColumn("__rn", F.row_number().over(w))
              .withColumn("__cnt", F.count(F.lit(1)).over(Window.partitionBy(by_col)))
              .withColumn("__keep", F.col("__rn") <= (F.col("__cnt") * F.lit(frac)))
              .filter(F.col("__keep"))
              .drop("__rn", "__cnt", "__keep")
        )


def sample_simple(df, frac, seed=42):
    return df.sample(withReplacement=False, fraction=frac, seed=seed)


# ---------- main ----------
def main():
    args = parse_args()
    random.seed(args.seed)

    spark = build_spark()

    # Lee entrada (detecta delta/parquet por bandera --format)
    df = read_input(spark, args.input, args.format)

    # Selección de etiqueta
    label_col = pick_label_column(df)

    # Si no es 3 clases, derivar desde rating/stars o mapear
    df, label_col = ensure_three_class_label(df, label_col)

    # Normaliza valores posibles
    df = df.withColumn(
        label_col,
        F.when(F.col(label_col).isin("negative", "neutral", "positive"), F.col(label_col))
         .otherwise(F.col(label_col))
    )

    # Si vamos a balancear, se ignoran --frac y --by
    if args.balance:
        mode = "balance"
        sampled = sample_balanced_three_class(df, label_col, seed=args.seed)
    else:
        # Modo normal
        mode = "normal"
        if args.by:
            sampled = sample_stratified(df, args.by, args.frac, seed=args.seed)
        else:
            sampled = sample_simple(df, args.frac, seed=args.seed)

    # Imprime distribución
    total = sampled.count()
    dist = (
        sampled.groupBy(label_col).count()
        .withColumn("pct", F.col("count") / F.lit(total) * 100.0)
        .orderBy(label_col)
    )
    print(f"[sample] modo={mode} | total={total}")
    dist.show(truncate=False)

    # Escribe salida robusta
    write_output(sampled, args.output, fmt=args.format, n_out_partitions=int(args.coalesce))
    print(f"✅ sample → {args.output} [{args.format}]")

    spark.stop()


if __name__ == "__main__":
    main()

