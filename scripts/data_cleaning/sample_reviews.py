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

import os
import argparse
from typing import Optional

from pyspark.sql import SparkSession, DataFrame, functions as F

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
    Acepta: --balance, --balance true/false, yes/no, 1/0
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {v}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Tabla/parquet de entrada (ej. data/trusted/reviews_full)")
    p.add_argument("--output", required=True, help="Salida del sample (Delta/Parquet)")
    p.add_argument("--format", default="delta", choices=["delta", "parquet"], help="Formato de salida")
    p.add_argument("--input-format", default="auto", choices=["auto", "delta", "parquet"], help="Formato de entrada")
    p.add_argument("--coalesce", type=int, default=64, help="Nº de archivos de salida")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frac", type=float, default=1.0, help="Fracción si no se usa balance")
    p.add_argument("--by", default=None, help="Columna para muestreo estratificado (si no balance)")
    p.add_argument("--label-col", default=None, help="Columna de etiqueta (autodetecta si no se indica)")
    # acepta '--balance' y '--balance true/false'
    p.add_argument("--balance", nargs="?", const=True, default=False, type=str2bool,
                   help="Si es true (o presente): NEG=all, NEU=all, POS=NEG. Si false: usa --frac/--by.")
    return p.parse_args()


# ---------- spark ----------
def build_spark(app_name="sample_reviews"):
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.files.maxRecordsPerFile", "500000")
        .config("spark.sql.parquet.compression.codec", "snappy")
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
    return os.path.isdir(os.path.join(path, "_delta_log"))


def read_input(spark: SparkSession, path: str, input_format: str) -> DataFrame:
    if input_format == "delta" or (input_format == "auto" and detect_is_delta(path)):
        return spark.read.format("delta").load(path)
    return spark.read.parquet(path)


# ---------- helpers ----------
def choose_label_col(df: DataFrame, user: Optional[str]) -> str:
    if user and user in df.columns:
        return user
    for c in ALT_LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit("[sample] No se encontró columna de etiqueta (prueba --label-col).")


def ensure_threeclass_label_for_balance(df: DataFrame, label_col: str) -> (DataFrame, str):
    """
    Si label_col ya contiene 'negative|neutral|positive', lo usamos tal cual.
    Si no, y existe 'rating', derivamos __label_bal de rating (<=2 neg, ==3 neu, >=4 pos).
    Devuelve (df_con_posible_col_temporal, nombre_col_usable).
    """
    # ¿ya es una de las 3 clases?
    has_expected = df.filter(F.col(label_col).isin("negative", "neutral", "positive")).limit(1).count() > 0
    if has_expected:
        return df, label_col

    if "rating" in df.columns:
        df2 = df.withColumn(
            "__label_bal",
            F.when(F.col("rating") <= 2, F.lit("negative"))
             .when(F.col("rating") == 3, F.lit("neutral"))
             .otherwise(F.lit("positive"))
        )
        print(f"[sample] '{label_col}' no es 3-clases; usando etiqueta derivada de 'rating' -> __label_bal")
        return df2, "__label_bal"

    raise SystemExit("[sample][ERROR] La columna de etiqueta no es 3-clases y no existe 'rating' para derivarla. "
                     "Pasa --label-col adecuada o añade 'label'/'sentiment_category'.")


def balance_neg_neu_pos_to_neg(df: DataFrame, label_col: str, seed: int) -> DataFrame:
    """
    Mantiene todos los NEG y NEU.
    Ajusta POS = Nº NEG (down/upsample con reposición).
    """
    # asegurar que el label es 3-clases válido
    df, use_label = ensure_threeclass_label_for_balance(df, label_col)

    neg_df = df.filter(F.col(use_label) == "negative")
    neu_df = df.filter(F.col(use_label) == "neutral")
    pos_df = df.filter(F.col(use_label) == "positive")

    # conteos (sin pandas)
    counts = {r[use_label]: int(r["count"]) for r in df.groupBy(use_label).count().collect()}
    n_neg = counts.get("negative", 0)
    n_pos = counts.get("positive", 0)
    target_pos = n_neg

    if n_neg == 0:
        raise SystemExit("[sample][ERROR] No hay ejemplos 'negative'. No se puede balancear con la regla pos=neg.")

    if n_pos >= target_pos:
        # downsample exacto
        pos_bal = pos_df.orderBy(F.rand(seed)).limit(target_pos)
    else:
        # upsample con reposición
        times = target_pos // max(n_pos, 1)
        rem = target_pos - times * n_pos

        pos_bal = None
        for _ in range(max(times, 1)):
            pos_bal = pos_df if pos_bal is None else pos_bal.unionByName(pos_df)

        if rem > 0:
            pos_extra = pos_df.orderBy(F.rand(seed + 1)).limit(rem)
            pos_bal = pos_bal.unionByName(pos_extra)

    out = neg_df.unionByName(neu_df).unionByName(pos_bal)

    # si creamos __label_bal, no hace falta dejarlo en el output; el resto del pipeline usa sus propias columnas
    if "__label_bal" in out.columns:
        out = out.drop("__label_bal")

    return out.orderBy(F.rand(seed + 2))


def stratified_fraction(df: DataFrame, by_col: str, frac: float, seed: int) -> DataFrame:
    keys = [r[0] for r in df.select(by_col).distinct().collect()]
    fracs = {k: frac for k in keys}
    return df.sampleBy(by_col, fractions=fracs, seed=seed)


def write_output(df: DataFrame, out_path: str, fmt: str, coalesce: int):
    if coalesce and coalesce > 0:
        df = df.coalesce(coalesce)
    writer = (
        df.write
        .mode("overwrite")
        .option("compression", "snappy")
        .option("maxRecordsPerFile", 500000)
    )
    if fmt == "delta":
        writer.format("delta").save(out_path)
    else:
        writer.parquet(out_path)


# ---------- main ----------
def main():
    args = parse_args()
    spark = build_spark()

    df = read_input(spark, args.input, args.input_format)
    label_col = choose_label_col(df, args.label_col)

    if args.balance:
        sampled = balance_neg_neu_pos_to_neg(df, label_col=label_col, seed=args.seed)
        mode = "balance"
    else:
        # modo no balanceado: aleatorio o estratificado
        if args.frac >= 1.0:
            sampled = df
            mode = "all"
        else:
            if args.by and args.by in df.columns:
                sampled = stratified_fraction(df, by_col=args.by, frac=args.frac, seed=args.seed)
                mode = f"estratificado(by={args.by}, frac={args.frac})"
            else:
                sampled = df.sample(withReplacement=False, fraction=args.frac, seed=args.seed)
                mode = f"aleatorio(frac={args.frac})"

    # informe rápido
    total = sampled.count()
    dist = (
        sampled.groupBy(label_col).count()
        .withColumn("pct", F.col("count") / F.lit(total) * 100.0)
        .orderBy(label_col)
    )
    print(f"[sample] modo={mode} | total={total}")
    dist.show(truncate=False)

    write_output(sampled, args.output, fmt=args.format, coalesce=int(args.coalesce))
    print(f"✅ sample → {args.output} [{args.format}]")

    spark.stop()


if __name__ == "__main__":
    main()
