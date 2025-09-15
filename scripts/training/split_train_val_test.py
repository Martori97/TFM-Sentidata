#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os
from pyspark.sql import SparkSession, functions as F

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Parquet (archivo o carpeta) con el SAMPLE")
    p.add_argument("--outdir", required=True, help="Carpeta destino: genera train/, val/, test/")
    p.add_argument("--by", default="sentiment_category", help="Columna para estratificar")
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id-col", default="review_id")
    p.add_argument("--text-col", default="review_text")
    p.add_argument("--rating-col", default="rating")
    return p.parse_args()

def derive_sentiment_from_rating(df, target_col="sentiment_category", rating_col="rating"):
    return df.withColumn(
        target_col,
        F.when((F.col(rating_col) >= 1) & (F.col(rating_col) <= 2), F.lit("neg"))
         .when(F.col(rating_col) == 3, F.lit("neu"))
         .when((F.col(rating_col) >= 4) & (F.col(rating_col) <= 5), F.lit("pos"))
         .otherwise(F.lit(None))
    )

def main():
    a = parse_args()
    spark = (SparkSession.builder
             .appName("split_train_val_test_from_sample")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Leer parquet (archivo o carpeta)
    df = spark.read.parquet(a.input)

    # Asegurar columna ID
    if a.id_col not in df.columns:
        df = df.withColumn(a.id_col, F.xxhash64(a.text_col).cast("string"))

    # Derivar columna de estrato si falta y hay rating
    by_col = a.by
    if by_col not in df.columns:
        if by_col == "sentiment_category" and a.rating_col in df.columns:
            df = derive_sentiment_from_rating(df, target_col=by_col, rating_col=a.rating_col)
        else:
            cols = ", ".join(df.columns)
            raise SystemExit(f"[split] La columna '{by_col}' no existe. Columnas: {cols}")

    # Filtrar nulos del estrato y del texto
    df = df.filter(F.col(by_col).isNotNull() & F.col(a.text_col).isNotNull())

    # Fracciones
    test_frac = float(a.test_size)
    val_frac  = float(a.val_size)
    train_frac = 1.0 - test_frac - val_frac
    if train_frac <= 0:
        raise SystemExit("[split] La suma de test-size y val-size debe ser < 1.0")

    # Estratos
    classes = [r[0] for r in df.select(by_col).distinct().collect()]
    if not classes:
        raise SystemExit("[split] No hay valores en la columna de estrato")

    # TEST
    fracs_test = {c: test_frac for c in classes}
    test = df.sampleBy(by_col, fractions=fracs_test, seed=a.seed)

    remain = df.join(test.select(a.id_col), on=a.id_col, how="left_anti")

    # VAL (proporción relativa sobre remain)
    fracs_val = {c: val_frac / (1.0 - test_frac) for c in classes}
    val = remain.sampleBy(by_col, fractions=fracs_val, seed=a.seed + 1)

    train = remain.join(val.select(a.id_col), on=a.id_col, how="left_anti")

    # Guardar
    os.makedirs(a.outdir, exist_ok=True)
    train.coalesce(8).write.mode("overwrite").parquet(os.path.join(a.outdir, "train"))
    val.coalesce(8).write.mode("overwrite").parquet(os.path.join(a.outdir, "val"))
    test.coalesce(8).write.mode("overwrite").parquet(os.path.join(a.outdir, "test"))

    # Manifest con contadores y % por clase
    def dist_json(df_):
        total = df_.count()
        by = (df_.groupBy(by_col).count()
                  .withColumn("pct", (F.col("count")/F.lit(total))*100.0)
                  .orderBy(by_col)
             ).collect()
        return {
            "total": int(total),
            "by_class": [{by_col: r[by_col], "count": int(r["count"]), "pct": float(r["pct"])} for r in by]
        }

    manifest = {
        "by": by_col,
        "splits": {
            "train": dist_json(train),
            "val":   dist_json(val),
            "test":  dist_json(test)
        }
    }
    with open(os.path.join(a.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("✅ split listo:", a.outdir)
    print("  rows:", {k: v["total"] for k,v in manifest["splits"].items()})

    spark.stop()

if __name__ == "__main__":
    main()
