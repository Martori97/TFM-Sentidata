#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
from pyspark.sql import SparkSession, functions as F

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "landing_all"], default="single",
                   help="single: EDA de una sola fuente (comportamiento previo). landing_all: ejecuta los 3 EDA de landing.")
    p.add_argument("--input", help="Archivo/s o carpeta (admite globs) con los CSV/Parquet/Delta (solo en mode=single)")
    p.add_argument("--kind", choices=["csv","parquet","delta"], default="csv",
                   help="Tipo de input (solo en mode=single)")
    p.add_argument("--idcols", nargs="*", default=[], help="Columnas identificadoras (opcional)")
    p.add_argument("--textcols", nargs="*", default=[], help="Columnas de texto (opcional)")
    p.add_argument("--numcols", nargs="*", default=[], help="Columnas numéricas (opcional)")
    p.add_argument("--outdir", default="reports/eda/landing",
                   help="Carpeta destino (solo en mode=single)")
    return p.parse_args()

def make_spark():
    return (SparkSession.builder
            .appName("EDA Landing Enriched")
            .getOrCreate())

def read_df(spark, kind, path):
    if kind == "csv":
        return (spark.read
                .option("header", True)
                .option("inferSchema", True)
                .csv(path))
    elif kind == "parquet":
        return spark.read.parquet(path)
    elif kind == "delta":
        return spark.read.format("delta").load(path)
    else:
        raise ValueError(f"kind no soportado: {kind}")

def write_csv(df, outpath):
    (df.coalesce(1)
       .write.mode("overwrite")
       .option("header", True)
       .csv(outpath))

def run_eda(spark, input_path, kind, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = read_df(spark, kind, input_path)

    # Resumen
    nrows = df.count()
    schema_list = df.dtypes
    summary = {
        "rows": nrows,
        "columns": len(df.columns),
        "columns_list": [c for c,_ in schema_list],
        "schema": schema_list,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Nulos por columna
    nulls_row = (df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
                   .collect()[0].asDict())
    nulls_rows = [{"column": c, "nulls": int(nulls_row.get(c, 0)),
                   "null_pct": (float(nulls_row.get(c,0)) / nrows if nrows else 0.0)}
                  for c in df.columns]
    write_csv(spark.createDataFrame(nulls_rows), os.path.join(outdir, "nulls_by_column.csv"))

    # Distintos por columna
    distincts = []
    for c in df.columns:
        try:
            dc = df.select(c).distinct().count()
        except Exception:
            dc = None
        distincts.append({"column": c, "distinct": dc})
    write_csv(spark.createDataFrame(distincts), os.path.join(outdir, "distinct_counts.csv"))

    # Top-20 de columnas string
    string_cols = [c for c,t in schema_list if t == "string"]
    top_dir = os.path.join(outdir, "top_values")
    os.makedirs(top_dir, exist_ok=True)
    for c in string_cols:
        top = df.groupBy(c).count().orderBy(F.desc("count")).limit(20)
        write_csv(top, os.path.join(top_dir, f"{c}.csv"))

    # Stats numéricas
    numeric_types = {"int", "bigint", "double", "float", "long", "decimal", "smallint"}
    numeric_cols = [c for c,t in schema_list if t in numeric_types]
    if numeric_cols:
        stats = (df.select([F.col(c).cast("double").alias(c) for c in numeric_cols])
                   .summary("count","mean","stddev","min","max"))
        write_csv(stats, os.path.join(outdir, "numeric_stats.csv"))

    # Reglas simples
    rules = []
    if "rating" in df.columns:
        viol = df.filter(~F.col("rating").between(1,5)).count()
        rules.append({"rule": "rating_in_[1,5]", "violations": viol})
    with open(os.path.join(outdir, "rules.json"), "w") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False, default=str)

    print(f"[EDA] OK -> rows={nrows}, cols={len(df.columns)}, outdir={outdir}")

def main():
    args = parse_args()
    spark = make_spark()

    if args.mode == "single":
        if not args.input:
            raise SystemExit("--input es obligatorio en mode=single")
        run_eda(spark, args.input, args.kind, args.outdir)

    elif args.mode == "landing_all":
        # 1) Sephora reviews
        run_eda(
            spark,
            "data/landing/sephora/raw/reviews_*.csv",
            "csv",
            "reports/eda/landing_sephora_reviews",
        )
        # 2) Ulta reviews
        run_eda(
            spark,
            "data/landing/ulta/raw/Ulta Skincare Reviews.csv",
            "csv",
            "reports/eda/landing_ulta_reviews",
        )
        # 3) Sephora products
        run_eda(
            spark,
            "data/landing/sephora/raw/product_info.csv",
            "csv",
            "reports/eda/landing_sephora_products",
        )
        print("[EDA Landing All] DONE")

    spark.stop()

if __name__ == "__main__":
    main()

