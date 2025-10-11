#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
from pyspark.sql import SparkSession, functions as F

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "trusted_all"], default="single",
                   help="single: EDA de una sola fuente. trusted_all: ejecuta varios EDA de trusted.")
    p.add_argument("--input", help="Ruta (archivo/carpeta) o patrón (solo en mode=single)")
    p.add_argument("--kind", choices=["csv","parquet","delta"], default="delta",
                   help="Tipo de input (solo en mode=single). Por defecto delta en trusted.")
    p.add_argument("--idcols", nargs="*", default=[], help="Columnas identificadoras (opcional)")
    p.add_argument("--textcols", nargs="*", default=[], help="Columnas de texto (opcional)")
    p.add_argument("--numcols", nargs="*", default=[], help="Columnas numéricas (opcional)")
    p.add_argument("--outdir", default="reports/eda/trusted_single",
                   help="Carpeta destino (solo en mode=single)")
    # Parámetros de control para heavy ops
    p.add_argument("--sample_frac_for_tops", type=float, default=0.1,
                   help="Fracción de muestreo para top_values cuando nrows es grande (0<frac<=1).")
    p.add_argument("--sample_threshold_rows", type=int, default=500_000,
                   help="Si filas > threshold, se aplica muestreo para top_values.")
    p.add_argument("--approx_rsd", type=float, default=0.05,
                   help="RSD para approx_count_distinct (0.05 = ~5%).")
    return p.parse_args()

# -------- Spark con Delta habilitado + ajustes de memoria --------
def make_spark():
    """
    Crea una SparkSession con Delta Lake y ajustes para reducir OOM en lectura Parquet.
    Requiere delta-spark (pip install delta-spark==3.1.0).
    """
    try:
        from delta import configure_spark_with_delta_pip
        builder = (
            SparkSession.builder
            .appName("EDA Trusted Enriched (Delta)")
            # Delta
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")
            # Memoria / lectura parquet
            .config("spark.sql.parquet.enableVectorizedReader", "false")  # baja presión de heap
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))  # 64MB
            .config("spark.sql.files.openCostInBytes", str(8 * 1024 * 1024))     # 8MB
            .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
            # Sube si tu máquina lo permite (en local, driver=executor)
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "4g")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        return spark
    except Exception as e:
        print(f"[make_spark] Aviso: fallback sin configure_spark_with_delta_pip ({e})")
        spark = (
            SparkSession.builder
            .appName("EDA Trusted Enriched (Delta-Fallback)")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")
            .config("spark.sql.parquet.enableVectorizedReader", "false")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
            .config("spark.sql.files.openCostInBytes", str(8 * 1024 * 1024))
            .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "4g")
            .getOrCreate()
        )
        return spark

# -------- utilidades --------
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

def run_eda(spark, input_path, kind, outdir,
            sample_frac_for_tops=0.1, sample_threshold_rows=500_000, approx_rsd=0.05):
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
        "source": {"path": input_path, "kind": kind},
        "notes": {
            "top_values_sampled": bool(nrows > sample_threshold_rows and 0 < sample_frac_for_tops < 1.0),
            "approx_distinct_rsd": approx_rsd
        }
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Nulos por columna (1 solo collect)
    try:
        nulls_row = (df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
                       .collect()[0].asDict())
    except Exception as e:
        print(f"[EDA Trusted] Nulls fallo -> {e}")
        nulls_row = {c: None for c in df.columns}
    nulls_rows = [{"column": c, "nulls": (int(nulls_row.get(c, 0)) if nulls_row.get(c) is not None else None),
                   "null_pct": ((float(nulls_row.get(c,0)) / nrows) if (nrows and nulls_row.get(c) is not None) else None)}
                  for c in df.columns]
    write_csv(spark.createDataFrame(nulls_rows), os.path.join(outdir, "nulls_by_column.csv"))

    # Distintos por columna (approx para evitar OOM)
    distincts = []
    for c in df.columns:
        try:
            dc = df.agg(F.approx_count_distinct(F.col(c), rsd=approx_rsd).alias("d")).collect()[0]["d"]
        except Exception as e:
            print(f"[EDA Trusted] approx_count_distinct fallo en {c} -> {e}")
            try:
                dc = df.select(c).distinct().count()
            except Exception:
                dc = None
        distincts.append({"column": c, "distinct_approx": dc, "rsd": approx_rsd})
    write_csv(spark.createDataFrame(distincts), os.path.join(outdir, "distinct_counts_approx.csv"))

    # Top-20 de columnas string (muestreando si el dataset es grande)
    string_cols = [c for c,t in schema_list if t == "string"]
    top_dir = os.path.join(outdir, "top_values")
    os.makedirs(top_dir, exist_ok=True)

    # Selecciona df_base para top-values (muestra si grande)
    if nrows > sample_threshold_rows and 0 < sample_frac_for_tops < 1.0:
        try:
            df_base = df.sample(False, sample_frac_for_tops, seed=42)
        except Exception as e:
            print(f"[EDA Trusted] sample fallo -> {e}; uso df completo para top_values")
            df_base = df
    else:
        df_base = df

    for c in string_cols:
        try:
            top = (df_base.groupBy(c).count().orderBy(F.desc("count")).limit(20))
            write_csv(top, os.path.join(top_dir, f"{c}.csv"))
        except Exception as e:
            print(f"[EDA Trusted] top_values fallo en {c} -> {e}")

    # Stats numéricas (summary agregado)
    numeric_types = {"int", "bigint", "double", "float", "long", "decimal", "smallint", "short"}
    numeric_cols = [c for c,t in schema_list if t in numeric_types]
    if numeric_cols:
        try:
            stats = (df.select([F.col(c).cast("double").alias(c) for c in numeric_cols])
                       .summary("count","mean","stddev","min","max"))
            write_csv(stats, os.path.join(outdir, "numeric_stats.csv"))
        except Exception as e:
            print(f"[EDA Trusted] numeric_stats fallo -> {e}")

    # Reglas simples
    rules = []
    if "rating" in df.columns:
        try:
            viol = df.filter(~F.col("rating").between(1,5)).count()
        except Exception as e:
            print(f"[EDA Trusted] regla rating fallo -> {e}")
            viol = None
        rules.append({"rule": "rating_in_[1,5]", "violations": viol})
    with open(os.path.join(outdir, "rules.json"), "w") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False, default=str)

    print(f"[EDA Trusted] OK -> rows={nrows}, cols={len(df.columns)}, outdir={outdir}")

# -------- main --------
def main():
    args = parse_args()
    spark = make_spark()

    if args.mode == "single":
        if not args.input:
            raise SystemExit("--input es obligatorio en mode=single")
        run_eda(
            spark, args.input, args.kind, args.outdir,
            sample_frac_for_tops=args.sample_frac_for_tops,
            sample_threshold_rows=args.sample_threshold_rows,
            approx_rsd=args.approx_rsd,
        )

    elif args.mode == "trusted_all":
        # 1) reviews_full (Delta)
        run_eda(
            spark,
            "data/trusted/reviews_full",
            "delta",
            "reports/eda/trusted_reviews_full",
        )
        # 2) reviews_product_info_clean_full (Delta)
        run_eda(
            spark,
            "data/trusted/reviews_product_info_clean_full",
            "delta",
            "reports/eda/trusted_reviews_product_info_clean_full",
        )
        # 3) reviews_sample_30pct (Delta)
        run_eda(
            spark,
            "data/trusted/reviews_sample_30pct",
            "delta",
            "reports/eda/trusted_reviews_sample_30pct",
        )
        print("[EDA Trusted All] DONE")

    spark.stop()

if __name__ == "__main__":
    main()


