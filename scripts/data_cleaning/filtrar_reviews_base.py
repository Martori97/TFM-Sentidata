# -*- coding: utf-8 -*-
"""
Genera el dataset base SOLO de Sephora a partir de trusted (sin spaCy).
- Append por lotes -> Delta canónico (incluye tokens/lemmas ligeros)
- Deduplicación por buckets SOLO para exportar Parquet/CSV (no tocamos el Delta canónico)
- Salidas: Delta (canónico) + Parquet (dedup) + CSV (dedup con tokens como string)

Ejecutar:
    python scripts/data_cleaning/filtrar_reviews_base.py
"""

import os
import shutil
from pyspark.sql import SparkSession, functions as F, types as T
from delta import configure_spark_with_delta_pip

# ---------------- Spark ----------------
builder = (
    SparkSession.builder
    .appName("FiltrarReviewsBase-SephoraOnly-NoSpaCy")
    .config("spark.driver.memory", "5g")
    .config("spark.executor.memory", "5g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "4")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))  # 64MB
    .config("spark.sql.broadcastTimeout", "600")
    .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
    # Delta
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Aseguramos que los workers ven utils_text.py
spark.sparkContext.addPyFile("scripts/data_cleaning/utils_text.py")
from utils_text import clean_text, tokenize_text, lemmatize_text  # noqa: E402

# UDFs ligeras
clean_udf = F.udf(clean_text, T.StringType())
tok_udf   = F.udf(tokenize_text, T.ArrayType(T.StringType()))
lem_udf   = F.udf(lemmatize_text, T.StringType())

# ---------------- Rutas ----------------
SEPH_TRUST = "data/trusted/sephora_clean"
OUT_DELTA  = "data/exploitation/modelos_input/reviews_base_delta"    # canónico (sin dedup final)
OUT_PARQ   = "data/exploitation/modelos_input/reviews_base_parquet"  # export dedup incremental
OUT_CSV    = "data/exploitation/modelos_input/reviews_base_csv"      # export dedup (desde Parquet)

# Columna de texto candidata
TEXT_CANDS = ["review_text", "review_body", "text", "review", "content", "reviewText", "body", "comments"]

# Nº de buckets para deduplicar por lotes
N_BUCKETS = int(os.environ.get("DEDUP_BUCKETS", "12"))


def _is_delta_table(path: str) -> bool:
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, "_delta_log"))


def _prepare_output_dirs():
    # Limpia SOLO exportaciones. El Delta canónico se sobreescribe en el primer append.
    for p in [OUT_PARQ, OUT_CSV]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
    os.makedirs(os.path.dirname(OUT_DELTA), exist_ok=True)


def _ensure_review_id(df):
    """
    Si no existe review_id, generamos uno determinista por hash del texto original.
    """
    if "review_id" in df.columns:
        return df
    if "review_text" in df.columns:
        return df.withColumn("review_id", F.sha2(F.col("review_text").cast("string"), 256))
    cols = [F.col(c).cast("string") for c in df.columns]
    return df.withColumn("review_id", F.sha2(F.concat_ws("||", *cols), 256))


def _normalize_and_enrich(df):
    # localizar columna de texto
    text_col = next((c for c in TEXT_CANDS if c in df.columns), None)
    if text_col is None:
        return None
    if text_col != "review_text":
        df = df.withColumnRenamed(text_col, "review_text")

    # tipos
    if "rating" in df.columns:
        df = df.withColumn("rating", F.col("rating").cast("double"))

    # id
    df = _ensure_review_id(df)

    # limpieza + features
    df = df.withColumn("review_text_clean", clean_udf(F.col("review_text")))
    df = df.withColumn("text_len", F.length(F.col("review_text_clean")))
    df = df.withColumn("tokens", tok_udf(F.col("review_text_clean")))
    df = df.withColumn("lemmas", lem_udf(F.col("review_text_clean")))

    # columnas deseadas
    wanted = ["review_id", "rating", "review_text", "review_text_clean", "text_len", "tokens", "lemmas"]
    existing = [c for c in wanted if c in df.columns]
    return df.select(*existing)


def _append_canon_from_trusted():
    """
    Hace append por lotes al Delta canónico desde todas las tablas de trusted.
    """
    if not os.path.isdir(SEPH_TRUST):
        raise RuntimeError(f"No existe {SEPH_TRUST}. Ejecuta antes la limpieza trusted.")

    tables = sorted(os.listdir(SEPH_TRUST))
    wrote_any = False

    for name in tables:
        in_path = os.path.join(SEPH_TRUST, name)
        if not _is_delta_table(in_path):
            continue

        df = spark.read.format("delta").load(in_path)
        df_norm = _normalize_and_enrich(df)
        if df_norm is None:
            print(f"[WARN] {in_path} sin columna de texto; se omite.")
            continue

        df_write = df_norm.coalesce(2)

        if not wrote_any:
            df_write.write.format("delta").mode("overwrite").save(OUT_DELTA)
            wrote_any = True
        else:
            df_write.write.format("delta").mode("append").save(OUT_DELTA)

        print(f"[OK] Append -> {OUT_DELTA} desde {name}")

    if not wrote_any:
        raise RuntimeError("No se escribió ninguna tabla. Revisa trusted.")


def _export_deduplicated():
    """
    Dedup por buckets y exportación incremental a Parquet. Luego CSV desde el Parquet deduplicado.
    No tocamos el Delta canónico.
    """
    base = spark.read.format("delta").load(OUT_DELTA)

    # Clave de deduplicación: contenido limpio + rating
    base = base.withColumn(
        "dedup_key",
        F.sha2(F.concat_ws("||", F.col("review_text_clean").cast("string"),
                           F.col("rating").cast("string")), 256)
    )

    # Bucket para procesar en trozos más pequeños
    base = base.withColumn("bucket", F.pmod(F.abs(F.hash(F.col("dedup_key"))), F.lit(N_BUCKETS)))

    # Borramos exportaciones previas
    if os.path.exists(OUT_PARQ):
        shutil.rmtree(OUT_PARQ, ignore_errors=True)
    if os.path.exists(OUT_CSV):
        shutil.rmtree(OUT_CSV, ignore_errors=True)

    # Procesar bucket a bucket y APPEND al Parquet
    for b in range(N_BUCKETS):
        subset = base.where(F.col("bucket") == F.lit(b))

        subset_dedup = (
            subset
            .dropDuplicates(["review_id"])
            .dropDuplicates(["dedup_key"])
            .select("review_id", "rating", "review_text", "review_text_clean", "text_len", "tokens", "lemmas")
        )

        subset_dedup.coalesce(1).write.mode("append").parquet(OUT_PARQ)

    print(f"[OK] Export deduplicada (Parquet) -> {OUT_PARQ}")

    # CSV: re-leemos el Parquet y transformamos tokens (ARRAY) -> STRING
    final_df = spark.read.parquet(OUT_PARQ)
    final_csv = (
        final_df
        .withColumn("tokens", F.concat_ws(" ", F.col("tokens")))  # <-- conversión requerida para CSV
        .select("review_id", "rating", "review_text", "review_text_clean", "text_len", "tokens", "lemmas")
    )
    final_csv.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT_CSV)
    print(f"[OK] Export deduplicada (CSV) -> {OUT_CSV}")


def generar_reviews_base_append():
    _prepare_output_dirs()
    _append_canon_from_trusted()
    _export_deduplicated()


if __name__ == "__main__":
    generar_reviews_base_append()
