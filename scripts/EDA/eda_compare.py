#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

from pyspark.sql import SparkSession, DataFrame, functions as F

# --- plotting/tokenizer (opcionales) ---
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging
    HAS_HF = True
except Exception:
    HAS_HF = False


# =========================
# Util
# =========================
def to_bool(x: str) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


# =========================
# Spark / Delta helpers
# =========================
def build_spark(app: str = "eda_compare") -> SparkSession:
    """Crea SparkSession con extensiones Delta si están disponibles."""
    try:
        from delta import configure_spark_with_delta_pip
        builder = (
            SparkSession.builder
            .appName(app)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
        )
        return configure_spark_with_delta_pip(builder).getOrCreate()
    except Exception:
        return (
            SparkSession.builder
            .appName(app)
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
        ).getOrCreate()


def is_delta(path: str) -> bool:
    return os.path.exists(os.path.join(path, "_delta_log"))


def read_any(spark: SparkSession, path: str, force: Optional[str]) -> DataFrame:
    """Lee Delta/Parquet con autodetección o formato forzado."""
    fmt = "delta" if (force == "delta" or (force == "auto" and is_delta(path))) else (
          "parquet" if force in {"parquet", "auto"} else force)
    return spark.read.format(fmt).load(path)


# =========================
# Lógica EDA
# =========================
def ensure_by_column(df: DataFrame, by: Optional[str]) -> Tuple[DataFrame, str]:
    """
    Garantiza columna de clase para comparar.
    - Si `by` existe, se usa.
    - Si no, probar ['sentiment_category','sentiment','label'].
    - Si no, derivar 'sentiment_category' desde 'stars'/'rating' si existen.
    """
    cols = set(df.columns)
    if by and by in cols:
        return df, by

    for c in ["sentiment_category", "sentiment", "label"]:
        if c in cols:
            return df, c

    num_col = "stars" if "stars" in cols else ("rating" if "rating" in cols else None)
    if num_col:
        df2 = df.withColumn(
            "sentiment_category",
            F.when(F.col(num_col) <= 2, F.lit("negative"))
             .when(F.col(num_col) == 3, F.lit("neutral"))
             .otherwise(F.lit("positive"))
        )
        return df2, "sentiment_category"

    raise SystemExit(
        f"[eda_compare] No encuentro columna de clases (by='{by}'). "
        f"Disponibles: {sorted(cols)}"
    )


def counts_by(df: DataFrame, by_col: str) -> DataFrame:
    """
    Recuentos y porcentaje por clase dentro del propio dataset.
    Implementado sin ventanas para evitar avisos y single-partition.
    """
    counts = df.groupBy(by_col).count().cache()
    total = counts.agg(F.sum("count").alias("total"))
    out = (
        counts.join(total)
              .withColumn("pct", F.col("count") / F.col("total"))
              .drop("total")
              .orderBy(F.desc("count"))
    )
    return out


def save_spark_df_csv(sdf: DataFrame, path: Path):
    """Guarda un único CSV (coalesce 1) para consulta fácil."""
    (sdf.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(path)))


def detect_text_col(df: DataFrame) -> Optional[str]:
    for c in ["review_text_clean", "review_text"]:
        if c in df.columns:
            return c
    return None


def tokenize_lengths(rows, tokenizer_name: str, text_key: str,
                     max_len: Optional[int], clip_tokens: bool) -> Optional[list]:
    if not HAS_HF or not HAS_MPL:
        print("[eda_compare] Aviso: no hay transformers/matplotlib; omito tokenización.")
        return None

    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    lens = []
    for r in rows:
        t = r[text_key]
        if t is None:
            continue
        enc = tok(t, add_special_tokens=True, truncation=False)
        L = len(enc["input_ids"])
        if clip_tokens and max_len is not None:
            L = min(L, max_len)
        lens.append(L)
    return lens


def plot_hist(data: list, title: str, out_png: Path):
    if not HAS_MPL or not data:
        return
    import matplotlib.pyplot as plt  # seguridad por si HAS_MPL cambió
    plt.figure()
    plt.hist(data, bins=60)
    plt.title(title)
    plt.xlabel("token length")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def run_compare(
    a_path: str,
    b_path: str,
    outdir: str,
    by: Optional[str],
    input_format: str,
    tokenizer_name: Optional[str],
    sample_rows: int,
    max_len: Optional[int],
    clip_tokens: bool,
    silence_tokenizer_warnings: bool,
):
    if silence_tokenizer_warnings and HAS_HF:
        hf_logging.set_verbosity_error()

    spark = build_spark()
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    # Leer datasets
    A = read_any(spark, a_path, input_format)
    B = read_any(spark, b_path, input_format)

    # Asegurar columna de clase y alinear nombres
    A, by_col = ensure_by_column(A, by)
    B, _      = ensure_by_column(B, by_col)

    # --- tamaños ---
    nA = A.count()
    nB = B.count()
    (outdir_p / "sizes.json").write_text(json.dumps({"A": nA, "B": nB}, indent=2))

    # --- distribuciones por clase (counts + pct, separadas por dataset) ---
    A_cls = counts_by(A, by_col).withColumn("dataset", F.lit("A"))
    B_cls = counts_by(B, by_col).withColumn("dataset", F.lit("B"))
    classes = A_cls.unionByName(B_cls, allowMissingColumns=True)
    save_spark_df_csv(classes, outdir_p / "class_distribution")

    # Pivots lado a lado (counts y pct)
    pivot_counts = (
        classes.groupBy("dataset")
               .pivot(by_col)
               .agg(F.first("count").cast("long"))
               .orderBy("dataset")
    )
    save_spark_df_csv(pivot_counts, outdir_p / "class_counts_pivot")

    pivot_pct = (
        classes.groupBy("dataset")
               .pivot(by_col)
               .agg(F.first("pct"))
               .orderBy("dataset")
    )
    save_spark_df_csv(pivot_pct, outdir_p / "class_pct_pivot")

    # --- Longitud de tokens (opcional) ---
    if tokenizer_name:
        text_col_A = detect_text_col(A)
        text_col_B = detect_text_col(B)

        if text_col_A and text_col_B:
            A_samp = (A.select(text_col_A)
                        .where(F.col(text_col_A).isNotNull())
                        .limit(sample_rows)
                        .collect())
            B_samp = (B.select(text_col_B)
                        .where(F.col(text_col_B).isNotNull())
                        .limit(sample_rows)
                        .collect())

            A_len = tokenize_lengths(A_samp, tokenizer_name, text_col_A, max_len, clip_tokens)
            B_len = tokenize_lengths(B_samp, tokenizer_name, text_col_B, max_len, clip_tokens)

            suffix = f"{tokenizer_name}"
            if clip_tokens and max_len is not None:
                suffix += f"_clip{max_len}"

            if A_len:
                plot_hist(A_len, f"A token lengths ({suffix})",
                          outdir_p / f"A_token_lengths_{suffix}.png")
            if B_len:
                plot_hist(B_len, f"B token lengths ({suffix})",
                          outdir_p / f"B_token_lengths_{suffix}.png")

            # Stats básicos
            import statistics as st
            stats = {}
            if A_len:
                A_sorted = sorted(A_len)
                stats["A"] = {
                    "count": len(A_len),
                    "mean": float(st.mean(A_len)),
                    "median": float(st.median(A_len)),
                    "p95": float(A_sorted[int(0.95 * len(A_sorted)) - 1]) if len(A_sorted) > 0 else None
                }
            if B_len:
                B_sorted = sorted(B_len)
                stats["B"] = {
                    "count": len(B_len),
                    "mean": float(st.mean(B_len)),
                    "median": float(st.median(B_len)),
                    "p95": float(B_sorted[int(0.95 * len(B_sorted)) - 1]) if len(B_sorted) > 0 else None
                }
            (outdir_p / "token_length_stats.json").write_text(json.dumps(stats, indent=2))
        else:
            print(f"[eda_compare] No encuentro columna de texto en A/B (busco 'review_text_clean' o 'review_text'). Omito tokenización.")

    # --- Export de 10 filas de ejemplo de cada dataset ---
    def save_head(df: DataFrame, name: str):
        cols = [c for c in [
            "review_id", "product_id", "brand_name", "product_name",
            "rating", "stars", "review_text_clean", "review_text", by_col
        ] if c in df.columns]
        head = df.select(*cols).limit(10)
        save_spark_df_csv(head, outdir_p / f"head_{name}")

    save_head(A, "A")
    save_head(B, "B")

    print(f"✅ EDA compare listo en {outdir_p} (by={by_col}, sizes: A={nA:,}, B={nB:,})")
    spark.stop()


# =========================
# CLI
# =========================
def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Ruta dataset A (Delta/Parquet)")
    ap.add_argument("--b", required=True, help="Ruta dataset B (Delta/Parquet)")
    ap.add_argument("--outdir", required=True, help="Carpeta de salida")
    ap.add_argument("--by", default=None, help="Columna de clase (auto-deriva si falta)")
    ap.add_argument("--input-format", default="auto", choices=["auto", "delta", "parquet"])
    ap.add_argument("--tokenizer", default=None,
                    help="Tokenizer HF para longitudes (opcional). Ej: albert-base-v2")
    ap.add_argument("--sample-rows", type=int, default=100000,
                    help="Máx. filas a tokenizar para histograma (por dataset)")
    ap.add_argument("--max-len", type=int, default=128,
                    help="Longitud tope para el histograma (p. ej. 128/192/224/512)")
    # Booleanos como strings para usarlos desde params.yaml
    ap.add_argument("--clip-tokens", default="false",
                    help="true/false: recortar longitudes al tope (--max-len)")
    ap.add_argument("--silence-tokenizer-warnings", default="true",
                    help="true/false: silenciar warnings de transformers")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    run_compare(
        a_path=args.a,
        b_path=args.b,
        outdir=args.outdir,
        by=args.by,
        input_format=args.input_format,
        tokenizer_name=args.tokenizer,
        sample_rows=args.sample_rows,
        max_len=args.max_len,
        clip_tokens=to_bool(args.clip_tokens),
        silence_tokenizer_warnings=to_bool(args.silence_tokenizer_warnings),
    )





