#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_absa_dataset.py

Pipeline ABSA end-to-end con 5 pasos:
1) Verificación de entradas (existencia, esquema, muestras).
2) Mapeo de spans -> categorías (via parquet mapeado).
3) Unión con sentimientos (columna categórica o numérica con umbrales).
4) Escritura del dataset final ABSA (parquet).
5) EDA + visualizaciones (CSV + PNG) con tolerancia a None.

Entradas esperadas para tu caso:
- --spans_parquet  reports/absa/ate_spans.parquet         (cols: review_id, aspect_span, ...)
- --map_parquet    reports/absa/ate_spans_mapped.parquet  (cols: review_id, span_text, aspect_norm, ...)
- --sentiment_parquet reports/albert_sample_30pct/eval/predictions.parquet (cols: review_id, pred_3, ...)

Ejemplo de uso (tu caso):
python scripts/absa/build_absa_dataset.py \
  --spans_parquet reports/absa/ate_spans.parquet \
  --map_parquet   reports/absa/ate_spans_mapped.parquet \
  --sentiment_parquet reports/albert_sample_30pct/eval/predictions.parquet \
  --id_col review_id \
  --span_text_col aspect_span \
  --category_col aspect_norm \
  --sentiment_col pred_3 \
  --output_dir reports/absa/final_all
"""

import os
import sys
import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ----------------------------
# Utilidades
# ----------------------------
def ensure_exists(path: str, label: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {label}: {path}")

def write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def normalize_text_col(df, col):
    return df.withColumn(col, F.trim(F.lower(F.col(col))))

def coalesce_first_nonnull(*cols):
    return F.coalesce(*cols)

def save_chart_bar(df_pd, x_col, y_col, title, out_path, rotate_xticks=False):
    """
    Guarda un gráfico de barras con matplotlib (headless) y tolerancia a None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_pd = df_pd.copy()
    # Evitar fallos con None en ejes / valores
    df_pd[x_col] = df_pd[x_col].fillna("<unmapped>").astype(str)
    df_pd[y_col] = df_pd[y_col].fillna(0)

    plt.figure()
    plt.bar(df_pd[x_col], df_pd[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if rotate_xticks:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()

def build_argparser():
    p = argparse.ArgumentParser(description="Construye dataset ABSA final (5 pasos completos).")
    # Entradas
    p.add_argument("--spans_parquet", required=True, help="Parquet con spans ATE.")
    p.add_argument("--map_parquet", default=None, help="Parquet con mapeo span->categoría (opcional, recomendado).")
    p.add_argument("--ontology_path", default=None, help="(No usar en tu caso) CSV/JSON ontología. Mantener None.")
    p.add_argument("--sentiment_parquet", required=True, help="Parquet con predicciones de sentimiento.")

    # Esquema / columnas
    p.add_argument("--id_col", default="review_id", help="Columna ID de review.")
    p.add_argument("--span_text_col", default="span_text", help="Columna del texto del aspecto en spans_parquet.")
    p.add_argument("--category_col", default="aspect_category", help="Nombre columna de categoría (vendrá del map).")
    p.add_argument("--sentiment_col", default="sentiment_category", help="Columna categórica de sentimiento si existe.")
    p.add_argument("--text_col", default=None, help="(Opcional) Columna texto de la review si existe en entradas.")

    # Fallback numérico -> sentimiento
    p.add_argument("--numeric_sentiment_col", default=None, help="Columna numérica (p.ej. stars) si no hay categórica.")
    p.add_argument("--neg_threshold", type=float, default=2.0, help="<= umbral => negative.")
    p.add_argument("--pos_threshold", type=float, default=4.0, help=">= umbral => positive.")

    # Salidas
    p.add_argument("--output_dir", required=True, help="Directorio de salida.")
    p.add_argument("--repartition", type=int, default=None, help="Reparticionado de salida (opcional).")
    # Visualizaciones
    p.add_argument("--top_k_categories_chart", type=int, default=30, help="Límite de categorías en el chart.")
    return p


def main():
    args = build_argparser().parse_args()

    # 0) Preparar carpetas de salida
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir   = os.path.join(args.output_dir, "logs")
    eda_dir    = os.path.join(args.output_dir, "eda")
    charts_dir = os.path.join(args.output_dir, "charts")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(eda_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # Spark
    spark = (
        SparkSession.builder
        .appName("ABSA-Build-5Steps")
        .config("spark.sql.shuffle.partitions", "200")  # default razonable
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ================================
    # 1) VERIFICACIÓN DE ENTRADAS
    # ================================
    ensure_exists(args.spans_parquet, "spans_parquet")
    ensure_exists(args.sentiment_parquet, "sentiment_parquet")
    if args.map_parquet:
        ensure_exists(args.map_parquet, "map_parquet")
    if args.ontology_path:
        # No usar en tu caso, pero lo dejamos por compatibilidad
        ensure_exists(args.ontology_path, "ontology_path")

    spans = spark.read.parquet(args.spans_parquet)
    sent  = spark.read.parquet(args.sentiment_parquet)

    # dumps de schema y muestras
    write_text(os.path.join(logs_dir, "spans_schema.txt"), spans._jdf.schema().treeString())
    write_text(os.path.join(logs_dir, "sent_schema.txt"),  sent._jdf.schema().treeString())

    spans_sample = spans.limit(20).toPandas()
    sent_sample  = sent.limit(20).toPandas()
    write_text(os.path.join(logs_dir, "spans_sample.txt"), spans_sample.to_string(index=False))
    write_text(os.path.join(logs_dir, "sent_sample.txt"),  sent_sample.to_string(index=False))

    # Validación con autodetección suave (por si el usuario pasa nombres no coincidentes)
    def pick(colnames, cands):
        for c in cands:
            if c and c in colnames:
                return c
        return None

    sp_cols = set(spans.columns)
    # Candidatos razonables por si cambian nombres
    id_candidates   = [args.id_col, "review_id", "doc_id", "id", "id_review"]
    span_candidates = [args.span_text_col, "span_text", "aspect_span", "aspect", "span", "term", "text"]

    id_col_det   = pick(sp_cols, id_candidates)
    span_col_det = pick(sp_cols, span_candidates)

    if not id_col_det or not span_col_det:
        raise ValueError(
            f"No encuentro columnas de ID/span en spans_parquet.\n"
            f"Columns: {sorted(spans.columns)}\n"
            f"IDs probados: {id_candidates}\n"
            f"SPANs probados: {span_candidates}"
        )
    # Actualizamos args a lo detectado (en tu caso: review_id / aspect_span)
    args.id_col = id_col_det
    args.span_text_col = span_col_det

    # limpieza mínima de spans
    spans = normalize_text_col(spans, args.span_text_col)
    spans = spans.filter(F.col(args.span_text_col).isNotNull() & (F.length(F.col(args.span_text_col)) > 0))

    # ================================
    # 2) MAPEO SPANS -> CATEGORÍAS
    # ================================
    mapped_df = spans

    if args.map_parquet:
        map_df = spark.read.parquet(args.map_parquet)
        # autodetectar columna de span en el map
        map_span_col = None
        for cand in [args.span_text_col, "span_text", "aspect_span", "aspect", "span", "term"]:
            if cand in map_df.columns:
                map_span_col = cand
                break
        if not map_span_col:
            raise ValueError("map_parquet no tiene una columna de aspecto detectable (busqué span_text/aspect_span/aspect/span/term).")

        # detectar columna de categoría
        cat_col_map = None
        for cand in [args.category_col, "aspect_norm", "aspect_category", "category"]:
            if cand in map_df.columns:
                cat_col_map = cand
                break
        if not cat_col_map:
            raise ValueError("map_parquet no tiene columna de categoría detectable (busqué aspect_norm/aspect_category/category).")

        map_df = normalize_text_col(map_df, map_span_col).select(
            F.col(map_span_col).alias("map_span_norm"),
            F.col(cat_col_map).alias("map_category")
        ).dropDuplicates(["map_span_norm"])

        mapped_df = mapped_df.join(
            map_df, on=F.col(args.span_text_col) == F.col("map_span_norm"), how="left"
        ).drop("map_span_norm")

        # Consolidar nombre de categoría final
        mapped_df = mapped_df.withColumn(
            args.category_col,
            coalesce_first_nonnull(F.col("map_category"))
        ).drop("map_category")

    # ================================
    # 3) UNIÓN CON SENTIMIENTO
    # ================================
    if args.id_col not in sent.columns:
        raise ValueError(f"sentiment_parquet debe incluir '{args.id_col}'.")

    # Si existe la columna categórica indicada, úsala
    if args.sentiment_col and args.sentiment_col in sent.columns:
        sent_use = sent.select(args.id_col, F.col(args.sentiment_col).alias(args.sentiment_col))
    else:
        # Si no, intenta con numeric -> categórico
        if not args.numeric_sentiment_col or args.numeric_sentiment_col not in sent.columns:
            raise ValueError(
                f"No existe '{args.sentiment_col}' en sentimiento y tampoco --numeric_sentiment_col válido."
            )
        neg_thr = float(args.neg_threshold)
        pos_thr = float(args.pos_threshold)
        sent_use = (
            sent
            .withColumn(
                args.sentiment_col,
                F.when(F.col(args.numeric_sentiment_col) <= neg_thr, "negative")
                 .when(F.col(args.numeric_sentiment_col) >= pos_thr, "positive")
                 .otherwise("neutral")
            )
            .select(args.id_col, args.sentiment_col)
        )

    absa = mapped_df.join(sent_use, on=args.id_col, how="left")

    # ================================
    # 4) ESCRITURA DATASET FINAL
    # ================================
    final_cols = [c for c in [args.id_col, args.span_text_col, args.category_col, args.sentiment_col] if c in absa.columns]
    if args.text_col and args.text_col in absa.columns:
        final_cols.append(args.text_col)

    absa_final = absa.select(*final_cols).dropDuplicates(final_cols)

    # Log particiones antes de escribir
    try:
        nparts = absa_final.rdd.getNumPartitions()
        print(f"[info] partitions before write = {nparts}")
        print("[info] spark.sql.shuffle.partitions =", spark.conf.get("spark.sql.shuffle.partitions"))
    except Exception as e:
        print("[warn] No pude obtener número de particiones:", e)

    if args.repartition and args.repartition > 0:
        print(f"[info] applying repartition({args.repartition})")
        absa_final = absa_final.repartition(args.repartition)

    out_parquet = os.path.join(args.output_dir, "absa_final.parquet")
    absa_final.write.mode("overwrite").parquet(out_parquet)

    # ================================
    # 5) EDA + VISUALIZACIONES
    # ================================
    # 5.1 CSVs
    if args.category_col in absa_final.columns:
        (absa_final.groupBy(args.category_col).agg(F.count("*").alias("n_aspects"))
         .orderBy(F.desc("n_aspects"))
         .coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(eda_dir, "count_by_category")))

    if args.sentiment_col in absa_final.columns:
        (absa_final.groupBy(args.sentiment_col).agg(F.count("*").alias("n_aspects"))
         .orderBy(F.desc("n_aspects"))
         .coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(eda_dir, "count_by_sentiment")))

    if (args.category_col in absa_final.columns) and (args.sentiment_col in absa_final.columns):
        (absa_final.groupBy(args.category_col, args.sentiment_col).agg(F.count("*").alias("n"))
         .orderBy(F.desc("n"))
         .coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(eda_dir, "count_by_category_sentiment")))

    if args.span_text_col in absa_final.columns:
        (absa_final.groupBy(args.span_text_col).agg(F.count("*").alias("freq"))
         .orderBy(F.desc("freq"))
         .coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(eda_dir, "top_aspects")))

    # Cobertura
    total_spans = absa_final.count()
    with_cat = absa_final.filter(F.col(args.category_col).isNotNull()).count() if args.category_col in absa_final.columns else 0
    with_sent = absa_final.filter(F.col(args.sentiment_col).isNotNull()).count() if args.sentiment_col in absa_final.columns else 0

    coverage_rows = [
        ("total_spans", float(total_spans)),
        ("with_category", float(with_cat)),
        ("with_sentiment", float(with_sent)),
        ("pct_category", round((with_cat / total_spans) * 100.0, 2) if total_spans else 0.0),
        ("pct_sentiment", round((with_sent / total_spans) * 100.0, 2) if total_spans else 0.0),
    ]
    coverage_schema = T.StructType([
        T.StructField("metric", T.StringType(), False),
        T.StructField("value", T.DoubleType(), False)
    ])
    spark.createDataFrame(coverage_rows, schema=coverage_schema)\
        .coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(eda_dir, "coverage"))

    # 5.2 PNGs (barras)
    # count_by_category (filtramos Null para el chart y limitamos top-K)
    if args.category_col in absa_final.columns:
        df_cat = (
            absa_final
            .filter(F.col(args.category_col).isNotNull())
            .groupBy(args.category_col).agg(F.count("*").alias("n_aspects"))
            .orderBy(F.desc("n_aspects"))
        ).toPandas()
        if not df_cat.empty:
            if args.top_k_categories_chart and args.top_k_categories_chart > 0:
                df_cat = df_cat.head(args.top_k_categories_chart)
            save_chart_bar(df_cat, args.category_col, "n_aspects",
                           "Aspectos por categoría",
                           os.path.join(charts_dir, "count_by_category.png"),
                           rotate_xticks=True)

    # count_by_sentiment
    if args.sentiment_col in absa_final.columns:
        df_sent = (
            absa_final.groupBy(args.sentiment_col).agg(F.count("*").alias("n_aspects"))
            .orderBy(F.desc("n_aspects"))
        ).toPandas()
        if not df_sent.empty:
            save_chart_bar(df_sent, args.sentiment_col, "n_aspects",
                           "Aspectos por sentimiento",
                           os.path.join(charts_dir, "count_by_sentiment.png"))

    # count_by_category_sentiment (stacked bar)
    if (args.category_col in absa_final.columns) and (args.sentiment_col in absa_final.columns):
        df_cat_sent = (
            absa_final.filter(F.col(args.category_col).isNotNull())
            .groupBy(args.category_col, args.sentiment_col).agg(F.count("*").alias("n"))
            .orderBy(F.desc("n"))
        ).toPandas()
        if not df_cat_sent.empty:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            pivot = df_cat_sent.pivot(index=args.category_col, columns=args.sentiment_col, values="n").fillna(0)
            if args.top_k_categories_chart and args.top_k_categories_chart > 0:
                # ordenar por suma y quedarnos con top-K
                pivot = pivot.assign(_sum=pivot.sum(axis=1)).sort_values("_sum", ascending=False).drop(columns=["_sum"]).head(args.top_k_categories_chart)
            ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
            ax.set_title("Aspectos por categoría y sentimiento")
            ax.set_xlabel(args.category_col)
            ax.set_ylabel("n")
            plt.tight_layout()
            os.makedirs(charts_dir, exist_ok=True)
            plt.savefig(os.path.join(charts_dir, "count_by_category_sentiment.png"), dpi=120)
            plt.close()

    # breve README de salida
    readme = f"""# ABSA Output

- Final parquet: `{out_parquet}`
- Logs: `{logs_dir}` (schemas y muestras)
- EDA CSV: `{eda_dir}`
- Charts: `{charts_dir}`

Cobertura:
- total_spans = {total_spans}
- with_category = {with_cat} ({(with_cat/total_spans*100.0 if total_spans else 0):.2f}%)
- with_sentiment = {with_sent} ({(with_sent/total_spans*100.0 if total_spans else 0):.2f}%)
"""
    write_text(os.path.join(args.output_dir, "README_ABSA.txt"), readme)

    print(f"[OK] ABSA final  -> {out_parquet}")
    print(f"[OK] Logs        -> {logs_dir}")
    print(f"[OK] EDA CSV     -> {eda_dir}")
    print(f"[OK] Charts PNG  -> {charts_dir}")

    spark.stop()


if __name__ == "__main__":
    main()

