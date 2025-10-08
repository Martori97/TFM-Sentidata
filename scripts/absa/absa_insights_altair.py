#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABSA Insights + Altair charts (HTML) leyendo rutas de absa_paths / absa_outputs en params.yaml.

Salidas (en absa_outputs.insights):
  charts/neg_top20_aspects.html
  charts/neg_top20_terms.html
  charts/global_influence.html
  charts/per_category_top10_positive.html
  charts/per_category_top10_negative.html
  charts/category_opportunity_matrix.html
  charts/brand_benchmarks_by_category.html
  charts/neg_aspect_cooccurrence_top50.html
  (y CSVs si insights.save_csv = true)

Requisitos:
  pip install duckdb altair pyyaml pandas
  (opcional para PNG/SVG) pip install vl-convert-python
"""

import os
import argparse
import yaml
import duckdb
import pandas as pd
import altair as alt


# ---------------------------
# Utilidades de gráficos
# ---------------------------
def save_chart(chart: alt.Chart, path_html: str):
    os.makedirs(os.path.dirname(path_html), exist_ok=True)
    chart.save(path_html)


def df_query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def chart_barh(df: pd.DataFrame, x: str, y: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, type="quantitative"),
            y=alt.Y(y, type="nominal", sort="-x"),
            tooltip=list(df.columns),
        )
        .properties(title=title, width=800, height=500)
    )


def chart_bar(df: pd.DataFrame, x: str, y: str, title: str, facet_row: str | None = None) -> alt.Chart:
    ch = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, type="nominal", sort="-y"),
            y=alt.Y(y, type="quantitative"),
            tooltip=list(df.columns),
        )
        .properties(title=title, width=700)
    )
    if facet_row:
        ch = ch.facet(row=alt.Row(facet_row, type="nominal"))
    return ch


def chart_scatter(df: pd.DataFrame, x: str, y: str, size: str, color: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X(x, type="quantitative"),
            y=alt.Y(y, type="quantitative"),
            size=alt.Size(size, type="quantitative"),
            color=alt.Color(color, type="quantitative"),
            tooltip=list(df.columns),
        )
        .properties(title=title, width=900, height=520)
    )


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--spans")
    ap.add_argument("--mapped")
    ap.add_argument("--sentiment")
    ap.add_argument("--product_snapshot")
    ap.add_argument("--outdir")
    ap.add_argument("--save_csv", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.params, "r", encoding="utf-8"))

    p_all = cfg.get("absa_paths", cfg["paths"])
    o_all = cfg.get("absa_outputs", cfg.get("outputs", {}))

    spans = args.spans or p_all["spans"]
    mapped = args.mapped or p_all["mapped"]
    sentiment = args.sentiment or p_all["sentiment"]
    snapshot = args.product_snapshot or p_all["product_snapshot"]
    outdir = args.outdir or o_all["insights"]
    save_csv = args.save_csv or bool(cfg.get("insights", {}).get("save_csv", False))

    charts_dir = os.path.join(outdir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    con = duckdb.connect(database=":memory:")

    # Vista unificada ABSA con normalización de sentimiento a 'pred3_norm'
    con.execute(f"""
    CREATE OR REPLACE VIEW absa AS
    WITH spans AS (
      SELECT review_id, aspect_span AS span_text
      FROM '{spans}'
    ),
    mapped AS (
      SELECT review_id, span_text, aspect_norm
      FROM '{mapped}'
    ),
    sent AS (
      SELECT
        review_id,
        CASE
          WHEN lower(CAST(pred_3 AS VARCHAR)) IN ('neg','negative') THEN 'neg'
          WHEN lower(CAST(pred_3 AS VARCHAR)) IN ('neu','neutral')  THEN 'neu'
          WHEN lower(CAST(pred_3 AS VARCHAR)) IN ('pos','positive') THEN 'pos'
          WHEN TRY_CAST(pred_3 AS INT) = 0 THEN 'neg'
          WHEN TRY_CAST(pred_3 AS INT) = 1 THEN 'neu'
          WHEN TRY_CAST(pred_3 AS INT) = 2 THEN 'pos'
          ELSE NULL
        END AS pred3_norm
      FROM '{sentiment}'
    ),
    prod AS (
      SELECT review_id, product_id,
             product_name2 AS product_name,
             brand_name2   AS brand_name,
             category2,
             pinfo_loves_count, pinfo_rating, pinfo_reviews, pinfo_price_usd, rating
      FROM '{snapshot}'
    )
    SELECT
      s.review_id,
      lower(m.span_text)   AS span_text,
      lower(m.aspect_norm) AS aspect_norm,
      sent.pred3_norm,
      p.product_id, p.product_name, p.brand_name, p.category2,
      p.pinfo_loves_count, p.pinfo_rating, p.pinfo_reviews, p.pinfo_price_usd, p.rating
    FROM spans s
    JOIN mapped m USING(review_id, span_text)
    LEFT JOIN sent  ON s.review_id = sent.review_id
    LEFT JOIN prod  p ON s.review_id = p.review_id
    WHERE m.aspect_norm IS NOT NULL AND sent.pred3_norm IS NOT NULL;
    """)

    # 1) Top-20 aspectos en negativo (global)
    neg_aspects = df_query(con, """
      SELECT
        aspect_norm,
        COUNT(*) AS neg_mentions,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS neg_share_pct
      FROM absa
      WHERE pred3_norm='neg'
      GROUP BY aspect_norm
      ORDER BY neg_mentions DESC
      LIMIT 20;
    """)
    save_chart(
        chart_barh(neg_aspects, "neg_mentions:Q", "aspect_norm:N", "Top-20 aspectos mencionados en negativo (global)"),
        os.path.join(charts_dir, "neg_top20_aspects.html"),
    )

    # 2) Top-20 términos negativos (literal)
    neg_terms = df_query(con, """
      SELECT
        lower(span_text) AS term,
        COUNT(*) AS neg_mentions
      FROM absa
      WHERE pred3_norm='neg'
      GROUP BY lower(span_text)
      ORDER BY neg_mentions DESC
      LIMIT 20;
    """)
    save_chart(
        chart_barh(neg_terms, "neg_mentions:Q", "term:N", "Top-20 términos negativos (literal)"),
        os.path.join(charts_dir, "neg_top20_terms.html"),
    )

    # 3) Influence score (global) por aspecto
    influence = df_query(con, """
    WITH agg AS (
      SELECT aspect_norm, pred3_norm, COUNT(*) AS n
      FROM absa
      GROUP BY aspect_norm, pred3_norm
    ),
    pv AS (
      SELECT
        aspect_norm,
        COALESCE(SUM(CASE WHEN pred3_norm='neg' THEN n END),0) AS neg,
        COALESCE(SUM(CASE WHEN pred3_norm='neu' THEN n END),0) AS neu,
        COALESCE(SUM(CASE WHEN pred3_norm='pos' THEN n END),0) AS pos
      FROM agg
      GROUP BY aspect_norm
    )
    SELECT
      aspect_norm,
      neg, neu, pos,
      (neg+neu+pos) AS volume_total,
      CASE WHEN (neg+neu+pos)>0 THEN pos*1.0/(neg+neu+pos) ELSE 0 END AS pos_rate,
      CASE WHEN (neg+neu+pos)>0 THEN neg*1.0/(neg+neu+pos) ELSE 0 END AS neg_rate,
      ((CASE WHEN (neg+neu+pos)>0 THEN pos*1.0/(neg+neu+pos) ELSE 0 END) -
       (CASE WHEN (neg+neu+pos)>0 THEN neg*1.0/(neg+neu+pos) ELSE 0 END)
      ) * ln(1 + (neg+neu+pos)) AS impact_score
    FROM pv
    ORDER BY impact_score DESC
    LIMIT 50;
    """)
    save_chart(
        chart_barh(influence, "impact_score:Q", "aspect_norm:N", "Influence score (global) — Top 50"),
        os.path.join(charts_dir, "global_influence.html"),
    )

    # 4) Top-10 positivos/negativos por categoría (impact score)
    percat = df_query(con, """
    WITH agg AS (
      SELECT category2, aspect_norm, pred3_norm, COUNT(*) AS n
      FROM absa
      WHERE category2 IS NOT NULL
      GROUP BY category2, aspect_norm, pred3_norm
    ),
    pv AS (
      SELECT
        category2, aspect_norm,
        COALESCE(SUM(CASE WHEN pred3_norm='neg' THEN n END),0) AS neg,
        COALESCE(SUM(CASE WHEN pred3_norm='neu' THEN n END),0) AS neu,
        COALESCE(SUM(CASE WHEN pred3_norm='pos' THEN n END),0) AS pos
      FROM agg
      GROUP BY category2, aspect_norm
    ),
    scored AS (
      SELECT
        category2, aspect_norm, neg, neu, pos,
        (neg+neu+pos) AS volume_total,
        CASE WHEN (neg+neu+pos)>0 THEN pos*1.0/(neg+neu+pos) ELSE 0 END AS pos_rate,
        CASE WHEN (neg+neu+pos)>0 THEN neg*1.0/(neg+neu+pos) ELSE 0 END AS neg_rate,
        ((CASE WHEN (neg+neu+pos)>0 THEN pos*1.0/(neg+neu+pos) ELSE 0 END) -
         (CASE WHEN (neg+neu+pos)>0 THEN neg*1.0/(neg+neu+pos) ELSE 0 END)
        ) * ln(1 + (neg+neu+pos)) AS impact_score
      FROM pv
    )
    SELECT * FROM scored;
    """)

    percat_pos = (
        percat.sort_values(["category2", "impact_score"], ascending=[True, False])
        .groupby("category2", as_index=False)
        .head(10)
    )
    save_chart(
        chart_bar(percat_pos, "aspect_norm:N", "impact_score:Q", "Top-10 aspectos por categoría (positivos)", facet_row="category2:N"),
        os.path.join(charts_dir, "per_category_top10_positive.html"),
    )

    percat_neg = (
        percat.sort_values(["category2", "impact_score", "neg"], ascending=[True, True, False])
        .groupby("category2", as_index=False)
        .head(10)
    )
    save_chart(
        chart_bar(percat_neg, "aspect_norm:N", "impact_score:Q", "Top-10 aspectos por categoría (negativos)", facet_row="category2:N"),
        os.path.join(charts_dir, "per_category_top10_negative.html"),
    )

    # 5) Matriz de oportunidades por categoría (volumen vs pos/neg rate)
    opp = df_query(con, """
    WITH agg AS (
      SELECT category2, pred3_norm, COUNT(*) AS n
      FROM absa
      WHERE category2 IS NOT NULL
      GROUP BY category2, pred3_norm
    ),
    pv AS (
      SELECT
        category2,
        COALESCE(SUM(CASE WHEN pred3_norm='neg' THEN n END),0) AS neg,
        COALESCE(SUM(CASE WHEN pred3_norm='neu' THEN n END),0) AS neu,
        COALESCE(SUM(CASE WHEN pred3_norm='pos' THEN n END),0) AS pos
      FROM agg
      GROUP BY category2
    )
    SELECT
      category2,
      neg, neu, pos,
      (neg+neu+pos) AS volume_total,
      CASE WHEN (neg+neu+pos)>0 THEN pos*1.0/(neg+neu+pos) ELSE 0 END AS pos_rate,
      CASE WHEN (neg+neu+pos)>0 THEN neg*1.0/(neg+neu+pos) ELSE 0 END AS neg_rate
    FROM pv
    ORDER BY volume_total DESC;
    """)
    save_chart(
        chart_scatter(opp, "pos_rate:Q", "neg_rate:Q", "volume_total:Q", "neg_rate:Q", "Matriz de oportunidades por categoría"),
        os.path.join(charts_dir, "category_opportunity_matrix.html"),
    )

    # 6) Benchmarks de marca por categoría (n ≥ 100)
    brand_bench = df_query(con, """
    WITH agg AS (
      SELECT category2, brand_name, pred3_norm, COUNT(*) AS n
      FROM absa
      WHERE category2 IS NOT NULL AND brand_name IS NOT NULL
      GROUP BY category2, brand_name, pred3_norm
    ),
    pv AS (
      SELECT
        category2, brand_name,
        COALESCE(SUM(CASE WHEN pred3_norm='neg' THEN n END),0) AS neg,
        COALESCE(SUM(CASE WHEN pred3_norm='neu' THEN n END),0) AS neu,
        COALESCE(SUM(CASE WHEN pred3_norm='pos' THEN n END),0) AS pos
      FROM agg
      GROUP BY category2, brand_name
    )
    SELECT
      category2, brand_name, neg, neu, pos,
      (neg+neu+pos) AS volume_total
    FROM pv
    WHERE (neg+neu+pos) >= 100
    ORDER BY category2, pos DESC, volume_total DESC;
    """)
    brand_long = brand_bench.melt(
        id_vars=["category2", "brand_name", "volume_total"],
        value_vars=["pos", "neu", "neg"],
        var_name="sentiment",
        value_name="n",
    )
    save_chart(
        alt.Chart(brand_long)
        .mark_bar()
        .encode(
            x=alt.X("brand_name:N", sort="-y"),
            y=alt.Y("n:Q", stack="normalize", title="Share de menciones"),
            color=alt.Color("sentiment:N"),
            tooltip=list(brand_long.columns),
        )
        .properties(title="Benchmark de marcas por categoría (n≥100)")
        .facet(row="category2:N"),
        os.path.join(charts_dir, "brand_benchmarks_by_category.html"),
    )

    # 7) Co-ocurrencias de aspectos negativos (Top-50) — construir 'pair' en SQL
    cooc = df_query(con, """
    WITH neg_reviews AS (
      SELECT review_id, LIST(DISTINCT aspect_norm) AS aspects
      FROM absa
      WHERE pred3_norm='neg' AND aspect_norm IS NOT NULL
      GROUP BY review_id
    ),
    pairs AS (
      SELECT
        CAST(a1 AS VARCHAR) AS aspect_a,
        CAST(a2 AS VARCHAR) AS aspect_b,
        CAST(a1 AS VARCHAR) || ' + ' || CAST(a2 AS VARCHAR) AS pair
      FROM neg_reviews, UNNEST(aspects) a1, UNNEST(aspects) a2
      WHERE a1 < a2
    )
    SELECT aspect_a, aspect_b, pair, COUNT(*) AS co_neg_count
    FROM pairs
    GROUP BY aspect_a, aspect_b, pair
    ORDER BY co_neg_count DESC
    LIMIT 50;
    """)
    save_chart(
        chart_barh(cooc, "co_neg_count:Q", "pair:N", "Co-ocurrencias de aspectos negativos (Top-50)"),
        os.path.join(charts_dir, "neg_aspect_cooccurrence_top50.html"),
    )

    # CSV opcionales
    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        neg_aspects.to_csv(os.path.join(outdir, "neg_top20_aspects.csv"), index=False)
        neg_terms.to_csv(os.path.join(outdir, "neg_top20_terms.csv"), index=False)
        influence.to_csv(os.path.join(outdir, "global_top50_influence.csv"), index=False)
        percat_pos.to_csv(os.path.join(outdir, "per_category_top10_positive.csv"), index=False)
        percat_neg.to_csv(os.path.join(outdir, "per_category_top10_negative.csv"), index=False)
        opp.to_csv(os.path.join(outdir, "category_opportunity_matrix.csv"), index=False)
        brand_bench.to_csv(os.path.join(outdir, "brand_benchmarks_by_category.csv"), index=False)
        cooc.to_csv(os.path.join(outdir, "neg_aspect_cooccurrence_top50.csv"), index=False)

    print(f"[ok] gráficos en {charts_dir}")


if __name__ == "__main__":
    main()

