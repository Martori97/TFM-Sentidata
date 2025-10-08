#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Box plot por brand×aspect desde preview_fused.parquet.

Comportamiento de color:
- Si --category_filter es una categoría concreta (secondary) -> color por secondary category (leyenda).
- Si --category_filter es __ALL__/All (todo Skincare) -> NO segmentar por secondary; un solo color (sin leyenda).

Notas:
- Se asume que el dataset ya es de PRIMARY = Skincare (como en tu pipeline). Si no fuera así,
  filtra antes o añade un filtro de primary en build_preview_fused/aspect_opportunity.
"""

import os
import argparse
import numpy as np
import pandas as pd
import altair as alt

def is_all(v: str) -> bool:
    return str(v).strip().lower() in {"", "none", "null", "__all__", "all"}

def main():
    ap = argparse.ArgumentParser("Box plot desde preview_fused.parquet (scroll; color opcional por secondary)")
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_html", default="reports/viz/box_by_aspect_brand.html")
    ap.add_argument("--metric", choices=["rating","pred3","pos_minus_neg"], default="rating")
    ap.add_argument("--category_filter", default="__ALL__",
                    help="Secondary category concreta o __ALL__ para todo Skincare sin segmentar")
    ap.add_argument("--min_n_per_cell", type=int, default=50)
    ap.add_argument("--max_aspects", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=1_000_000)
    ap.add_argument("--pixels_per_brand", type=int, default=18, help="ancho por marca para scroll")
    ap.add_argument("--panel_height", type=int, default=300)
    ap.add_argument("--color_when_all", default="#4C78A8", help="color único cuando category_filter=All")
    # compat: aceptar --panel_width aunque no se usa (ancho dinámico)
    ap.add_argument("--panel_width", type=int, default=0)
    args = ap.parse_args()

    alt.data_transformers.disable_max_rows()
    df = pd.read_parquet(args.input_parquet)[:args.max_rows].copy()

    # filtro por secondary category si NO es All
    if not is_all(args.category_filter):
        df = df[df["category"] == args.category_filter]

    # métrica a visualizar
    if args.metric == "rating":
        df["metric_value"] = df["rating"].astype(float)
        y_title = "Rating (1–5)"
    elif args.metric == "pred3":
        m = df["pred_3"].astype(float)
        df["metric_value"] = (m - 1) if m.min() >= 0 else m  # 0,1,2 → -1,0,1
        y_title = "Sentiment label (−1,0,1)"
    else:  # pos_minus_neg
        df["metric_value"] = (df["p_pos"].astype(float) - df["p_neg"].astype(float)).clip(-1,1)
        y_title = "Pos − Neg (≈ −1..1)"

    # solidez: al menos N reviews únicas por brand×aspect
    df = df.dropna(subset=["brand","aspect_norm","metric_value"])
    cnt = (df.groupby(["brand","aspect_norm"])["review_id"].nunique()
             .reset_index(name="n_rev"))
    df = df.merge(cnt, on=["brand","aspect_norm"], how="left")
    df = df[df["n_rev"] >= args.min_n_per_cell]

    # top aspectos por volumen (#reviews)
    top_aspects = (df.groupby("aspect_norm")["review_id"].nunique()
                     .sort_values(ascending=False).head(args.max_aspects).index)
    df = df[df["aspect_norm"].isin(top_aspects)]

    # ancho dinámico para scroll
    n_brands = max(1, df["brand"].nunique())
    panel_w = max(800, n_brands * args.pixels_per_brand)

    # encoding de color: por secondary SOLO si hay categoría concreta; si es All -> color fijo
    if is_all(args.category_filter):
        color_enc = alt.value(args.color_when_all)  # sin leyenda
    else:
        color_enc = alt.Color("category:N", legend=alt.Legend(title="Secondary category"))

    base = alt.Chart(df).transform_filter(alt.datum.metric_value != None)

    box = (base.mark_boxplot(size=12, extent="min-max")
           .encode(
               x=alt.X("brand:N", sort="-y", title=None),
               y=alt.Y("metric_value:Q", title=y_title),
               color=color_enc,
               tooltip=[
                   "brand:N",
                   "category:N",
                   "aspect_norm:N",
                   alt.Tooltip("metric_value:Q", title="value", format=".3f"),
                   alt.Tooltip("n_rev:Q", title="#reviews")
               ],
           )
           .properties(width=panel_w, height=args.panel_height))

    chart = box.facet(row=alt.Row("aspect_norm:N", header=alt.Header(title="Aspect")))
    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    chart.save(args.output_html)
    print("[OK] boxplot:", args.output_html,
          "| mode:", "ALL (color único)" if is_all(args.category_filter) else f"Secondary={args.category_filter}")

if __name__ == "__main__":
    main()

