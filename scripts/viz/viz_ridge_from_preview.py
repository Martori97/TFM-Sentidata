#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import altair as alt

def main():
    ap = argparse.ArgumentParser("Ridge plot (density) por brand×aspect desde preview_fused.parquet")
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_html", default="reports/viz/ridge_by_aspect_brand.html")
    ap.add_argument("--metric", choices=["rating","pos_minus_neg"], default="pos_minus_neg")
    ap.add_argument("--category_filter", default="__ALL__")
    ap.add_argument("--min_n_per_cell", type=int, default=50)
    ap.add_argument("--max_aspects", type=int, default=8)
    ap.add_argument("--max_brands_per_aspect", type=int, default=25)
    ap.add_argument("--extent_lo", type=float, default=0.0, help="mín. para densidad")
    ap.add_argument("--extent_hi", type=float, default=1.0, help="máx. para densidad")
    args = ap.parse_args()

    alt.data_transformers.disable_max_rows()
    df = pd.read_parquet(args.input_parquet)

    catf = None if str(args.category_filter).lower() in {"", "none", "null", "__all__", "all"} else args.category_filter
    if catf:
        df = df[df["category"] == catf]

    # métrica a densidad
    if args.metric == "rating":
        df["metric_value"] = df["rating"].astype(float).clip(1,5)
        extent = [1,5]
        x_title = "Rating (1–5)"
    else:
        df["metric_value"] = (df["p_pos"].astype(float) - df["p_neg"].astype(float)).clip(0,1)
        extent = [args.extent_lo, args.extent_hi]
        x_title = "Pos − Neg (0–1)"

    df = df.dropna(subset=["brand","aspect_norm","metric_value"])

    # solidez
    cnt = (df.groupby(["brand","aspect_norm"])["review_id"].nunique()
             .reset_index(name="n_rev"))
    df = df.merge(cnt, on=["brand","aspect_norm"], how="left")
    df = df[df["n_rev"] >= args.min_n_per_cell]

    # top aspectos por nº de reviews
    top_aspects = (df.groupby("aspect_norm")["review_id"].nunique()
                     .sort_values(ascending=False).head(args.max_aspects).index)
    df = df[df["aspect_norm"].isin(top_aspects)]

    # top brands por aspecto (para no saturar)
    df["rank_brand_in_aspect"] = (df.groupby("aspect_norm")["n_rev"]
                                    .rank(ascending=False, method="first"))
    df = df[df["rank_brand_in_aspect"] <= args.max_brands_per_aspect]

    base = alt.Chart(df)

    ridge = (
        base.transform_density(
                "metric_value",
                as_=["metric_value","density"],
                groupby=["aspect_norm","brand","category"],
                extent=extent, steps=256
            )
            .mark_area(opacity=0.6, interpolate="monotone")
            .encode(
                x=alt.X("metric_value:Q", title=x_title),
                y=alt.Y("density:Q", title=None, stack=None),
                row=alt.Row("aspect_norm:N",
                            header=alt.Header(title="Aspect", labelAngle=0, labelAlign="left")),
                color=alt.Color("category:N", legend=alt.Legend(title="Secondary category")),
                tooltip=["brand:N","category:N","aspect_norm:N",
                         alt.Tooltip("metric_value:Q", format=".2f"),
                         alt.Tooltip("density:Q", format=".3f")]
            )
            .properties(width=800, height=80)
    )

    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    ridge.save(args.output_html)
    print("[OK] ridge:", args.output_html)

if __name__ == "__main__":
    main()


