#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import altair as alt

def is_all(v: str) -> bool:
    return str(v).strip().lower() in {"", "none", "null", "__all__", "all"}

def parse_bandwidth(bw):
    if bw is None:
        return None
    s = str(bw).strip().lower()
    if s in {"", "none", "null", "auto"}:
        return None
    return float(s)

def parse_bool(v) -> bool:
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def smooth_gaussian(arr: np.ndarray, sigma_bins: float | None) -> np.ndarray:
    if sigma_bins is None or sigma_bins <= 0:
        return arr
    radius = int(max(1, round(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kern = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kern /= kern.sum()
    return np.convolve(arr, kern, mode="same")

def compute_violin_data(df: pd.DataFrame, value_col: str, category_col: str,
                        extent: tuple[float,float], bins: int = 200,
                        sigma_bins: float | None = None) -> pd.DataFrame:
    lo, hi = extent
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    rows = []
    for cat, sub in df.groupby(category_col):
        vals = sub[value_col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        hist, _ = np.histogram(vals, bins=edges, density=True)
        hist = smooth_gaussian(hist, sigma_bins)
        bin_w = (hi - lo) / bins
        area = hist.sum() * bin_w
        if area > 0:
            hist = hist / area
        rows.append(pd.DataFrame({"category": cat, "value": centers, "density": hist.astype(float)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["category","value","density"])

def main():
    ap = argparse.ArgumentParser("Violin de rating por secondary category (pre-aggregado)")
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_html", default="reports/viz/violin_secondary_rating.html")
    ap.add_argument("--metric", choices=["rating","pos_minus_neg"], default="rating")
    ap.add_argument("--category_filter", default="All")
    ap.add_argument("--min_reviews_per_cat", type=int, default=200)
    ap.add_argument("--max_categories", type=int, default=40)
    ap.add_argument("--per_cat_width", type=int, default=120)
    ap.add_argument("--row_height", type=int, default=300)
    ap.add_argument("--bandwidth", default="auto")  # sigma de suavizado en bins; 'auto' = None
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--dedup_reviews", default="true", help="true/false: 1 review = 1 voto (por defecto true)")
    args = ap.parse_args()

    sigma = parse_bandwidth(args.bandwidth)
    do_dedup = parse_bool(args.dedup_reviews)

    df = pd.read_parquet(args.input_parquet)
    for c in ["review_id","category","rating","p_pos","p_neg"]:
        if c not in df.columns: df[c] = np.nan
    df["category"] = df["category"].fillna("Unknown")

    # métrica
    if args.metric == "rating":
        df["value"] = df["rating"].astype(float).clip(1, 5)
        y_title, extent = "Rating (1–5)", (1.0, 5.0)
    else:
        df["value"] = (df["p_pos"].astype(float) - df["p_neg"].astype(float)).clip(0, 1)
        y_title, extent = "Pos − Neg (0–1)", (0.0, 1.0)

    # filtro (All / secundaria concreta)
    if not is_all(args.category_filter):
        df = df[df["category"] == args.category_filter]

    # deduplicación: 1 review = 1 voto
    if do_dedup:
        df = df.sort_values("review_id").drop_duplicates(subset=["review_id"])

    # solidez por categoría
    vc = df.groupby("category")["review_id"].nunique().sort_values(ascending=False)
    cats = vc[vc >= args.min_reviews_per_cat].head(args.max_categories).index.tolist()
    df = df[df["category"].isin(cats)]
    if df.empty:
        raise SystemExit("Sin datos tras filtros. Ajusta --min_reviews_per_cat/--max_categories.")

    den = compute_violin_data(df[["category","value"]], "value", "category",
                              extent=extent, bins=args.bins, sigma_bins=sigma)
    if den.empty:
        raise SystemExit("No se pudo calcular densidades.")

    base = alt.Chart(den)
    spec = (
        base.mark_area(orient="horizontal", opacity=0.75)
            .encode(
                y=alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=list(extent))),
                x=alt.X("density:Q", stack="center", title=None, axis=None),
                color=alt.Color("category:N", legend=alt.Legend(title="Secondary category")),
                tooltip=["category:N", alt.Tooltip("value:Q", format=".2f"),
                         alt.Tooltip("density:Q", format=".3f")],
            )
            .properties(width=args.per_cat_width, height=args.row_height)
    )
    chart = spec.facet(column=alt.Column("category:N", header=alt.Header(title=None))) \
                .resolve_scale(y="shared", x="independent", color="independent")

    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    chart.save(args.output_html)
    print(f"[OK] violin: {args.output_html} | cats={len(cats)} | bins={args.bins} "
          f"| smooth_sigma={sigma if sigma is not None else 'auto'} | dedup={do_dedup}")

if __name__ == "__main__":
    main()

