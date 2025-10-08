#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import altair as alt
import json

# ---------------- utilidades ----------------
def parse_bool(v) -> bool:
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def parse_bw(bw):
    if bw is None: return None
    s = str(bw).strip().lower()
    if s in {"","none","null","auto"}: return None
    return float(s)

def smooth_gaussian(arr: np.ndarray, sigma_units: float | None, bins: int, lo: float, hi: float) -> np.ndarray:
    if not sigma_units or sigma_units <= 0:
        return arr
    sigma_bins = sigma_units * bins / (hi - lo)
    radius = int(max(1, round(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2); k /= k.sum()
    return np.convolve(arr, k, mode="same")

def density_from_series(vals: np.ndarray, lo: float, hi: float, bins: int, sigma_units: float | None):
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist, _ = np.histogram(vals, bins=edges, density=True)
    hist = smooth_gaussian(hist, sigma_units, bins, lo, hi)
    bin_w = (hi - lo) / bins
    area = hist.sum() * bin_w
    if area > 0: hist /= area
    return centers, hist

# --------- CARGA secondary category DESDE DELTA (mismo enfoque que aspect_opportunity.py) ---------
def load_secondary_category_map(reviews_delta: str) -> pd.DataFrame:
    """Devuelve: product_id, category_secondary (desde pinfo_secondary_category o similares)."""
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession, functions as F
    builder = (
        SparkSession.builder.appName("load-secondary-cat-panel")
        .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    sdf = spark.read.format("delta").load(reviews_delta)

    col = None
    for c in ["pinfo_secondary_category", "secondary_category", "category_secondary"]:
        if c in sdf.columns:
            col = c; break
    if col is None:
        # último recurso: primaria (mejor que nada)
        for c in ["pinfo_primary_category", "category"]:
            if c in sdf.columns:
                col = c; break

    if col is None:
        spark.stop()
        raise RuntimeError("No encuentro columna de category secundaria/primaria en la Delta.")

    cat = (sdf.select("product_id",
                      F.col(col).alias("category_secondary"))
             .where(F.col("product_id").isNotNull())
             .dropna()
             .dropDuplicates(["product_id"]))
    pdf = cat.toPandas()
    spark.stop()
    print(f"[INFO] Secondary map desde Delta: columna origen='{col}', filas={len(pdf):,}")
    return pdf

def apply_secondary_category(df_preview: pd.DataFrame, reviews_delta: str) -> pd.DataFrame:
    """Sobrescribe df['category'] con la secundaria cargada de Delta (solo en memoria)."""
    sec = load_secondary_category_map(reviews_delta)
    out = df_preview.merge(sec, on="product_id", how="left")
    before = out["category"].nunique(dropna=True)
    out["category"] = out["category_secondary"].fillna(out["category"])
    out.drop(columns=["category_secondary"], inplace=True)
    after = out["category"].nunique(dropna=True)
    print(f"[INFO] category nunique: {before} -> {after} tras aplicar secondary desde Delta")
    return out

def build_market_and_brand_densities(df: pd.DataFrame, value_col: str,
                                     extent: tuple[float,float], bins: int,
                                     sigma_units: float | None,
                                     top_brands_per_cat: int):
    lo, hi = extent
    cats = sorted([c for c in df["category"].dropna().unique().tolist() if c != "Unknown"])

    # MARKET
    rows_mkt = []
    vals = df[value_col].dropna().to_numpy()
    n = df["review_id"].nunique()
    if len(vals) > 0:
        v, d = density_from_series(vals, lo, hi, bins, sigma_units)
        rows_mkt.append(pd.DataFrame({"category": "Total Skincare", "value": v, "density": d, "n_reviews": n}))
    for cat in cats:
        sub = df[df["category"] == cat]
        vals = sub[value_col].dropna().to_numpy()
        if len(vals) == 0: continue
        n = sub["review_id"].nunique()
        v, d = density_from_series(vals, lo, hi, bins, sigma_units)
        rows_mkt.append(pd.DataFrame({"category": cat, "value": v, "density": d, "n_reviews": n}))
    market_den = pd.concat(rows_mkt, ignore_index=True) if rows_mkt else pd.DataFrame()

    # BRANDS
    rows_b = []
    sub_all = df.copy()
    top_all = (sub_all.groupby("brand")["review_id"].nunique()
                     .sort_values(ascending=False).head(top_brands_per_cat).index.tolist())
    sub_all = sub_all[sub_all["brand"].isin(top_all)]
    counts_all = sub_all.groupby("brand")["review_id"].nunique()
    for br, grp in sub_all.groupby("brand"):
        vals = grp[value_col].dropna().to_numpy()
        if len(vals) == 0: continue
        v, d = density_from_series(vals, lo, hi, bins, sigma_units)
        rows_b.append(pd.DataFrame({
            "panel_cat": "Total Skincare", "brand": br, "value": v, "density": d,
            "n_reviews": counts_all.get(br, len(grp["review_id"].unique()))
        }))
    for cat in cats:
        sub = df[df["category"] == cat]
        top = (sub.groupby("brand")["review_id"].nunique()
                   .sort_values(ascending=False).head(top_brands_per_cat).index.tolist())
        sub = sub[sub["brand"].isin(top)]
        counts = sub.groupby("brand")["review_id"].nunique()
        for br, grp in sub.groupby("brand"):
            vals = grp[value_col].dropna().to_numpy()
            if len(vals) == 0: continue
            v, d = density_from_series(vals, lo, hi, bins, sigma_units)
            rows_b.append(pd.DataFrame({
                "panel_cat": cat, "brand": br, "value": v, "density": d,
                "n_reviews": counts.get(br, len(grp["review_id"].unique()))
            }))
    brand_den = pd.concat(rows_b, ignore_index=True) if rows_b else pd.DataFrame()

    options = ["Total Skincare"] + cats
    print(f"[INFO] Secondary categories detectadas: {len(cats)} -> {cats[:8]}{'...' if len(cats)>8 else ''}")
    return market_den, brand_den, options

# ---------------- HTML sticky (selector + mercado fijos a la izquierda) ----------------
STICKY_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Market vs Brands (sticky)</title>
<style>
  html, body { height: 100%; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
  .container { display: grid; grid-template-columns: 380px 1fr; gap: 16px; height: 100vh; }
  .left {
    position: sticky; top: 0; align-self: start;
    background: #fff; border-right: 1px solid #eee; padding: 12px;
    height: 100vh; display: flex; flex-direction: column;
  }
  .selector { position: sticky; top: 0; background: #fff; z-index: 10; padding: 8px 0; border-bottom: 1px solid #eee; margin-bottom: 8px; }
  .right { padding: 12px; overflow-y: auto; height: 100vh; }
  label { font-size: 14px; margin-right: 6px; }
  select { padding: 4px 8px; font-size: 14px; max-width: 100%; }
</style>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head>
<body>
<div class="container">
  <div class="left">
    <div class="selector">
      <label for="catSelect"><b>Secondary category:</b></label>
      <select id="catSelect"></select>
    </div>
    <div id="market"></div>
  </div>
  <div class="right">
    <div id="brands"></div>
  </div>
</div>
<script>
  const marketSpec = __MARKET_SPEC__;
  const brandsSpec = __BRANDS_SPEC__;
  const options = __OPTIONS__;
  const initValue = options[0] || "Total Skincare";

  const select = document.getElementById("catSelect");
  options.forEach(o => { const opt = document.createElement("option"); opt.value = o; opt.textContent = o; select.appendChild(opt); });
  select.value = initValue;

  function filterSpecs(cat) {
    const m = JSON.parse(JSON.stringify(marketSpec)); m.transform = [{filter: `datum.category == '${cat.replace(/'/g,"\\'")}'`}];
    const b = JSON.parse(JSON.stringify(brandsSpec)); b.transform = [{filter: `datum.panel_cat == '${cat.replace(/'/g,"\\'")}'`}];
    return {m,b};
  }

  async function render(cat) {
    const {m,b} = filterSpecs(cat);
    await vegaEmbed('#market', m, {actions:false});
    await vegaEmbed('#brands', b, {actions:false});
  }

  select.addEventListener('change', e => render(e.target.value));
  render(initValue);
</script>
</body>
</html>
"""

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Panel mercado vs marcas (violines, izquierda fija)")
    ap.add_argument("--input_parquet", required=True, help="reports/absa/final_all/preview_fused.parquet")
    ap.add_argument("--reviews_delta", required=True, help="data/trusted/reviews_product_info_clean_full")
    ap.add_argument("--output_html_sticky", default="reports/viz/market_brand_panel_sticky.html")
    ap.add_argument("--metric", choices=["rating","pos_minus_neg"], default="rating")
    ap.add_argument("--dedup_reviews", default="true")
    ap.add_argument("--min_reviews_per_cat", type=int, default=200)
    ap.add_argument("--max_categories", type=int, default=40)
    ap.add_argument("--bins", type=int, default=300)
    ap.add_argument("--bandwidth", default="auto")
    ap.add_argument("--jitter_sigma", type=float, default=0.12)
    ap.add_argument("--market_width", type=int, default=320)
    ap.add_argument("--market_height", type=int, default=360)
    ap.add_argument("--brand_width", type=int, default=160)
    ap.add_argument("--brand_height", type=int, default=120)
    ap.add_argument("--brand_cols", type=int, default=6)
    ap.add_argument("--top_brands_per_cat", type=int, default=60)
    args = ap.parse_args()

    sigma_units = parse_bw(args.bandwidth)
    do_dedup = parse_bool(args.dedup_reviews)

    # datos (preview)
    df = pd.read_parquet(args.input_parquet)
    for c in ["review_id","product_id","brand","category","rating","p_pos","p_neg"]:
        if c not in df.columns: df[c] = np.nan
    df["brand"] = df["brand"].fillna("Unknown")

    # secondary desde Delta (como en aspect_opportunity.py) => sobrescribe 'category'
    df = apply_secondary_category(df, args.reviews_delta)

    # métrica
    if args.metric == "rating":
        df["value"] = df["rating"].astype(float).clip(1,5)
        extent = (1.0, 5.0); y_title = "Rating (1–5)"
        if args.jitter_sigma and args.jitter_sigma > 0:
            rng = np.random.default_rng(42)
            df["value"] = (df["value"] + rng.normal(0.0, args.jitter_sigma, len(df))).clip(1,5)
        if sigma_units is None: sigma_units = 0.18
    else:
        df["value"] = (df["p_pos"].astype(float) - df["p_neg"].astype(float)).clip(0,1)
        extent = (0.0, 1.0); y_title = "Pos − Neg (0–1)"
        if sigma_units is None: sigma_units = 0.04

    if do_dedup:
        df = df.sort_values("review_id").drop_duplicates(subset=["review_id"])

    # recorte categorías para solidez
    vc = df.groupby("category")["review_id"].nunique().sort_values(ascending=False)
    cats_ok = vc[vc >= args.min_reviews_per_cat].head(args.max_categories).index.tolist()
    df = df[df["category"].isin(cats_ok)]
    if df.empty:
        raise SystemExit("Sin datos tras filtros. Ajusta --min_reviews_per_cat/--max_categories.")
    print(f"[INFO] Categorías finales (min_reviews={args.min_reviews_per_cat}): {len(cats_ok)}")

    # densidades
    market_den, brand_den, options = build_market_and_brand_densities(
        df, "value", extent, args.bins, sigma_units, args.top_brands_per_cat
    )

    # Altair sin límite de filas
    alt.data_transformers.disable_max_rows()

    # specs
    mkt_spec = (alt.Chart(market_den)
                  .mark_area(orient="horizontal", opacity=0.9)
                  .encode(
                      y=alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=list(extent))),
                      x=alt.X("density:Q", stack="center", title="Density",
                              axis=alt.Axis(labels=False,ticks=False)),
                      color=alt.Color("category:N", legend=None),
                      tooltip=[ "category:N", alt.Tooltip("value:Q", format=".2f"),
                                alt.Tooltip("density:Q", format=".3f"),
                                alt.Tooltip("n_reviews:Q", title="N reviews") ],
                  )
                  .properties(width=args.market_width, height=args.market_height, title="Mercado")
               ).to_dict()

    brands_spec = (alt.Chart(brand_den)
                    .mark_area(orient="horizontal", opacity=0.85)
                    .encode(
                        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=list(extent))),
                        x=alt.X("density:Q", stack="center", title=None,
                                axis=alt.Axis(labels=False, ticks=False)),
                        color=alt.Color("brand:N", legend=None),
                        tooltip=[ "brand:N", alt.Tooltip("value:Q", format=".2f"),
                                  alt.Tooltip("density:Q", format=".3f"),
                                  alt.Tooltip("n_reviews:Q", title="N reviews") ],
                    )
                    .properties(width=args.brand_width, height=args.brand_height)
                    .facet(facet=alt.Facet("brand:N",
                                           header=alt.Header(title="Perfiles por marca (top por N)", labelLimit=120)),
                           columns=args.brand_cols)
                    .resolve_scale(y="shared", x="independent", color="independent")
                  ).to_dict()

    html_str = STICKY_HTML \
        .replace("__MARKET_SPEC__", json.dumps(mkt_spec)) \
        .replace("__BRANDS_SPEC__", json.dumps(brands_spec)) \
        .replace("__OPTIONS__", json.dumps(options))

    os.makedirs(os.path.dirname(args.output_html_sticky), exist_ok=True)
    with open(args.output_html_sticky, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"[OK] sticky -> {args.output_html_sticky}")
    print(f"[INFO] Selector incluye: {len(options)} opciones (Total Skincare + {len(options)-1} secundarias)")

if __name__ == "__main__":
    main()











