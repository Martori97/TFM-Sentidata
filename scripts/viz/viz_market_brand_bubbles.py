#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market (violin) left + brand bubbles with drill-down to products (sticky HTML)

- Quality = p_pos - p_neg  (∈[-1,1])
- Brand bubbles: X domain [0, 1]; size = n_reviews (sumados a nivel marca)
- Product bubbles: X domain [-1, 1]; size = n_reviews (por producto)
- Secondary category inject (map -> delta -> snapshot)
- Reviews dedup opcional
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import altair as alt


# ----------------- utils -----------------
def parse_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def add_loves_map(products_parquet: str) -> pd.DataFrame:
    """
    Devuelve loves absolutos por product_id (deduplicado).
    Solo para tooltip; ya no se usan para size.
    """
    if not products_parquet or not os.path.exists(products_parquet):
        return pd.DataFrame(columns=["product_id", "loves"])
    prod = pd.read_parquet(products_parquet)
    col = None
    for c in ["pinfo_loves_count", "loves", "loves_count"]:
        if c in prod.columns:
            col = c
            break
    if col is None:
        return pd.DataFrame(columns=["product_id", "loves"])
    out = (
        prod[["product_id", col]]
        .rename(columns={col: "loves"})
        .dropna()
        .drop_duplicates("product_id")
    )
    out["loves"] = pd.to_numeric(out["loves"], errors="coerce").fillna(0.0)
    return out


def load_secondary_map(parquet_path: str) -> pd.DataFrame | None:
    if parquet_path and os.path.exists(parquet_path):
        m = pd.read_parquet(parquet_path)
        for c in ["secondary_category", "pinfo_secondary_category", "category", "sec_cat"]:
            if c in m.columns:
                out = (
                    m[["product_id", c]]
                    .rename(columns={c: "secondary_category"})
                    .dropna()
                    .drop_duplicates()
                )
                print(f"[INFO] Secondary desde MAP: {parquet_path} rows={len(out):,} col='{c}'")
                return out
        print(f"[WARN] {parquet_path} no contiene columna de secundaria reconocible")
    return None


def load_secondary_from_delta(reviews_delta: str) -> pd.DataFrame | None:
    if not reviews_delta:
        return None
    try:
        from delta import configure_spark_with_delta_pip
        from pyspark.sql import SparkSession, functions as F

        builder = (
            SparkSession.builder.appName("sec-from-delta")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()

        sdf = spark.read.format("delta").load(reviews_delta)

        col = None
        for c in ["pinfo_secondary_category", "secondary_category", "category_secondary"]:
            if c in sdf.columns:
                col = c; break
        if col is None:
            for c in ["pinfo_primary_category", "category"]:
                if c in sdf.columns:
                    col = c; break
        if col is None:
            print("[WARN] Delta sin columna de categoría")
            spark.stop()
            return None

        pdf = (
            sdf.select("product_id", F.col(col).alias("secondary_category"))
            .where("product_id is not null")
            .dropna()
            .dropDuplicates(["product_id"])
            .toPandas()
        )
        spark.stop()
        print(f"[INFO] Secondary desde DELTA: {reviews_delta} rows={len(pdf):,} col='{col}'")
        return pdf
    except Exception as e:
        print(f"[WARN] No pude leer delta: {e}")
        return None


def load_secondary_from_snapshot(products_parquet: str) -> pd.DataFrame | None:
    if not products_parquet or not os.path.exists(products_parquet):
        return None
    prod = pd.read_parquet(products_parquet)
    for c in ["pinfo_secondary_category", "secondary_category", "category_secondary"]:
        if c in prod.columns:
            out = (
                prod[["product_id", c]]
                .rename(columns={c: "secondary_category"})
                .dropna()
                .drop_duplicates()
            )
            print(f"[INFO] Secondary desde SNAPSHOT: rows={len(out):,}")
            return out
    print("[WARN] Snapshot sin columna secondary")
    return None


def inject_secondary(
    df: pd.DataFrame, secondary_map_parquet: str, reviews_delta: str, products_parquet: str
) -> pd.DataFrame:
    """
    Imputa secondary en este orden: mapa externo -> delta -> snapshot.
    """
    for loader, tag in [
        (lambda: load_secondary_map(secondary_map_parquet), "MAP"),
        (lambda: load_secondary_from_delta(reviews_delta), "DELTA"),
        (lambda: load_secondary_from_snapshot(products_parquet), "SNAPSHOT"),
    ]:
        m = loader()
        if m is not None and not m.empty:
            before = df["category"].nunique(dropna=True)
            out = df.merge(m, on="product_id", how="left")
            out["category"] = out["secondary_category"].fillna(out["category"])
            out.drop(columns=["secondary_category"], inplace=True)
            after = out["category"].nunique(dropna=True)
            print(f"[INFO] category nunique: {before} -> {after} tras aplicar secondary ({tag})")
            return out
    print("[WARN] No pude aplicar secondary; se deja la original")
    return df


def build_market_violin(
    df_cat: pd.DataFrame, metric_col: str = "rating", bins: int = 300, lo: float = 1.0, hi: float = 5.0, jitter: float = 0.12
) -> pd.DataFrame:
    """
    Histograma normalizado (aprox. densidad) para el violín horizontal.
    """
    vals = pd.to_numeric(df_cat[metric_col], errors="coerce").fillna(np.nan)
    vals = vals.clip(lo, hi).dropna().to_numpy()
    if len(vals) == 0:
        return pd.DataFrame({"value": [], "density": []})
    if jitter and jitter > 0:
        rng = np.random.default_rng(42)
        vals = np.clip(vals + rng.normal(0.0, jitter, len(vals)), lo, hi)
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist, _ = np.histogram(vals, bins=edges, density=True)
    area = hist.sum() * (hi - lo) / bins
    if area > 0:
        hist /= area
    return pd.DataFrame({"value": centers, "density": hist})


# ----------------- agregaciones -----------------
def aggregate_product_level(
    df_reviews: pd.DataFrame, loves_map: pd.DataFrame
) -> pd.DataFrame:
    """
    Producto:
      - quality = mean(p_pos - p_neg) por review
      - complaints = % pred_3 == 0
      - n_reviews = nº reviews únicos
      - loves (solo tooltip)
    """
    tmp = df_reviews.assign(
        q=(pd.to_numeric(df_reviews["p_pos"], errors="coerce") - pd.to_numeric(df_reviews["p_neg"], errors="coerce")),
        is_neg=(df_reviews["pred_3"] == 0).astype(int),
    )

    p = (
        tmp.groupby(["category", "brand", "product_id", "product_name"], dropna=False)
        .agg(
            quality=("q", "mean"),
            complaints=("is_neg", "mean"),
            n_reviews=("review_id", "nunique"),
        )
        .reset_index()
    )
    p["product_name"] = p["product_name"].where(p["product_name"].notna(), p["product_id"])

    if loves_map is not None and not loves_map.empty:
        p = p.merge(loves_map, on="product_id", how="left")
    else:
        p["loves"] = 0.0
    p["loves"] = pd.to_numeric(p["loves"], errors="coerce").fillna(0.0)

    return p


def aggregate_brand_from_products(prod_df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca (desde productos):
      - quality: media ponderada por nº reviews
      - complaints: media ponderada por nº reviews
      - n_reviews: suma nº reviews
      - sum_loves: suma de loves (solo tooltip)
    """
    rows = []
    for (cat, brand), d in prod_df.groupby(["category", "brand"], dropna=False):
        nrev = pd.to_numeric(d["n_reviews"], errors="coerce").fillna(0).to_numpy()
        q = pd.to_numeric(d["quality"], errors="coerce").fillna(0).to_numpy()
        comp = pd.to_numeric(d["complaints"], errors="coerce").fillna(0).to_numpy()

        wsum = nrev.sum()
        quality = float(np.average(q, weights=nrev)) if wsum > 0 else float(q.mean() if len(q) else 0.0)
        complaints = float(np.average(comp, weights=nrev)) if wsum > 0 else float(comp.mean() if len(comp) else 0.0)

        rows.append(
            {
                "category": cat,
                "brand": brand,
                "quality": quality,
                "complaints": complaints,
                "n_reviews": int(pd.to_numeric(d["n_reviews"], errors="coerce").fillna(0).sum()),
                "sum_loves": float(pd.to_numeric(d["loves"], errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows)


# ----------------- plantilla HTML (sticky + click listener) -----------------
HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Market & Brand/Product bubbles (sticky)</title>
<style>
  html, body { height: 100%; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
  .grid { display:grid; grid-template-columns: 360px 1fr; gap:16px; height:100vh; }
  .left { position:sticky; top:0; height:100vh; background:#fff; padding:12px; border-right:1px solid #eee; }
  .controls { position:sticky; top:0; background:#fff; z-index:10; padding-bottom:8px; border-bottom:1px solid #eee; margin-bottom:8px; }
  .right { overflow-y:auto; height:100vh; padding:12px; }
  label { font-size:14px; margin-right:6px; }
  select { padding:4px 8px; font-size:14px; }
  h3 { margin:6px 0 2px; }
</style>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head>
<body>
<div class="grid">
  <div class="left">
    <div class="controls">
      <label for="secSel"><b>Secondary category:</b></label>
      <select id="secSel"></select>
    </div>
    <div id="market"></div>
  </div>
  <div class="right">
    <h3>Brands</h3>
    <div id="brands"></div>
    <h3>Products <span id="brandHint" style="color:#666;font-weight:normal;"></span></h3>
    <div id="products"></div>
  </div>
</div>
<script>
  const marketSpec  = __MKT__;
  const brandsSpec  = __BR__;
  const productsSpec= __PR__;
  const options     = __OPTS__;
  const initCat = options[0];
  const sel = document.getElementById('secSel');
  const brandHint = document.getElementById('brandHint');
  options.forEach(o => { const op=document.createElement('option'); op.value=o; op.textContent=o; sel.appendChild(op); });
  sel.value = initCat;

  let currentBrand = null;

  function clone(obj){ return JSON.parse(JSON.stringify(obj)); }

  function specsFor(cat, brand=null) {
    const m = clone(marketSpec);
    m.transform = (m.transform || []);
    m.transform.unshift({filter:`datum.category == '${cat.replace(/'/g,"\\'")}'`});

    const b = clone(brandsSpec);
    b.transform = (b.transform || []);
    b.transform.unshift({filter:`datum.category == '${cat.replace(/'/g,"\\'")}'`});

    const p = clone(productsSpec);
    p.transform = (p.transform || []);
    const filters = [`datum.category == '${cat.replace(/'/g,"\\'")}'`];
    if (brand) { filters.push(`datum.brand == '${brand.replace(/'/g,"\\'")}'`); }
    p.transform.unshift({filter: filters.join(' && ')});
    return {m,b,p};
  }

  async function render(cat, brand=null) {
    brandHint.textContent = brand ? `— ${brand}` : '';
    const {m,b,p} = specsFor(cat, brand);

    // Brands
    const bView = await vegaEmbed('#brands', b, {actions:false});
    bView.view.addEventListener('click', (event, item) => {
      if (item && item.datum && item.datum.brand) {
        currentBrand = item.datum.brand;
        render(sel.value, currentBrand);
      }
    });

    // Mercado
    await vegaEmbed('#market', m, {actions:false});

    // Productos
    await vegaEmbed('#products', p, {actions:false});
  }

  sel.addEventListener('change', e => { currentBrand=null; render(e.target.value, null); });
  render(initCat, null);
</script>
</body>
</html>
"""


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(
        "Market left (violin) + brands bubbles with drill-down to products"
    )
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--products_parquet", default="")
    ap.add_argument("--reviews_delta", default="")
    ap.add_argument("--secondary_map_parquet", nargs="?", default="")
    ap.add_argument("--output_html", default="reports/viz/market_brand_bubbles.html")
    ap.add_argument("--dedup_reviews", default="true")
    ap.add_argument("--min_reviews_per_cat", type=int, default=200)
    ap.add_argument("--max_categories", type=int, default=40)
    ap.add_argument("--weight_cap_pct", type=float, default=95.0)  # compat, ignorado
    ap.add_argument("--top_products", type=int, default=200)
    args = ap.parse_args()

    alt.data_transformers.disable_max_rows()

    df = pd.read_parquet(args.input_parquet)

    # Columnas mínimas
    for c in ["review_id","product_id","product_name","brand","category","rating","pred_3","p_pos","p_neg"]:
        if c not in df.columns:
            df[c] = np.nan

    # Filtrar filas sin brand/product_id
    df = df[df["brand"].notna() & df["product_id"].notna()]

    # Secondary category
    df = inject_secondary(df, args.secondary_map_parquet, args.reviews_delta, args.products_parquet)

    # Dedup reviews
    if parse_bool(args.dedup_reviews):
        df = df.sort_values("review_id").drop_duplicates(subset=["review_id"])

    # Mapa LOVES (tooltip)
    loves_map = add_loves_map(args.products_parquet)

    # Categorías con masa crítica
    vc = df.groupby("category")["review_id"].nunique().sort_values(ascending=False)
    cats_ok = vc[vc >= args.min_reviews_per_cat].head(args.max_categories).index.tolist()
    df = df[df["category"].isin(cats_ok)]
    cats_sorted = sorted(cats_ok) if cats_ok else ["Skincare"]
    print(f"[INFO] Categorías finales: {len(cats_sorted)} -> {cats_sorted[:8]}...")
    if args.weight_cap_pct is not None:
        print("[INFO] --weight_cap_pct recibido pero IGNORADO (size = n_reviews).")

    # Mercado (violín)
    vlist = []
    for cat in cats_sorted:
        v = build_market_violin(df[df["category"] == cat], "rating")
        v["category"] = cat
        vlist.append(v)
    market_df = pd.concat(vlist, ignore_index=True) if vlist else pd.DataFrame(columns=["value","density","category"])

    # Productos (agregado)
    product_df = aggregate_product_level(df, loves_map)
    # Recorte por marca para evitar saturación (ordenar por nº reviews)
    product_df = product_df.sort_values(["category","brand","n_reviews"], ascending=[True, True, False])
    product_df = product_df.groupby(["category","brand"], as_index=False, group_keys=False).head(args.top_products)

    # Marcas desde productos
    brand_df = aggregate_brand_from_products(product_df)

    # ------- Altair specs -------
    # Mercado
    mkt = (
        alt.Chart(market_df)
        .mark_area(orient="horizontal", opacity=0.9)
        .encode(
            y=alt.Y("value:Q", title="Rating (1–5)", scale=alt.Scale(domain=[1, 5])),
            x=alt.X("density:Q", title="Density", stack="center", axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("category:N", legend=None),
            tooltip=["category:N", alt.Tooltip("value:Q", format=".2f"), alt.Tooltip("density:Q", format=".3f")],
        )
        .properties(width=320, height=340, title="Mercado")
    ).to_dict()

    # Burbujas marca: X [0,1], size = n_reviews (suma por marca)
    br = (
        alt.Chart(brand_df)
        .mark_circle()
        .encode(
            x=alt.X("quality:Q", title="Quality (p_pos − p_neg)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("complaints:Q", title="Complaint rate", scale=alt.Scale(domain=[0.0, 0.5])),
            size=alt.Size("n_reviews:Q", title="# Reviews (brand)", legend=alt.Legend(clipHeight=60)),
            color=alt.Color("brand:N", legend=None),
            tooltip=[
                "brand:N","category:N",
                alt.Tooltip("quality:Q", format=".3f"),
                alt.Tooltip("complaints:Q", format=".3f"),
                alt.Tooltip("n_reviews:Q", format=",.0f"),
                alt.Tooltip("sum_loves:Q", format=",.0f", title="Sum LOVES (brand)"),
            ],
        )
        .properties(width=980, height=320)
    ).to_dict()

    # Burbujas producto: X [-1,1], size = n_reviews (producto)
    pr = (
        alt.Chart(product_df)
        .mark_circle()
        .encode(
            x=alt.X("quality:Q", title="Quality (p_pos − p_neg)", scale=alt.Scale(domain=[-1, 1])),
            y=alt.Y("complaints:Q", title="Complaint rate", scale=alt.Scale(domain=[0.0, 0.5])),
            size=alt.Size("n_reviews:Q", title="# Reviews (product)", legend=None),
            color=alt.Color("brand:N", legend=None),
            tooltip=[
                "brand:N","product_name:N",
                alt.Tooltip("quality:Q", format=".3f"),
                alt.Tooltip("complaints:Q", format=".3f"),
                alt.Tooltip("n_reviews:Q", format=",.0f"),
                alt.Tooltip("loves:Q", format=",.0f", title="LOVES (product)"),
            ],
        )
        .properties(width=980, height=360)
    ).to_dict()

    # HTML ensamblado
    html = (
        HTML.replace("__MKT__", json.dumps(mkt))
            .replace("__BR__", json.dumps(br))
            .replace("__PR__", json.dumps(pr))
            .replace("__OPTS__", json.dumps(cats_sorted))
    )

    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] {args.output_html}")


if __name__ == "__main__":
    main()





