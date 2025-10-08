#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingredientes diferenciales por (brand, secondary category) frente al resto de su categoría.
Fallback: si el snapshot no trae ingredientes, los cargamos desde la Delta de reviews.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from collections import Counter

STOP_ING = {
    "water","aqua","eau","fragrance","parfum","perfume",
    "alcohol","denat","colorant","ci","mica","titanium dioxide"
}

def parse_ingredients(s):
    if pd.isna(s): return []
    s = str(s).lower()
    parts = re.split(r"[;/,]+", s)
    toks = []
    for p in parts:
        p = re.sub(r"\(.*?\)", "", p).strip()
        if not p: continue
        p = re.sub(r"[^a-z0-9\s\-']", " ", p).strip()
        p = re.sub(r"\s+", " ", p)
        if len(p) < 3: continue
        if p in STOP_ING: continue
        toks.append(p)
    return toks

def log_odds_ratio(count_a, total_a, count_b, total_b, alpha=0.5):
    pa = (count_a + alpha) / (total_a + alpha * 2)
    pb = (count_b + alpha) / (total_b + alpha * 2)
    return np.log(pa / (1 - pa)) - np.log(pb / (1 - pb))

def _first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_products_with_ingredients(products_parquet: str,
                                   reviews_delta: str = "data/trusted/reviews_product_info_clean_full") -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas: product_id, brand, category (secondary), ingredients_tokens.
    Intenta leer del snapshot; si no hay columna de ingredientes, cae a Delta reviews.
    """
    if os.path.exists(products_parquet):
        prod = pd.read_parquet(products_parquet)
        ing_col = _first_present(prod, ["pinfo_ingredients","ingredients","INCI","inci"])
        brand_col = _first_present(prod, ["brand_name","pinfo_brand_name","brand"])
        cat_col = _first_present(prod, ["pinfo_secondary_category","secondary_category","category_secondary","category"])
        if ing_col and brand_col and cat_col:
            out = prod[["product_id", brand_col, cat_col, ing_col]].copy()
            out.rename(columns={brand_col:"brand", cat_col:"category", ing_col:"ingredients"}, inplace=True)
            out["ingredients_tokens"] = out["ingredients"].map(parse_ingredients)
            out = out[["product_id","brand","category","ingredients_tokens"]]
            n_ok = (out["ingredients_tokens"].str.len() > 0).sum()
            print(f"[INFO] Ingredientes desde SNAPSHOT: {n_ok} productos con tokens (>0).")
            if n_ok > 0:
                return out

    # Fallback a Delta reviews (Spark)
    print("[WARN] No hay columna de ingredientes válida en snapshot. Intento cargar desde Delta reviews…")
    try:
        from delta import configure_spark_with_delta_pip
        from pyspark.sql import SparkSession, functions as F

        builder = (
            SparkSession.builder.appName("ingredients-fallback")
            .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        sdf = spark.read.format("delta").load(reviews_delta)

        # columnas posibles
        brand = None
        for c in ["pinfo_brand_name","brand_name"]:
            if c in sdf.columns: brand = c; break
        cat = None
        for c in ["pinfo_secondary_category","secondary_category","category_secondary"]:
            if c in sdf.columns: cat = c; break
        ing = None
        for c in ["pinfo_ingredients","ingredients","INCI","inci"]:
            if c in sdf.columns: ing = c; break

        if not (brand and cat and ing):
            spark.stop()
            raise RuntimeError("Delta de reviews no contiene brand/secondary/ingredients necesarios.")

        sdf2 = (sdf.select(
                    "product_id",
                    F.coalesce(brand).alias("brand"),
                    F.coalesce(cat).alias("category"),
                    F.col(ing).alias("ingredients")
                )
                .where(F.col("product_id").isNotNull())
                .dropna(subset=["ingredients"])
                .dropDuplicates(["product_id"]))
        pdf = sdf2.toPandas()
        spark.stop()

        pdf["ingredients_tokens"] = pdf["ingredients"].map(parse_ingredients)
        out = pdf[["product_id","brand","category","ingredients_tokens"]]
        n_ok = (out["ingredients_tokens"].str.len() > 0).sum()
        print(f"[INFO] Ingredientes desde DELTA: {n_ok} productos con tokens (>0).")
        if n_ok == 0:
            raise RuntimeError("No se obtuvieron tokens de ingredientes desde Delta.")
        return out

    except Exception as e:
        raise RuntimeError(f"Fallback a Delta reviews fallido: {e}")

def main():
    ap = argparse.ArgumentParser("Ingredientes diferenciales para Top-Riesgos (con fallback a Delta)")
    ap.add_argument("--products_parquet", default="data/exploitation/product_info_snapshot.parquet")
    ap.add_argument("--top_risks_csv", default="reports/viz/simple_top_risks_overall.csv")
    ap.add_argument("--out_dir", default="reports/ingredients_diff")
    ap.add_argument("--top_cells", type=int, default=20)
    ap.add_argument("--top_terms", type=int, default=40)
    ap.add_argument("--reviews_delta", default="data/trusted/reviews_product_info_clean_full",
                    help="usado como fallback si el snapshot no tiene ingredientes")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Carga productos con ingredientes (snapshot o delta)
    prod = load_products_with_ingredients(args.products_parquet, args.reviews_delta)

    # Normaliza y dedup
    prod["brand"] = prod["brand"].fillna("Unknown")
    prod["category"] = prod["category"].fillna("Unknown")
    prod = prod.drop_duplicates(subset=["product_id"])

    # Lee top riesgos
    risks = pd.read_csv(args.top_risks_csv).head(args.top_cells)
    if risks.empty:
        raise SystemExit("[ERROR] No hay filas en top_risks_csv. Ejecuta priority_tables_simple/priority_tables antes.")

    # Para cada celda riesgo: comparar ingredientes del brand vs resto de la categoría (presencia por producto)
    for _, r in risks.iterrows():
        b, c = r["brand"], r["category"]
        A = prod[(prod.brand == b) & (prod.category == c)]
        B = prod[(prod.brand != b) & (prod.category == c)]

        if len(A) == 0 or len(B) == 0:
            print(f"[WARN] Saltando {b} / {c}: no hay productos suficientes (A={len(A)}, B={len(B)}).")
            continue

        # presencia a nivel producto (set por producto)
        toks_A = [t for toks in A["ingredients_tokens"] for t in set(toks)]
        toks_B = [t for toks in B["ingredients_tokens"] for t in set(toks)]

        ca, cb = Counter(toks_A), Counter(toks_B)
        all_terms = sorted(set(ca.keys()) | set(cb.keys()))
        total_a, total_b = max(1, len(A)), max(1, len(B))

        rows = []
        for term in all_terms:
            la, lb = ca.get(term, 0), cb.get(term, 0)
            lor = log_odds_ratio(la, total_a, lb, total_b, alpha=0.5)
            rows.append((term, la, lb, lor))

        out = (pd.DataFrame(rows, columns=["ingredient","n_brand","n_rest","log_odds"])
                 .sort_values("log_odds", ascending=False)
                 .head(args.top_terms))
        out["brand"], out["category"] = b, c

        out_path = os.path.join(args.out_dir, f"ingredients_diff__{b}__{c}.csv")
        out.to_csv(out_path, index=False)
        print("[OK] ->", out_path)

    print("[OK] Ingredientes diferenciales ->", args.out_dir)

if __name__ == "__main__":
    main()

