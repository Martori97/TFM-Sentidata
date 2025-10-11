#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-10 productos por loves (global y por categoría).
- Lee rutas de params.yaml (absa_paths.delta_product_reviews y absa_outputs.loves_top10).
- Permite sobreescribir por CLI con --src y --outdir.
"""

import argparse, os, yaml, duckdb, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--src", help="Carpeta con ficheros parquet (por ej. Delta folder) para leer product_reviews")
    ap.add_argument("--outdir", help="Carpeta de salida para CSVs")
    args = ap.parse_args()

    # Carga params.yaml si existe (permite correr solo con CLI)
    cfg = {}
    if os.path.exists(args.params):
        cfg = yaml.safe_load(open(args.params, "r", encoding="utf-8")) or {}

    # Lee bloques con fallback
    p_all = cfg.get("absa_paths", cfg.get("paths", {}))
    o_all = cfg.get("absa_outputs", cfg.get("outputs", {}))

    # Resuelve src y outdir con prioridad CLI > params > default
    src = args.src or p_all.get("delta_product_reviews")
    if not src:
        print("❌ No se proporcionó fuente de datos. Use --src o defina absa_paths.delta_product_reviews en params.yaml", file=sys.stderr)
        sys.exit(2)

    outdir = args.outdir or o_all.get("loves_top10", "reports/absa/loves_top10")
    os.makedirs(outdir, exist_ok=True)

    con = duckdb.connect(database=":memory:")

    # Global top-10
    con.execute(f"""
    COPY (
      SELECT 
        product_id,
        ANY_VALUE(pinfo_product_name) AS product_name,
        ANY_VALUE(pinfo_brand_name)   AS brand_name,
        MAX(pinfo_loves_count)        AS loves
      FROM '{src}/*.parquet'
      WHERE pinfo_loves_count IS NOT NULL
      GROUP BY product_id
      ORDER BY loves DESC
      LIMIT 10
    ) TO '{outdir}/global_top10.csv' (HEADER, DELIMITER ',');
    """)

    # Top-10 por categoría
    con.execute(f"""
    COPY (
      SELECT *
      FROM (
        SELECT 
          pinfo_secondary_category AS category,
          product_id,
          ANY_VALUE(pinfo_product_name) AS product_name,
          ANY_VALUE(pinfo_brand_name)   AS brand_name,
          MAX(pinfo_loves_count)        AS loves,
          ROW_NUMBER() OVER (
            PARTITION BY pinfo_secondary_category
            ORDER BY MAX(pinfo_loves_count) DESC
          ) AS rk
        FROM '{src}/*.parquet'
        WHERE pinfo_loves_count IS NOT NULL
        GROUP BY pinfo_secondary_category, product_id
      ) t
      WHERE rk <= 10
      ORDER BY category, loves DESC
    ) TO '{outdir}/per_category_top10.csv' (HEADER, DELIMITER ',');
    """)
    print(f"[ok] CSVs en {outdir}")

if __name__ == "__main__":
    main()
