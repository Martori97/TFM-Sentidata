#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspecciona los valores 'radar' en reports/absa/cards/product_cards.jsonl
y compara con los ejes esperados por categoría (params.yaml -> cards.axes_by_category / axes_global).

Imprime:
- Muestra de 5 productos (producto, categoría, radar dict).
- Top-10 aspectos más frecuentes en 'radar' y sus rangos min/max/avg.
- Rango min/max/avg por categoría y por eje esperado (faltantes -> 0).
- Sugerencia de normalización global [0,1].

Opcional: escribe CSVs de resumen en reports/absa/cards/debug_radar_*.csv
"""

import os, json, yaml, math
from collections import Counter, defaultdict
import pandas as pd

JSONL = "reports/absa/cards/product_cards.jsonl"
PARAMS = "params.yaml"
OUTDIR = "reports/absa/cards"

def load_params():
    with open(PARAMS, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cards_cfg = cfg.get("cards", {})
    axes_by_cat = cards_cfg.get("axes_by_category", {})
    axes_global = cards_cfg.get("axes_global", [])
    return axes_by_cat, axes_global

def load_cards():
    rows = []
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            obj = json.loads(line)
            rows.append({
                "product_id": obj.get("product_id"),
                "product_name": obj.get("product_name"),
                "brand_name": obj.get("brand_name"),
                "category2": obj.get("category2"),
                "loves": obj.get("loves", 0),
                "reviews_analizadas": obj.get("reviews_analizadas", 0),
                "radar": obj.get("radar", {}) or {}
            })
    return rows

def main():
    assert os.path.exists(JSONL), f"No existe {JSONL}"
    axes_by_cat, axes_global = load_params()
    cards = load_cards()
    os.makedirs(OUTDIR, exist_ok=True)

    # --- 1) Muestra de 5 productos
    print("\n=== MUESTRA (5 productos) ===")
    for r in cards[:5]:
        print(f"- {r['product_name']} | cat={r['category2']} | radar={r['radar']}")

    # --- 2) Frecuencia de aspectos y rangos globales
    all_pairs = []
    for r in cards:
        for k, v in (r["radar"] or {}).items():
            try:
                val = float(v)
            except Exception:
                continue
            all_pairs.append((k, val))

    if not all_pairs:
        print("\n[ALERTA] No se encontraron valores numéricos en 'radar'.")
        return

    df = pd.DataFrame(all_pairs, columns=["aspect", "value"])
    freq = df["aspect"].value_counts().reset_index().rename(columns={"index": "aspect", "aspect": "count"})
    stats_global = df.groupby("aspect").agg(count=("value", "size"),
                                            min=("value", "min"),
                                            max=("value", "max"),
                                            mean=("value", "mean")).reset_index()

    print("\n=== TOP-10 aspectos más frecuentes (global) ===")
    print(freq.head(10).to_string(index=False))

    print("\n=== RANGOS globales por aspecto ===")
    print(stats_global.sort_values("count", ascending=False).head(15).to_string(index=False))

    stats_global.to_csv(os.path.join(OUTDIR, "debug_radar_stats_global.csv"), index=False)

    # --- 3) Rango por categoría y por ejes esperados
    rows_cat = []
    for cat in sorted(set(r["category2"] for r in cards if r["category2"])):
        axes = axes_by_cat.get(cat, axes_global)[:5]  # mismos 5 ejes
        subset = [r for r in cards if r["category2"] == cat]
        for ax in axes:
            vals = []
            for r in subset:
                v = (r["radar"] or {}).get(ax, 0.0)
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(0.0)
            if not vals:
                continue
            rows_cat.append({
                "category2": cat,
                "axis": ax,
                "n_products_with_axis": sum(1 for x in vals if x != 0),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "mean": float(sum(vals)/len(vals))
            })

    stats_cat = pd.DataFrame(rows_cat)
    if not stats_cat.empty:
        print("\n=== RANGOS por categoría y eje (esperados en params) ===")
        print(stats_cat.sort_values(["category2","axis"]).head(30).to_string(index=False))
        stats_cat.to_csv(os.path.join(OUTDIR, "debug_radar_stats_by_category.csv"), index=False)
    else:
        print("\n[INFO] No se pudieron computar rangos por categoría (¿faltan ejes en params o en radar?).")

    # --- 4) Sugerencia de normalización
    gmin = df["value"].min()
    gmax = df["value"].max()
    print(f"\n=== SUGERENCIA ===\nValores observados en radar: min={gmin:.4f}, max={gmax:.4f}")
    if gmax - gmin < 1e-9:
        print("Todos los valores son prácticamente constantes -> normaliza forzando 0..1 por aspecto")
    else:
        print("Puedes normalizar a [0,1] por aspecto usando (x - min_aspect) / (max_aspect - min_aspect).")
        print("Si prefieres comparación intra-categoría, usa min/max por categoría y por eje.")

    print(f"\n[ok] CSVs de depuración en {OUTDIR}/debug_radar_stats_*.csv")

if __name__ == "__main__":
    main()
