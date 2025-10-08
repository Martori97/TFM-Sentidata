#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye tarjetas de producto (JSONL) con:
- Distribución de opiniones (neg/neu/pos)
- Radar por ejes de la ontología (por categoría, con fallback global)
- Top términos positivos y negativos
- Nº de reviews por aspecto (mentions) para barras de peso en HTML
- rating_media como float

Respeta las claves y rutas de tu params/dvc:
- Lee rutas desde absa_paths (o paths) en params.yaml
- Escribe en reports/absa/cards/product_cards.jsonl (igual que antes)

Requiere: duckdb, pyyaml, pandas
"""

import argparse
import os
import json
import math
import yaml
import duckdb
import pandas as pd
from collections import defaultdict


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def soft_get_paths(cfg):
    """
    Usa tus claves existentes sin romper si faltara alguna:
    - Preferencia: cfg['absa_paths']; si no, cfg['paths'].
    - outdir: cfg['absa_outputs']['cards'] o cfg['outputs']['cards'] o default.
    """
    P = cfg.get("absa_paths") or cfg.get("paths") or {}
    spans = P.get("spans")
    mapped = P.get("mapped")
    sentiment = P.get("sentiment")
    snapshot = P.get("product_snapshot") or P.get("product_snapshot_path")  # tolerante

    O = cfg.get("absa_outputs") or cfg.get("outputs") or {}
    outdir = O.get("cards") or "reports/absa/cards"

    if not all([spans, mapped, sentiment, snapshot]):
        raise SystemExit(
            "[paths] Faltan rutas en params.yaml (absa_paths / paths). "
            "Necesitamos: spans, mapped, sentiment, product_snapshot."
        )
    return spans, mapped, sentiment, snapshot, outdir


def minmax_norm(vals, v):
    if not vals:
        return 0.0
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    # overrides opcionales (mantiene tus flags actuales)
    ap.add_argument("--spans")
    ap.add_argument("--mapped")
    ap.add_argument("--sentiment")
    ap.add_argument("--product-snapshot", dest="product_snapshot")
    ap.add_argument("--outdir")
    args = ap.parse_args()

    cfg = load_cfg(args.params)
    if all([args.spans, args.mapped, args.sentiment, args.product_snapshot, args.outdir]):
        spans, mapped, sentiment, snapshot, outdir = (
            args.spans, args.mapped, args.sentiment, args.product_snapshot, args.outdir
        )
    else:
        spans, mapped, sentiment, snapshot, outdir = soft_get_paths(cfg)

    os.makedirs(outdir, exist_ok=True)

    cards_cfg = cfg.get("cards", {})
    axes_by_cat = cards_cfg.get("axes_by_category", {})
    axes_global = cards_cfg.get("axes_global", ["fragrance", "longevity", "texture", "packaging", "price", "effectiveness"])
    min_reviews = int(cards_cfg.get("min_reviews_per_product", 30))
    top_k_terms = int(cards_cfg.get("top_k_terms", 3))
    normalize = str(cards_cfg.get("normalize", "global")).lower()  # "global" o "product"

    con = duckdb.connect(database=":memory:")

    # Vista unificada con normalización de sentimiento a 'pred3_norm'
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

    # Filtra productos con suficiente volumen
    prod_sizes = con.execute("""
      SELECT product_id, COUNT(*) AS n_rows
      FROM absa
      WHERE product_id IS NOT NULL
      GROUP BY product_id
    """).df()
    valid_products = set(prod_sizes.loc[prod_sizes["n_rows"] >= min_reviews, "product_id"].tolist())
    if not valid_products and not prod_sizes.empty:
        valid_products = set(prod_sizes["product_id"].tolist())

    # Pivot producto x aspecto (conteos por sentimiento)
    pivot = con.execute("""
      WITH agg AS (
        SELECT product_id, aspect_norm, pred3_norm, COUNT(*) AS n
        FROM absa
        WHERE product_id IS NOT NULL
        GROUP BY product_id, aspect_norm, pred3_norm
      )
      SELECT
        product_id, aspect_norm,
        COALESCE(SUM(CASE WHEN pred3_norm='neg' THEN n END), 0) AS neg,
        COALESCE(SUM(CASE WHEN pred3_norm='neu' THEN n END), 0) AS neu,
        COALESCE(SUM(CASE WHEN pred3_norm='pos' THEN n END), 0) AS pos
      FROM agg
      GROUP BY product_id, aspect_norm
    """).df()

    if pivot.empty:
        print("[warn] pivot vacío: no hay datos para construir tarjetas")
        return

    pivot["volume_total"] = pivot["neg"] + pivot["neu"] + pivot["pos"]
    denom = pivot["volume_total"].replace(0, 1)
    pivot["pos_rate"] = pivot["pos"] / denom
    pivot["neg_rate"] = pivot["neg"] / denom
    pivot["impact_score"] = (pivot["pos_rate"] - pivot["neg_rate"]) * (pivot["volume_total"].apply(lambda x: math.log1p(x)))

    # Distribución por producto
    dist = con.execute("""
      SELECT product_id,
             SUM(CASE WHEN pred3_norm='neg' THEN 1 ELSE 0 END) AS neg,
             SUM(CASE WHEN pred3_norm='neu' THEN 1 ELSE 0 END) AS neu,
             SUM(CASE WHEN pred3_norm='pos' THEN 1 ELSE 0 END) AS pos
      FROM absa
      WHERE product_id IS NOT NULL
      GROUP BY product_id
    """).df()
    dist["total"] = dist["neg"] + dist["neu"] + dist["pos"]
    dden = dist["total"].replace(0, 1)
    dist["neg_share"] = dist["neg"] / dden
    dist["neu_share"] = dist["neu"] / dden
    dist["pos_share"] = dist["pos"] / dden

    # Nº de reviews (distinct) por producto y por (producto, aspecto)
    reviews_per_product = con.execute("""
      SELECT product_id, COUNT(DISTINCT review_id) AS reviews_total
      FROM absa
      WHERE product_id IS NOT NULL
      GROUP BY product_id
    """).df()

    mentions_per_axis = con.execute("""
      SELECT product_id, aspect_norm, COUNT(DISTINCT review_id) AS reviews_with_aspect
      FROM absa
      WHERE product_id IS NOT NULL
      GROUP BY product_id, aspect_norm
    """).df()

    # Metadatos (rating/price como float)
    meta = con.execute("""
      SELECT product_id,
             ANY_VALUE(product_name) AS product_name,
             ANY_VALUE(brand_name)   AS brand_name,
             ANY_VALUE(category2)    AS category2,
             MAX(TRY_CAST(pinfo_loves_count AS BIGINT))  AS loves,
             AVG(TRY_CAST(pinfo_rating     AS DOUBLE))   AS rating_product,
             AVG(TRY_CAST(pinfo_price_usd  AS DOUBLE))   AS price_usd
      FROM absa
      WHERE product_id IS NOT NULL
      GROUP BY product_id
    """).df()

    # Términos
    terms = con.execute("""
      SELECT product_id, aspect_norm, lower(span_text) AS term, pred3_norm, COUNT(*) AS n
      FROM absa
      WHERE product_id IS NOT NULL
      GROUP BY product_id, aspect_norm, lower(span_text), pred3_norm
    """).df()

    pivot_g = pivot.groupby("product_id")
    dist_i = dist.set_index("product_id")
    meta_i = meta.set_index("product_id")
    revs_i = reviews_per_product.set_index("product_id")
    ment_i = mentions_per_axis.set_index(["product_id","aspect_norm"])

    def axes_for_cat(cat):
        if cat and cat in axes_by_cat:
            return axes_by_cat[cat]
        return axes_global

    # Normalización global si procede
    impact_global = defaultdict(list)
    if normalize == "global":
        for pid, grp in pivot_g:
            if valid_products and pid not in valid_products:
                continue
            cat = None
            if pid in meta_i.index:
                try:
                    cat = meta_i.loc[pid, "category2"]
                except Exception:
                    cat = None
            axes = axes_for_cat(cat)
            impact_map = dict(zip(grp["aspect_norm"].tolist(), grp["impact_score"].tolist()))
            for a in axes:
                impact_global[a].append(float(impact_map.get(a, 0.0)))

    # Salida JSONL (misma ruta de siempre)
    out_path = os.path.join(outdir, "product_cards.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for pid, grp in pivot_g:
            if valid_products and pid not in valid_products:
                continue

            # Meta
            m = meta_i.loc[pid] if pid in meta_i.index else {}
            product_name = m.get("product_name", None)
            brand_name = m.get("brand_name", None)
            category2 = m.get("category2", None)
            loves = float(m.get("loves", 0) or 0)
            rating_prod = float(m.get("rating_product", 0) or 0)
            price = float(m.get("price_usd", 0) or 0)

            # shares y reviews_analizadas
            if pid in dist_i.index:
                d = dist_i.loc[pid]
                neg_share = float(d.get("neg_share", 0))
                neu_share = float(d.get("neu_share", 0))
                pos_share = float(d.get("pos_share", 0))
            else:
                neg_share = neu_share = pos_share = 0.0
            reviews_analizadas = int(revs_i.loc[pid, "reviews_total"]) if pid in revs_i.index else 0

            # Impacto por aspecto y radar
            impact_map = dict(zip(grp["aspect_norm"].tolist(), grp["impact_score"].tolist()))
            axes = axes_for_cat(category2)
            radar = {}
            if normalize == "product":
                vals = [impact_map.get(a, 0.0) for a in axes]
                for a in axes:
                    radar[a] = round(minmax_norm(vals, impact_map.get(a, 0.0)), 4)
            else:
                for a in axes:
                    v = float(impact_map.get(a, 0.0))
                    radar[a] = round(minmax_norm(impact_global.get(a, []), v), 4)

            # Mentions por eje (distinct reviews que lo mencionan)
            mentions = {}
            for a in axes:
                key = (pid, a)
                if key in ment_i.index:
                    try:
                        mentions[a] = int(ment_i.loc[key, "reviews_with_aspect"])
                    except Exception:
                        mentions[a] = int(pd.Series(ment_i.loc[key, "reviews_with_aspect"]).iloc[0])
                else:
                    mentions[a] = 0

            # Top términos ±
            tsub = terms[terms["product_id"] == pid]
            pos_terms = (
                tsub[tsub["pred3_norm"] == "pos"]
                .sort_values("n", ascending=False)
                .head(top_k_terms)["term"].tolist()
            )
            neg_terms = (
                tsub[tsub["pred3_norm"] == "neg"]
                .sort_values("n", ascending=False)
                .head(top_k_terms)["term"].tolist()
            )

            card = {
                "product_id": str(pid),
                "product_name": product_name,
                "brand_name": brand_name,
                "category2": category2,
                "rating_media": round(rating_prod, 2),
                "price_usd": price,
                "loves": int(loves),
                "reviews_analizadas": reviews_analizadas,
                "opinion_distribution": {
                    "neg": round(neg_share, 3),
                    "neu": round(neu_share, 3),
                    "pos": round(pos_share, 3),
                },
                "radar": radar,
                "mentions": mentions,  # <-- NUEVO
                "top_positive": pos_terms,
                "top_negative": neg_terms,
            }

            fout.write(json.dumps(card, ensure_ascii=False) + "\n")

    print("[ok] product cards ->", out_path)


if __name__ == "__main__":
    main()




