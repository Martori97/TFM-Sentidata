#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cribaje SIMPLE de Top-Riesgos y Top-Fortalezas a partir de reports/viz/negatives_table.csv

Riesgo:     excess_neg >= 0.03  AND complaint_rate_y >= 0.12
Fortaleza:  excess_neg <= -0.03 AND complaint_rate_y <= 0.08 AND quality_x >= 0.65
Filtros:    n_reviews >= 50 AND volume_w >= 10
"""

import os
import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser("Cribaje simple Top-Riesgos / Top-Fortalezas (secondary category)")
    ap.add_argument("--input_csv", default="reports/viz/negatives_table.csv")
    ap.add_argument("--out_dir", default="reports/viz")
    ap.add_argument("--thr_excess_high", type=float, default=0.03)
    ap.add_argument("--thr_compl_high", type=float, default=0.12)
    ap.add_argument("--thr_compl_low",  type=float, default=0.08)
    ap.add_argument("--thr_quality_high", type=float, default=0.65)
    ap.add_argument("--min_reviews", type=int, default=50)
    ap.add_argument("--min_volume_w", type=float, default=10.0)
    ap.add_argument("--top_k_overall", type=int, default=100)
    ap.add_argument("--top_k_per_cat", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    keep = (df["n_reviews"] >= args.min_reviews) & (df["volume_w"] >= args.min_volume_w)
    df = df.loc[keep].copy()
    if df.empty:
        raise SystemExit("[ERROR] No hay filas tras filtros; ajusta --min_reviews / --min_volume_w.")

    e, c, q = df["excess_neg"].astype(float), df["complaint_rate_y"].astype(float), df["quality_x"].astype(float)
    df["is_risk"] = (e >= args.thr_excess_high) & (c >= args.thr_compl_high)
    df["is_strength"] = (e <= -args.thr_excess_high) & (c <= args.thr_compl_low) & (q >= args.thr_quality_high)

    def order_risk(d): return d.sort_values(["volume_w","excess_neg","complaint_rate_y"], ascending=[False, False, False])
    def order_strength(d): return d.sort_values(["volume_w","quality_x","complaint_rate_y"], ascending=[False, False, True])

    risk_overall = order_risk(df[df["is_risk"]]).head(args.top_k_overall)
    strength_overall = order_strength(df[df["is_strength"]]).head(args.top_k_overall)

    risk_overall.to_csv(os.path.join(args.out_dir,"simple_top_risks_overall.csv"), index=False)
    strength_overall.to_csv(os.path.join(args.out_dir,"simple_top_strengths_overall.csv"), index=False)

    risk_cat = (df[df["is_risk"]].groupby("category", group_keys=False).apply(lambda d: order_risk(d).head(args.top_k_per_cat)))
    strength_cat = (df[df["is_strength"]].groupby("category", group_keys=False).apply(lambda d: order_strength(d).head(args.top_k_per_cat)))

    risk_cat.to_csv(os.path.join(args.out_dir,"simple_top_risks_by_category.csv"), index=False)
    strength_cat.to_csv(os.path.join(args.out_dir,"simple_top_strengths_by_category.csv"), index=False)

    df["label_simple"] = np.where(df["is_risk"], "RISK", np.where(df["is_strength"], "STRENGTH", "OTHER"))
    df.to_csv(os.path.join(args.out_dir,"simple_labels_all.csv"), index=False)

    print("[OK] Guardado CSVs en", args.out_dir)

if __name__ == "__main__":
    main()
