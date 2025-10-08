#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def normalize(txt):
    if pd.isna(txt): return ""
    txt = str(txt).lower()
    txt = re.sub(r"[^a-z0-9\s']", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def top_terms(texts_pos, texts_bg, top=40, ngram=(1,2), min_df=5):
    if len(texts_pos) == 0 or len(texts_bg) == 0:
        return pd.DataFrame(columns=["term","score"])
    vect = TfidfVectorizer(ngram_range=ngram, min_df=min_df, stop_words="english")
    X_bg = vect.fit_transform(texts_bg)
    X_pos = vect.transform(texts_pos)
    s_pos = np.asarray(X_pos.mean(axis=0)).ravel()
    s_bg  = np.asarray(X_bg.mean(axis=0)).ravel()
    diff = s_pos - s_bg
    idx = diff.argsort()[::-1][:top]
    return pd.DataFrame({"term": np.array(vect.get_feature_names_out())[idx],
                         "score": diff[idx]})

def main():
    ap = argparse.ArgumentParser("Deep dive textual para Top-Riesgos")
    ap.add_argument("--preview_parquet", default="reports/absa/final_all/preview_fused.parquet")
    ap.add_argument("--top_risks_csv", default="reports/viz/top_risks_overall.csv")
    ap.add_argument("--out_dir", default="reports/deep_dive")
    ap.add_argument("--top_cells", type=int, default=20)
    ap.add_argument("--neg_label", type=int, default=0, help="label negativo en pred_3 (0 ó -1)")
    ap.add_argument("--sample_reviews_per_cell", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_parquet(args.preview_parquet)
    df["category"] = df["category"].fillna("Unknown")

    risks = pd.read_csv(args.top_risks_csv).head(args.top_cells)

    for _, r in risks.iterrows():
        b, c, a = r["brand"], r["category"], r["aspect_norm"]
        cell = df[(df.brand==b) & (df.category==c) & (df.aspect_norm==a)]
        neg = cell[cell.pred_3==args.neg_label].copy()
        bg = df[(df.category==c) & (df.aspect_norm==a) & (df.brand!=b) & (df.pred_3==args.neg_label)].copy()

        neg["txt"] = neg["review_text_clean"].map(normalize)
        bg["txt"]  = bg["review_text_clean"].map(normalize)

        terms = top_terms(neg["txt"].tolist(), bg["txt"].tolist(), top=40, ngram=(1,2), min_df=8)
        terms["brand"], terms["category"], terms["aspect_norm"] = b, c, a
        terms.to_csv(os.path.join(args.out_dir, f"terms__{b}__{c}__{a}.csv"), index=False)

        co = (neg.groupby("review_id")["aspect_norm"].apply(set).reset_index()
                .explode("aspect_norm").query("aspect_norm != @a")
                .aspect_norm.value_counts().reset_index())
        co.columns = ["aspect","count"]
        co["brand"], co["category"], co["anchor_aspect"] = b, c, a
        co.to_csv(os.path.join(args.out_dir, f"coaspects__{b}__{c}__{a}.csv"), index=False)

        sample = neg[["review_id","product_id","product_name","brand","category","aspect_norm",
                      "review_text_clean","rating","pred_3","p_pos","p_neg"]].drop_duplicates("review_id") \
                     .head(args.sample_reviews_per_cell)
        sample.to_csv(os.path.join(args.out_dir, f"sample_reviews__{b}__{c}__{a}.csv"), index=False)

    print("[OK] Deep-dive packs ->", args.out_dir)

if __name__ == "__main__":
    main()

