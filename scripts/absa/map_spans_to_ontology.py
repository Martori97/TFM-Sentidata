#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Map raw aspect spans (English) to a fixed ontology of aspects using SBERT cosine similarity + rules.

Inputs:
- Parquet of spans: review_id, aspect_span, prob
- YAML/JSON of ontology with aliases per aspect (e.g., fragrance: [scent, smell, aroma], ...)

Output Parquet:
- review_id, span_text, aspect_norm, score, prob_span

Usage:
python scripts/absa/map_spans_to_ontology.py \
  --spans_parquet reports/absa/ate_spans.parquet \
  --ontology_json configs/aspects_en.json \
  --output_parquet reports/absa/ate_spans_mapped.parquet \
  --threshold 0.45
"""
import argparse, json, re, os
import pandas as pd
import numpy as np

def normalize_text(s:str)->str:
    return re.sub(r"\s+"," ", s.strip().lower())

def build_ontology(ontology_json:str):
    with open(ontology_json,"r",encoding="utf-8") as f:
        ont = json.load(f)
    # flatten alias list per aspect
    items = []
    for asp, aliases in ont.items():
        aliases = list(set([normalize_text(a) for a in ([asp] + aliases)]))
        items.append((asp, aliases))
    return items

def embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return vecs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans_parquet", required=True)
    ap.add_argument("--ontology_json", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()

    spans = pd.read_parquet(args.spans_parquet)
    if spans.empty:
        spans.assign(aspect_norm=None, score=np.nan).to_parquet(args.output_parquet, index=False)
        print(f"[saved-empty] {args.output_parquet}")
        return

    spans["span_norm"] = spans["aspect_span"].astype(str).map(normalize_text)

    ontology = build_ontology(args.ontology_json)
    # Build alias catalog
    alias_texts, alias_map = [], []
    for asp, aliases in ontology:
        for al in aliases:
            alias_texts.append(al); alias_map.append(asp)

    # Embed
    span_vecs = embed_texts(spans["span_norm"].tolist())
    alias_vecs = embed_texts(alias_texts)

    # Cosine match (embeddings are normalized)
    scores = np.dot(span_vecs, alias_vecs.T)  # (N_spans, N_aliases)
    best_idx = scores.argmax(axis=1)
    best_score = scores.max(axis=1)
    best_aspect = [alias_map[i] for i in best_idx]

    spans["aspect_norm"] = best_aspect
    spans["score"] = best_score

    # threshold
    spans.loc[spans["score"] < args.threshold, "aspect_norm"] = None

    out = spans.rename(columns={"aspect_span":"span_text","prob":"prob_span"})[
        ["review_id","span_text","aspect_norm","score","prob_span"]
    ]
    out.to_parquet(args.output_parquet, index=False)
    print(f"[saved] {args.output_parquet} | rows={len(out)} | mapped_ratio={(out['aspect_norm'].notnull().mean()*100):.1f}%")

if __name__ == "__main__":
    main()
