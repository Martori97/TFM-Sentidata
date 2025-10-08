#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cards-in", required=True)
    p.add_argument("--stats-parquet", required=True)
    p.add_argument("--cards-out", required=True)
    p.add_argument("--id-col", default="product_id")
    return p.parse_args()

def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.cards_out), exist_ok=True)
    stats = pd.read_parquet(a.stats_parquet).set_index(a.id_col)

    with open(a.cards_in, "r", encoding="utf-8") as fin, \
         open(a.cards_out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): 
                continue
            o = json.loads(line)
            pid = o.get(a.id_col)
            if pid in stats.index:
                s = stats.loc[pid]
                o["rating"] = float(s["rating_avg_2d"]) if pd.notna(s["rating_avg_2d"]) else None
                o["ratings_count"] = int(s["ratings_count"]) if pd.notna(s["ratings_count"]) else 0
                o["reviews_count"] = int(s["reviews_count"]) if pd.notna(s["reviews_count"]) else 0
                o["rating_source"] = "recomputed_from_reviews"
            else:
                o["rating_source"] = "product_info_fallback"
            fout.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[ok] cards recalc -> {a.cards_out}")

if __name__ == "__main__":
    main()
