#!/usr/bin/env python3
import os, shutil, yaml, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()

    # cargar config
    with open(args.params, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["absa_report"]

    CARDS = paths["cards"]
    INSIGHTS = paths["insights"]
    LOVES_GLOBAL = paths["loves_global"]
    LOVES_PERCAT = paths["loves_percat"]
    FINAL = paths["outdir"]

    os.makedirs(FINAL, exist_ok=True)

    # copia cards
    shutil.copy2(CARDS, os.path.join(FINAL, "product_cards.jsonl"))

    # copia insights (carpeta entera)
    dst_insights = os.path.join(FINAL, "insights_charts")
    if os.path.exists(dst_insights):
        shutil.rmtree(dst_insights)
    shutil.copytree(INSIGHTS, dst_insights)

    # copia loves
    shutil.copy2(LOVES_GLOBAL, os.path.join(FINAL, "global_top10.csv"))
    shutil.copy2(LOVES_PERCAT, os.path.join(FINAL, "per_category_top10.csv"))

    print(f"[ok] Final report generado en {FINAL}")

if __name__ == "__main__":
    main()

