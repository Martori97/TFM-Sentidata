#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import pandas as pd
import numpy as np

def parse_args():
  p = argparse.ArgumentParser("Top brand por category×aspect (mediana)")
  p.add_argument("--input_parquet", required=True)
  p.add_argument("--output_csv", default="reports/viz/top_brand_per_category_aspect.csv")
  p.add_argument("--metric", choices=["rating","pred3","pos_minus_neg"], default="rating")
  p.add_argument("--min_n_per_cell", type=int, default=50)
  p.add_argument("--min_categories", type=int, default=1, help="Solo considerar rows con category non-null")
  return p.parse_args()

def build_metric(df, metric):
  if metric == "rating":
    return df["rating"]
  if metric == "pred3":
    return df["pred_3"].map({-1:-1, 0:0, 1:1, 2:1})
  if metric == "pos_minus_neg":
    return df["p_pos"] - df["p_neg"]
  raise ValueError("metric no soportada")

def main():
  a = parse_args()
  os.makedirs(os.path.dirname(a.output_csv), exist_ok=True)

  usecols = ["brand","category","aspect_norm","rating","pred_3","p_neg","p_pos"]
  df = pd.read_parquet(a.input_parquet, columns=usecols)
  df = df.dropna(subset=["brand","aspect_norm"])
  if a.min_categories:
    df = df.dropna(subset=["category"])

  df["metric_value"] = build_metric(df, a.metric)
  df = df.dropna(subset=["metric_value"])

  # tamaño de celda
  grp = df.groupby(["category","aspect_norm","brand"]).size().rename("n").reset_index()
  grp = grp[grp["n"] >= a.min_n_per_cell]

  # mediana por celda válida
  med = (
      df.merge(grp[["category","aspect_norm","brand"]], on=["category","aspect_norm","brand"], how="inner")
        .groupby(["category","aspect_norm","brand"])["metric_value"].median().reset_index(name="median_metric")
  )

  # ganador por (category, aspect)
  idx = med.groupby(["category","aspect_norm"])["median_metric"].idxmax()
  winners = med.loc[idx].sort_values(["category","aspect_norm"]).reset_index(drop=True)

  # puedes añadir n por transparencia
  winners = winners.merge(grp, on=["category","aspect_norm","brand"], how="left")

  winners.to_csv(a.output_csv, index=False)
  print("[OK] Guardado", a.output_csv)

if __name__ == "__main__":
  main()
