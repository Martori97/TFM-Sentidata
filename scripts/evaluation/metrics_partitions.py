#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

p = argparse.ArgumentParser()
p.add_argument("--inputs_dir", required=True)
p.add_argument("--pattern", default="*_preds.parquet")
p.add_argument("--out_csv", required=True)
a = p.parse_args()

files = sorted(glob.glob(os.path.join(a.inputs_dir, a.pattern)))
if not files:
    raise SystemExit(f"No se encontraron ficheros con patrón {a.pattern} en {a.inputs_dir}")

rows = []
for f in files:
    df = pd.read_parquet(f)
    if not {"pred_index"}.issubset(df.columns):
        print("Saltando (falta pred_index):", f)
        continue

    # Si no hay true_label, no podemos medir esa partición (guardamos NaN)
    if "true_label" not in df.columns:
        rows.append({
            "split": os.path.basename(f).replace("_preds.parquet", ""),
            "n": int(len(df)),
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "has_true_label": False,
        })
        continue

    y = df["true_label"].astype("int32").to_numpy()
    yhat = df["pred_index"].astype("int32").to_numpy()

    rows.append({
        "split": os.path.basename(f).replace("_preds.parquet", ""),
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, yhat)),
        "f1_macro": float(f1_score(y, yhat, average="macro")),
        "has_true_label": True,
    })

out = pd.DataFrame(rows).sort_values("split")
os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
out.to_csv(a.out_csv, index=False)
print("✅ guardado", a.out_csv)
