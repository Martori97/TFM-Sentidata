#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--preds", required=True, help="reports/albert_subset_all/predictions_all.parquet")
p.add_argument("--out_dir", required=True)
a = p.parse_args()

df = pd.read_parquet(a.preds)
if "true_label" not in df.columns or "pred_index" not in df.columns:
    raise SystemExit("Faltan columnas true_label/pred_index en el parquet de entrada.")

y = df["true_label"].astype("int32").to_numpy()
yhat = df["pred_index"].astype("int32").to_numpy()

metrics = {
    "n": int(len(df)),
    "accuracy": float(accuracy_score(y, yhat)),
    "f1_macro": float(f1_score(y, yhat, average="macro")),
}

os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, "metrics_overall.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ---------- Matriz de confusión (absoluta) ----------
labels = [0, 1, 2]
cm = confusion_matrix(y, yhat, labels=labels)

plt.figure(figsize=(5.2, 4.6))
plt.imshow(cm, interpolation="nearest")
plt.xticks(range(3), ["neg", "neu", "pos"])
plt.yticks(range(3), ["neg", "neu", "pos"])
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.title("Confusion Matrix (all)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(a.out_dir, "confusion_all.png"), dpi=140)
plt.close()

# ---------- Matriz de confusión normalizada por fila (%) ----------
cm = cm.astype(float)
row = cm.sum(axis=1, keepdims=True)
row[row == 0] = 1.0  # evita división por cero
cmn = (cm / row) * 100.0

plt.figure(figsize=(5.2, 4.6))
plt.imshow(cmn, interpolation="nearest")
plt.xticks(range(3), ["neg", "neu", "pos"])
plt.yticks(range(3), ["neg", "neu", "pos"])
for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{cmn[i, j]:.1f}%", ha="center", va="center")
plt.title("Confusion Matrix (all, % row)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(a.out_dir, "confusion_all_norm.png"), dpi=140)
plt.close()

print("✅ overall ->", metrics)
