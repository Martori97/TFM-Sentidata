#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

def rating_to_label3_int(r:int)->int: return 0 if r<=2 else (1 if r==3 else 2)

p=argparse.ArgumentParser()
p.add_argument("--input", required=True)   # carpeta o parquet
p.add_argument("--output", required=True)  # p.ej. models/albert_subset_0_250/test.parquet
p.add_argument("--text_col", default="review_text")
p.add_argument("--rating_col", default="rating")
p.add_argument("--id_col", default="review_id")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--test_size", type=float, default=0.15)
a=p.parse_args()

# cargar
if os.path.isdir(a.input):
    parts=[os.path.join(a.input,f) for f in os.listdir(a.input) if f.endswith(".parquet")]
    assert parts, f"Sin parquet en {a.input}"
    df=pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
else:
    df=pd.read_parquet(a.input)

# columnas y label3
for c in [a.text_col, a.rating_col]: assert c in df.columns, f"Falta {c}"
if a.id_col not in df.columns:
    df[a.id_col]=pd.util.hash_pandas_object(df[a.text_col].astype(str), index=False).astype(str)

df=df.dropna(subset=[a.text_col,a.rating_col]).copy()
df[a.rating_col]=df[a.rating_col].astype(int).clip(1,5)
df["label3"]=df[a.rating_col].apply(rating_to_label3_int).astype(int)

# split estratificado (solo test)
idx=np.arange(len(df))
_, idx_te = train_test_split(idx, test_size=a.test_size, random_state=a.seed, stratify=df["label3"].values)
os.makedirs(os.path.dirname(a.output), exist_ok=True)
df.iloc[idx_te].to_parquet(a.output, index=False)
print(f"✅ test guardado en {a.output} (filas={len(idx_te)})")
