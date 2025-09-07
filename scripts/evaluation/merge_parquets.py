#!/usr/bin/env python3
import argparse, os, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("inputs", nargs="+")
a = p.parse_args()

os.makedirs(os.path.dirname(a.out), exist_ok=True)
df = pd.concat([pd.read_parquet(f) for f in a.inputs], ignore_index=True)
df.to_parquet(a.out, index=False)
print("OK", a.out, "n_files=", len(a.inputs), "rows=", len(df))
