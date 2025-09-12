# -*- coding: utf-8 -*-
# Recorre Sephora Delta y limpia cada tabla -> Trusted

import os, subprocess, sys

SEPH_DELTA = "data/landing/sephora/delta"
SEPH_TRUST = "data/trusted/sephora_clean"

os.makedirs(SEPH_TRUST, exist_ok=True)

def _is_delta_table(path: str) -> bool:
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, "_delta_log"))

def run():
    for name in sorted(os.listdir(SEPH_DELTA)):
        in_path = os.path.join(SEPH_DELTA, name)
        out_path = os.path.join(SEPH_TRUST, name)  # mismo nombre
        if not _is_delta_table(in_path):
            print(f"[SKIP] {in_path} no es tabla Delta válida")
            continue
        print(f"[RUN] Limpieza Sephora: {name}")
        subprocess.run(
            ["python", "scripts/data_cleaning/trusted_clean_single.py", in_path, out_path],
            check=False,
        )

if __name__ == "__main__":
    run()
