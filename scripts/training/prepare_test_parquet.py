#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def rating_to_label3_int(r: int) -> int:
    r = int(r)
    if r <= 2: return 0
    if r == 3: return 1
    return 2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Carpeta o fichero parquet de entrada")
    # Compatibilidad: acepta --output y --outdir (alias)
    p.add_argument("--output", required=False, help="Ruta de salida (directorio raíz con subcarpetas train/ val/ test)")
    p.add_argument("--outdir", required=False, help="Alias de --output")
    p.add_argument("--text_col",   default="review_text")
    p.add_argument("--rating_col", default="rating")
    p.add_argument("--id_col",     default="review_id")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--test_size",  type=float, default=0.15, help="Proporción para test (0-1)")
    p.add_argument("--val_size",   type=float, default=0.10, help="Proporción para val (0-1, sobre el total)")

    a = p.parse_args()
    if not a.output and not a.outdir:
        p.error("Debes pasar --output o --outdir")

    # prioridad a --output si vienen ambos
    a.output = a.output if a.output else a.outdir

    # queremos SIEMPRE un directorio raíz que contenga train/ val/ test
    # si el usuario pasó un .parquet por error, usamos su carpeta padre
    if a.output.lower().endswith(".parquet"):
        a.output = os.path.dirname(a.output) or "."

    return a

def load_parquet_folder_or_file(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        parts = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
        if not parts:
            raise SystemExit(f"Sin ficheros .parquet en {path}")
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return pd.read_parquet(path)

def main():
    a = parse_args()

    # cargar
    df = load_parquet_folder_or_file(a.input)

    # checks mínimos
    for c in [a.text_col, a.rating_col]:
        if c not in df.columns:
            raise SystemExit(f"Falta la columna requerida '{c}' en el dataset. Columnas: {sorted(df.columns)}")

    # id si no existe
    if a.id_col not in df.columns:
        df[a.id_col] = pd.util.hash_pandas_object(
            df[a.text_col].astype(str), index=False
        ).astype(str)

    # limpieza básica
    df = df.dropna(subset=[a.text_col, a.rating_col]).copy()
    df[a.rating_col] = df[a.rating_col].astype(int).clip(1, 5)

    # etiquetas
    df["label3"] = df[a.rating_col].apply(rating_to_label3_int).astype(int)
    # útil para inspecciones/EDA o si algún script espera texto
    df["label"] = df["label3"].map({0: "negative", 1: "neutral", 2: "positive"})

    # índices y estratificación
    idx = np.arange(len(df))
    test_size = float(a.test_size)
    val_size_total = float(a.val_size)  # fracción sobre total

    if not (0.0 < test_size < 1.0) or not (0.0 < val_size_total < 1.0) or (test_size + val_size_total >= 1.0):
        raise SystemExit("Parámetros inválidos: asegúrate de que 0<test_size<1, 0<val_size<1 y (test+val)<1.")

    # split test
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=a.seed,
        stratify=df["label3"].values
    )

    # split val dentro de train+val (val relativo a remanente)
    rel_val = val_size_total / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=rel_val,
        random_state=a.seed,
        stratify=df["label3"].values[idx_trainval]
    )

    # preparar salidas
    out_root = a.output
    out_train = os.path.join(out_root, "train")
    out_val   = os.path.join(out_root, "val")
    out_test  = os.path.join(out_root, "test")
    for d in [out_train, out_val, out_test]:
        os.makedirs(d, exist_ok=True)

    # escribir (un archivo por split; si prefieres, cambia a múltiples)
    df.iloc[idx_train].to_parquet(os.path.join(out_train, "data.parquet"), index=False)
    df.iloc[idx_val  ].to_parquet(os.path.join(out_val,   "data.parquet"), index=False)
    df.iloc[idx_test ].to_parquet(os.path.join(out_test,  "data.parquet"), index=False)

    print(f"✅ splits guardados en {out_root}")
    print(f"   train: {len(idx_train)} | val: {len(idx_val)} | test: {len(idx_test)}")

if __name__ == "__main__":
    main()