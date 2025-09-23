#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Reutilizamos del train
from train_random_forest import TextCleaner, ExtraFeatures, read_parquet_folder_or_file

MAP_ID_TO_NAME = {0:"negative",1:"neutral",2:"positive"}

def safe_mkdir(p): Path(p).mkdir(parents=True, exist_ok=True)

def run_infer_df(df: pd.DataFrame, text_col: str, cleaner, vectorizer, add_extra=True):
    Xclean = cleaner.transform(df[text_col])
    Xtfidf = vectorizer.transform(Xclean)
    if add_extra:
        extra = ExtraFeatures()
        Xextra = extra.transform(Xclean)
        from scipy.sparse import hstack
        X = hstack([Xtfidf, Xextra], format="csr")
    else:
        X = Xtfidf
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True,
                    help="(a) Carpeta con subcarpetas parquet (trusted partitions) o "
                         "(b) carpeta de split (test) o (c) parquet único")
    ap.add_argument("--output-dir", required=True, help="Carpeta base de salida de predicciones")
    ap.add_argument("--text-col", default="review_text")
    ap.add_argument("--id-col", default="review_id")
    ap.add_argument("--derive-true-from", default="rating",
                    help="Si true label no existe: deriva desde 'rating'→{0,1,2}")
    ap.add_argument("--use-extra", type=int, default=1, help="Añadir features extra (1/0)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_base = Path(args.output_dir)
    safe_mkdir(out_base)

    # cargar artefactos
    rf = joblib.load(model_dir / "rf_model.joblib")
    vec = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    pre_cfg = json.loads((model_dir / "preprocess_config.json").read_text())
    cleaner = TextCleaner(use_lemma=pre_cfg.get("use_lemma", True),
                          lang=pre_cfg.get("language", "en")).fit([])

    # ¿input es un directorio con subcarpetas (particiones trusted)?
    in_path = Path(args.input)
    outputs = []

    def add_truth(df: pd.DataFrame):
        # intenta label/label3; si no, deriva desde rating
        if "label" in df.columns:
            s = df["label"].astype(str).str.lower().map({"negative":0,"neg":0,"neutral":1,"neu":1,"positive":2,"pos":2})
            if s.notna().all(): return s.astype("int32").to_numpy()
        if "label3" in df.columns:
            return df["label3"].astype("int32").to_numpy()
        if args.derive_true_from in df.columns:
            r = pd.to_numeric(df[args.derive_true_from], errors="coerce").round().clip(1,5).astype("Int64")
            return r.map({1:0,2:0,3:1,4:2,5:2}).astype("int32").to_numpy()
        return None

    if in_path.is_dir():
        # caso A: particiones (subcarpetas o datasets separados)
        subdirs = sorted([p for p in in_path.iterdir() if p.is_dir()])
        if not subdirs:
            # dataset único en carpeta (split test)
            df = read_parquet_folder_or_file(str(in_path))
            X = run_infer_df(df, args.text_col, cleaner, vec, add_extra=bool(args.use_extra))
            preds = rf.predict(X)
            out_df = pd.DataFrame({
                args.id_col: df[args.id_col].values if args.id_col in df.columns else np.arange(len(df)),
                "pred_index": preds.astype("int32"),
                "pred_label": [MAP_ID_TO_NAME[int(i)] for i in preds],
            })
            y_true = add_truth(df)
            if y_true is not None:
                out_df["true_label"] = y_true
            out_path = out_base / "test_predictions.parquet"
            out_df.to_parquet(out_path, index=False)
            outputs.append(str(out_path))
        else:
            # iterar por subcarpetas
            part_out_dir = out_base / "partitions"
            part_out_dir.mkdir(parents=True, exist_ok=True)
            for d in subdirs:
                try:
                    df = read_parquet_folder_or_file(str(d))
                except Exception:
                    continue
                X = run_infer_df(df, args.text_col, cleaner, vec, add_extra=bool(args.use_extra))
                preds = rf.predict(X)
                out_df = pd.DataFrame({
                    args.id_col: df[args.id_col].values if args.id_col in df.columns else np.arange(len(df)),
                    "pred_index": preds.astype("int32"),
                    "pred_label": [MAP_ID_TO_NAME[int(i)] for i in preds],
                })
                y_true = add_truth(df)
                if y_true is not None:
                    out_df["true_label"] = y_true
                out_path = part_out_dir / f"{d.name}_preds.parquet"
                out_df.to_parquet(out_path, index=False)
                outputs.append(str(out_path))
    else:
        # parquet único
        df = pd.read_parquet(in_path)
        X = run_infer_df(df, args.text_col, cleaner, vec, add_extra=bool(args.use_extra))
        preds = rf.predict(X)
        out_df = pd.DataFrame({
            args.id_col: df[args.id_col].values if args.id_col in df.columns else np.arange(len(df)),
            "pred_index": preds.astype("int32"),
            "pred_label": [MAP_ID_TO_NAME[int(i)] for i in preds],
        })
        y_true = add_truth(df)
        if y_true is not None:
            out_df["true_label"] = y_true
        out_path = out_base / "predictions.parquet"
        out_df.to_parquet(out_path, index=False)
        outputs.append(str(out_path))

    print("Inferencia RF OK")
    for p in outputs:
        print("  ->", p)

if __name__ == "__main__":
    main()
