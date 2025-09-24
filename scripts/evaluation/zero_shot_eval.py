#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse, datetime
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ---------- Utilidades ----------
def load_params():
    # intenta detectar raíz del repo con git; si falla, usa cwd
    try:
        import subprocess
        repo_root = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"]
        ).decode().strip())
    except Exception:
        repo_root = Path.cwd()

    pfile = repo_root / "params.yaml"
    if not pfile.exists():
        raise FileNotFoundError(f"No se encontró params.yaml en {repo_root}")
    with open(pfile, "r") as f:
        return yaml.safe_load(f), repo_root

def load_dataframe_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe la ruta: {path}")

    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Sin .parquet dentro de {path}")
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path)

    # intento genérico
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if isinstance(c, str) and c in df.columns:
            return c
    return None

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="GPU id (usa -1 para CPU)")
    args = parser.parse_args()

    params, repo_root = load_params()
    eda = params.get("eda", {})
    zsp = params.get("zero_shot", {})

    data_path = eda.get("data_path", "data/trusted/sephora_clean/reviews_0_250")
    text_column = eda.get("text_column", "review_text")
    star_column = eda.get("star_column", "rating")
    label_column = eda.get("label_column", None)
    star_to_label = eda.get("star_to_label", {1:"NEG",2:"NEG",3:"NEU",4:"POS",5:"POS"})

    model_name = zsp.get("model_name", "joeddav/xlm-roberta-large-xnli")
    batch_size = int(zsp.get("batch_size", 16))
    sample_size = zsp.get("sample_size", 2000)
    label_set = zsp.get("label_set", ["positivo","neutro","negativo"])
    map_to_internal = zsp.get("map_to_internal", {"positivo":"POS","neutro":"NEU","negativo":"NEG"})

    # rutas
    out_dir = repo_root / "reports" / "zero_shot" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # cargar datos
    data_path_abs = (repo_root / data_path).resolve()
    df = load_dataframe_any(data_path_abs)
    # elegir columnas
    TEXT_COL = pick_col(df, [text_column, "review_text", "text", "review", "content", "body"])
    if TEXT_COL is None:
        raise ValueError(f"No se encontró columna de texto. Disponibles: {list(df.columns)[:40]}")

    # determinar etiquetas verdaderas
    LABEL_COL = None
    if label_column and label_column in df.columns:
        LABEL_COL = label_column
    else:
        STAR_COL = pick_col(df, [star_column, "rating", "stars", "score", "overall"])
        if STAR_COL is None:
            raise ValueError("No hay columna de etiqueta ni de estrellas para derivar.")
        stars = pd.to_numeric(df[STAR_COL], errors="coerce")
        if stars.max() <= 1.0:
            stars = stars * 5
        stars = stars.round().clip(1,5).astype("Int64")
        df["sentiment_3c"] = stars.map(star_to_label)
        LABEL_COL = "sentiment_3c"

    # drop NaNs y tipado
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)

    # muestreo opcional
    if sample_size and int(sample_size) > 0 and len(df) > sample_size:
        df = df.sample(int(sample_size), random_state=42).reset_index(drop=True)

    print(f"[Zero-Shot] Registros a evaluar: {len(df)} | Texto: {TEXT_COL} | Label: {LABEL_COL}")

    # pipeline zero-shot
    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=args.device,       # -1 CPU, 0 primera GPU
        truncation=True
    )

    # predicción en lotes
    texts = df[TEXT_COL].tolist()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        res = clf(batch, candidate_labels=label_set, multi_label=False)
        # res es dict o lista de dicts según batch size
        if isinstance(res, dict):
            res = [res]
        labels_top = [r["labels"][0] for r in res]
        preds.extend(labels_top)

        if (i // batch_size) % 20 == 0:
            print(f"  Proceso {i+len(batch)}/{len(texts)}")

    df["pred_zero_shot_label"] = preds
    # mapear a etiquetas internas
    df["pred_zero_shot_internal"] = df["pred_zero_shot_label"].map(map_to_internal)

    # normalizar etiquetas verdaderas a interno {NEG,NEU,POS} o {0,1,2}
    true_vals = df[LABEL_COL]
    # intenta detectar strings vs enteros
    if true_vals.dtype.name == "category" or true_vals.dtype == object:
        # normaliza variantes
        norm_map = {
            "NEG":"NEG","NEU":"NEU","POS":"POS",
            "neg":"NEG","neu":"NEU","pos":"POS",
            "negativo":"NEG","neutro":"NEU","positivo":"POS",
            "negative":"NEG","neutral":"NEU","positive":"POS"
        }
        y_true = true_vals.astype(str).str.lower().map(norm_map).fillna(true_vals).tolist()
    else:
        # asume 0/1/2 → mapea a strings para comparar (ajusta si tu codificación es distinta)
        m = {0:"NEG",1:"NEU",2:"POS"}
        y_true = [m.get(int(v), str(v)) for v in true_vals]

    y_pred = df["pred_zero_shot_internal"].tolist()

    # Métricas
    labels_order = ["NEG","NEU","POS"]  # orden consistente
    rep = classification_report(y_true, y_pred, labels=labels_order, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    # Guardar resultados
    (out_dir / "predicciones_zero_shot.parquet").write_bytes(df.to_parquet(index=False))
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    np.savetxt(out_dir / "confusion_matrix.txt", cm, fmt="%d")

    # Print resumen
    print("\n== Classification Report (Zero-Shot) ==")
    print(pd.DataFrame(rep).T.loc[["NEG","NEU","POS","accuracy","macro avg","weighted avg"]])
    print("\n== Confusion Matrix (rows=true, cols=pred) [NEG,NEU,POS] ==")
    print(cm)
    print(f"\nArchivos guardados en: {out_dir}")

if __name__ == "__main__":
    main()
