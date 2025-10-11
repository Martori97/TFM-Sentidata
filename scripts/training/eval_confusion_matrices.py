#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evalúa matrices de confusión y classification report para:
- ALBERT (usa true_label y pred_3)  [evaluate_albert.py]
- RandomForest (usa true_label/pred_label o pred_index) [infer_random_forest.py]
- SVM (usa prediction; labels del propio preds o del test/rating)

Guarda resultados en: reports/eval/confusion_matrices/
"""

import os, glob
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

LABELS = ["negative", "neutral", "positive"]
IDX2LAB = {0:"negative", 1:"neutral", 2:"positive"}

# ---------- utilidades de IO ----------
def _read_many(paths):
    dfs = []
    for p in paths:
        if p.endswith(".parquet"):
            dfs.append(pd.read_parquet(p))
        elif p.endswith(".csv"):
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError("No se encontraron .parquet/.csv en el directorio especificado.")
    return pd.concat(dfs, ignore_index=True)

def load_any(path_str: str) -> pd.DataFrame:
    """Lee un fichero .parquet/.csv o concatena todos los .parquet/.csv de una carpeta."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    if p.is_dir():
        files = sorted(glob.glob(str(p / "**" / "*.parquet"), recursive=True)) + \
                sorted(glob.glob(str(p / "**" / "*.csv"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No hay .parquet/.csv en {p}")
        return _read_many(files)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Formato no soportado: {p.suffix} -> {p}")

# ---------- helpers de etiquetas ----------
def derive_label_from_rating(series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").round().clip(1, 5).astype("Int64")
    return s.map({1:"negative",2:"negative",3:"neutral",4:"positive",5:"positive"})

def coerce_to_label_str(col: pd.Series) -> pd.Series:
    """Acepta 0/1/2 o strings variados y los normaliza a LABELS."""
    s = col.copy()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int).map(IDX2LAB)
    s = s.astype(str).str.strip().str.lower()
    mapping = {
        "0":"negative","neg":"negative","negative":"negative",
        "1":"neutral","neu":"neutral","neutral":"neutral",
        "2":"positive","pos":"positive","positive":"positive"
    }
    return s.map(mapping)

# ---------- mapeos por modelo ----------
def extract_truth_pred_for_albert(preds: pd.DataFrame, test_df: pd.DataFrame|None):
    """
    evaluate_albert.py guarda:
      - true_label (string)
      - pred_3     (string)
    """
    if "true_label" not in preds.columns or "pred_3" not in preds.columns:
        raise ValueError("ALBERT: faltan columnas 'true_label' y/o 'pred_3'.")
    y_true = coerce_to_label_str(preds["true_label"])
    y_pred = coerce_to_label_str(preds["pred_3"])
    return y_true, y_pred

def extract_truth_pred_for_rf(preds: pd.DataFrame, test_df: pd.DataFrame|None):
    """
    infer_random_forest.py puede guardar:
      - pred_label (string) y opcionalmente true_label (0/1/2)
      - si no hay pred_label, usa pred_index (0/1/2)
      - si no hay true_label, intenta derivarlo del test o rating
    """
    # y_pred
    if "pred_label" in preds.columns:
        y_pred = coerce_to_label_str(preds["pred_label"])
    elif "pred_index" in preds.columns:
        y_pred = coerce_to_label_str(preds["pred_index"])
    else:
        raise ValueError("RF: faltan 'pred_label' y 'pred_index'.")

    # y_true
    if "true_label" in preds.columns:
        y_true = coerce_to_label_str(preds["true_label"])
    else:
        # intentar desde test_df por review_id
        if test_df is None:
            raise ValueError("RF: no hay 'true_label' en preds ni test_df para derivarlo.")
        td = test_df.copy()
        if "sentiment_category" not in td.columns:
            if "rating" in td.columns:
                td["sentiment_category"] = derive_label_from_rating(td["rating"])
            else:
                raise ValueError("RF: test_df no trae 'sentiment_category' ni 'rating'.")
        # join por id si es posible
        left_id = next((c for c in ["review_id","id","review_id_unico"] if c in preds.columns), None)
        right_id = next((c for c in ["review_id","id","review_id_unico"] if c in td.columns), None)
        if left_id and right_id:
            merged = preds[[left_id]].merge(
                td[[right_id,"sentiment_category"]].drop_duplicates(subset=[right_id]),
                left_on=left_id, right_on=right_id, how="left"
            )
            y_true = coerce_to_label_str(merged["sentiment_category"])
        else:
            # fallback por índice si longitudes coinciden
            if len(preds) != len(td):
                raise ValueError("RF: no hay id común y longitudes difieren; no puedo alinear etiquetas.")
            y_true = coerce_to_label_str(td["sentiment_category"].reset_index(drop=True))
    return y_true, y_pred

def extract_truth_pred_for_svm(preds: pd.DataFrame, test_df: pd.DataFrame|None):
    """
    infer_svm.py guarda:
      - prediction (string)
      - opcionalmente sentiment_category o rating
      - si no hay etiqueta, se deriva del test_df o de rating
    """
    if "prediction" not in preds.columns:
        raise ValueError("SVM: falta columna 'prediction'.")

    y_pred = coerce_to_label_str(preds["prediction"])

    if "sentiment_category" in preds.columns:
        y_true = coerce_to_label_str(preds["sentiment_category"])
        return y_true, y_pred
    if "rating" in preds.columns:
        y_true = derive_label_from_rating(preds["rating"])
        return y_true, y_pred

    if test_df is None:
        raise ValueError("SVM: no hay etiquetas en preds y no se proporcionó test_df.")

    td = test_df.copy()
    if "sentiment_category" not in td.columns:
        if "rating" in td.columns:
            td["sentiment_category"] = derive_label_from_rating(td["rating"])
        else:
            raise ValueError("SVM: test_df no trae 'sentiment_category' ni 'rating'.")

    left_id = next((c for c in ["review_id","id","review_id_unico"] if c in preds.columns), None)
    right_id = next((c for c in ["review_id","id","review_id_unico"] if c in td.columns), None)
    if left_id and right_id:
        merged = preds[[left_id]].merge(
            td[[right_id,"sentiment_category"]].drop_duplicates(subset=[right_id]),
            left_on=left_id, right_on=right_id, how="left"
        )
        y_true = coerce_to_label_str(merged["sentiment_category"])
        return y_true, y_pred

    if len(preds) == len(td):
        y_true = coerce_to_label_str(td["sentiment_category"].reset_index(drop=True))
        return y_true, y_pred

    raise ValueError("SVM: no hay forma de obtener etiquetas (ni en preds, ni test df alineable).")

# ---------- diccionario de modelos ----------
MODELS = {
    "ALBERT": {
        "pred": "reports/albert_sample_30pct/eval/predictions.parquet",  # ✅ cambio #1 aplicado
        "test": None,  # ya trae true_label y pred_3
        "extract": extract_truth_pred_for_albert,
    },
    "RandomForest": {
        "pred": "reports/sentiment_rf_sample/eval/test_predictions.parquet",
        "test": "data/exploitation/modelos_input/sample_tvt/test",
        "extract": extract_truth_pred_for_rf,
    },
    "SVM": {
        "pred": "reports/sentiment_svm_sample/eval/test_predictions.parquet",
        "test": "data/exploitation/modelos_input/sample_tvt/test",
        "extract": extract_truth_pred_for_svm,
    }
}

# ---------- evaluación ----------
def evaluate_and_save(name: str, pred_path: str, test_path: str|None, extractor, out_dir: Path):
    if not Path(pred_path).exists():
        print(f"⚠️ {name}: no existe {pred_path}. Me lo salto.")
        return

    preds = load_any(pred_path)
    test_df = load_any(test_path) if test_path else None

    y_true, y_pred = extractor(preds, test_df)

    if y_true.isna().any():
        missing = int(y_true.isna().sum())
        print(f"⚠️ {name}: {missing} etiquetas nulas en y_true (se excluirán en el reporte).")
        keep = ~y_true.isna()
        y_true = y_true[keep]
        y_pred = y_pred[keep]

    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(out_dir / f"{name.lower()}_confusion_matrix.csv")
    with open(out_dir / f"{name.lower()}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ {name}: OK → {pred_path}")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    print(report)

def main():
    out_dir = Path("reports/eval/confusion_matrices")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in MODELS.items():
        evaluate_and_save(name, spec["pred"], spec["test"], spec["extract"], out_dir)

    print(f"\n✅ Resultados guardados en: {out_dir.resolve()}")

if __name__ == "__main__":
    main()


