#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, re, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.validation import check_is_fitted

from scipy.sparse import hstack, csr_matrix

# ===== Utilidades de IO =====
def read_parquet_folder_or_file(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if p.is_dir():
        files = sorted(p.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No se encontraron .parquet en {p}")
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.read_parquet(p)

# ===== Limpieza de texto =====
class TextCleaner:
    def __init__(self, use_lemma: bool = True, lang: str = "en"):
        self.use_lemma = bool(use_lemma)
        self.lang = lang
        self._spacy = None  # lazy

    def _ensure_spacy(self):
        if self._spacy is not None:
            return
        if not self.use_lemma:
            return
        try:
            import spacy
            model = {"en": "en_core_web_sm", "es": "es_core_news_sm"}.get(self.lang, "en_core_web_sm")
            try:
                self._spacy = spacy.load(model)
            except Exception:
                warnings.warn(f"[TextCleaner] spaCy model {model} no disponible; sigue sin lematizar.")
                self._spacy = None
        except Exception:
            warnings.warn("[TextCleaner] spaCy no instalado; sigue sin lematizar.")
            self._spacy = None

    def fit(self, X):
        # stateless
        return self

    def _basic_clean(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"http\S+|www\.\S+", " ", s)
        s = re.sub(r"[^a-záéíóúñüç0-9\s]", " ", s)
        s = re.sub(r"\d+", " ", s)  # quitar números
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _lemmatize_list(self, toks):
        if not toks:
            return toks
        self._ensure_spacy()
        if self._spacy is None:
            return toks
        doc = self._spacy(" ".join(toks))
        return [t.lemma_ for t in doc if not t.is_space]

    def transform(self, texts):
        out = []
        for t in texts.astype(str).tolist():
            c = self._basic_clean(t)
            toks = c.split()
            if self.use_lemma:
                toks = self._lemmatize_list(toks)
            out.append(" ".join(toks))
        return np.array(out, dtype=object)

# ===== Features extra sencillas =====
_NEG_LEXICON = {"bad","terrible","awful","worse","worst","horrible","disappointed","poor","hate","lousy",
                "malo","terrible","horrible","peor","decepcionado","pobre","odiar","fatal"}

_CONTRAST_MARKERS = {"but","however","though","although","yet","aunque","pero","sin embargo"}

class ExtraFeatures:
    """Devuelve una matriz CSR con columnas:
       [neg_lex_count, contrast_markers_count, token_len]
    """
    def __init__(self, use_neg=True, use_contrast=True, use_len=True):
        self.use_neg = use_neg
        self.use_contrast = use_contrast
        self.use_len = use_len

    def transform(self, cleaned_texts):
        rows = []
        for s in cleaned_texts:
            toks = s.split()
            neg = sum(1 for w in toks if w in _NEG_LEXICON) if self.use_neg else 0
            cm  = sum(1 for w in toks if w in _CONTRAST_MARKERS) if self.use_contrast else 0
            ln  = len(toks) if self.use_len else 0
            rows.append([neg, cm, ln])
        arr = np.asarray(rows, dtype=np.float32)
        return csr_matrix(arr)

# ===== Helpers label =====
def derive_label_from_rating(series) -> np.ndarray:
    r = pd.to_numeric(series, errors="coerce").round().clip(1, 5).astype("Int64")
    # 1-2 -> 0 (NEG), 3 -> 1 (NEU), 4-5 -> 2 (POS)
    return r.map({1:0, 2:0, 3:1, 4:2, 5:2}).astype("int32").to_numpy()

def to_labels(df, label_col, rating_col) -> np.ndarray:
    if label_col and label_col in df.columns:
        s = df[label_col].astype(str).str.lower()
        mapper = {"negative":0,"neg":0,"0":0,"neutral":1,"neu":1,"1":1,"positive":2,"pos":2,"2":2}
        y = s.map(mapper)
        if y.notna().all():
            return y.astype("int32").to_numpy()
    if rating_col in df.columns:
        return derive_label_from_rating(df[rating_col])
    raise ValueError("No se pudo construir y: falta label_col y rating_col")

# ===== MLflow opcional =====
def mlflow_start_run_if_enabled(cfg):
    if not cfg.get("mlflow", {}).get("enable", False):
        return None, lambda *a, **k: None, lambda *a, **k: None, lambda *a, **k: None
    import mlflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    run = mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "rf_train"))
    return run, mlflow.log_params, mlflow.log_metrics, mlflow.log_artifact

def load_params(params_path: str):
    import yaml
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Carpeta o parquet de train")
    ap.add_argument("--val",   required=True, help="Carpeta o parquet de val")
    ap.add_argument("--model-out", required=True, help="Directorio de salida de artefactos")
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()

    cfg = load_params(args.params)
    mcfg = cfg.get("model", {}).get("rf", {})
    text_col   = mcfg.get("text_col", "review_text")
    label_col  = mcfg.get("label_col", None)
    rating_col = mcfg.get("rating_col", "rating")
    preproc    = mcfg.get("preproc", {})
    tfidf_cfg  = mcfg.get("tfidf", {})
    feats_cfg  = mcfg.get("features", {})
    rf_params  = mcfg.get("model_params", {})
    seed       = mcfg.get("seed", 42)

    out_dir = Path(args.model_out)
    rep_dir = Path("reports/sentiment_rf")
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # IO
    df_tr = read_parquet_folder_or_file(args.train)
    df_va = read_parquet_folder_or_file(args.val)
    if text_col not in df_tr.columns or text_col not in df_va.columns:
        raise KeyError(f"No existe columna de texto '{text_col}' en train/val")

    # y
    y_tr = to_labels(df_tr, label_col, rating_col)
    y_va = to_labels(df_va, label_col, rating_col)

    # limpieza
    cleaner = TextCleaner(use_lemma=preproc.get("use_lemma", True),
                          lang=preproc.get("language", "en")).fit(df_tr[text_col])

    Xtr_clean = cleaner.transform(df_tr[text_col])
    Xva_clean = cleaner.transform(df_va[text_col])

    # TF-IDF
    vec = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 50000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=tfidf_cfg.get("min_df", 3),
        max_df=tfidf_cfg.get("max_df", 0.95),
        stop_words=tfidf_cfg.get("stop_words", "english")
    )
    Xtr_tfidf = vec.fit_transform(Xtr_clean)
    Xva_tfidf = vec.transform(Xva_clean)

    # extra features
    extra = ExtraFeatures(
        use_neg=feats_cfg.get("add_neg_lexicon", True),
        use_contrast=feats_cfg.get("add_contrast_markers", True),
        use_len=feats_cfg.get("add_len", True)
    )
    Xtr = hstack([Xtr_tfidf, extra.transform(Xtr_clean)], format="csr")
    Xva = hstack([Xva_tfidf, extra.transform(Xva_clean)], format="csr")

    # modelo
    rf = RandomForestClassifier(random_state=seed, **rf_params)
    rf.fit(Xtr, y_tr)

    # evaluación
    va_pred = rf.predict(Xva)
    acc = float(accuracy_score(y_va, va_pred))
    f1m = float(f1_score(y_va, va_pred, average="macro"))
    cm  = confusion_matrix(y_va, va_pred, labels=[0,1,2]).tolist()

    # guardar artefactos
    joblib.dump(rf, out_dir / "rf_model.joblib")
    joblib.dump(vec, out_dir / "tfidf_vectorizer.joblib")
    (out_dir / "preprocess_config.json").write_text(json.dumps({
        "use_lemma": preproc.get("use_lemma", True),
        "language": preproc.get("language", "en"),
        "features": feats_cfg
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # predicciones de validación (para trazabilidad / gráficos)
    val_preds_path = rep_dir / "val_predictions.parquet"
    pd.DataFrame({
        "true_label": y_va.astype("int32"),
        "pred_index": va_pred.astype("int32"),
    }).to_parquet(val_preds_path, index=False)

    # MLflow (opcional)
    run, log_params, log_metrics, log_artifact = mlflow_start_run_if_enabled(cfg)
    if run:
        # params
        log_params({
            **{f"tfidf_{k}": v for k, v in tfidf_cfg.items()},
            **{f"feat_{k}": v for k, v in feats_cfg.items()},
            **{f"rf_{k}": v for k, v in rf_params.items()},
            "seed": seed
        })
        log_metrics({"val_accuracy": acc, "val_f1_macro": f1m})
        # artefactos
        try:
            import mlflow
            mlflow.log_artifact(str(out_dir / "preprocess_config.json"))
            mlflow.log_artifact(str(val_preds_path))
        except Exception:
            pass
        import mlflow
        mlflow.end_run()

    print("Entrenamiento RF OK")
    print(f"  Acc: {acc:.4f} | F1-macro: {f1m:.4f}")
    print("  Artefactos:", out_dir)
    print("  Preds val:", val_preds_path)

if __name__ == "__main__":
    main()

# NOTA: esta clase/funciones se reutilizan desde infer_random_forest.py
