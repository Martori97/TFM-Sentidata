#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

# ===== IO =====
def read_parquet_folder_or_file(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if p.is_dir():
        files = sorted(p.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No se encontraron .parquet en {p}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(p)

# ===== Limpieza texto (igual filosofía que RF/SVM train) =====
class TextCleaner:
    def __init__(self, use_lemma: bool = True, lang: str = "en"):
        self.use_lemma = bool(use_lemma)
        self.lang = lang
        self._spacy = None

    def _ensure_spacy(self):
        if self._spacy is not None or not self.use_lemma:
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

    def fit(self, X):  # por compatibilidad
        return self

    def _basic_clean(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"http\S+|www\.\S+", " ", s)
        s = re.sub(r"[^a-záéíóúñüç0-9\s]", " ", s)
        s = re.sub(r"\d+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _lemmatize_list(self, toks):
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

# ===== Extra features (idéntico a train) =====
_NEG_LEXICON = {"bad","terrible","awful","worse","worst","horrible","disappointed","poor","hate","lousy",
                "malo","terrible","horrible","peor","decepcionado","pobre","odiar","fatal"}
_CONTRAST_MARKERS = {"but","however","though","although","yet","aunque","pero","sin embargo"}

class ExtraFeatures:
    """CSR con columnas: [neg_lex_count, contrast_markers_count, token_len]"""
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

# ===== Labels helpers =====
def derive_label_from_rating(series) -> np.ndarray:
    r = pd.to_numeric(series, errors="coerce").round().clip(1, 5).astype("Int64")
    return r.map({1:"negative", 2:"negative", 3:"neutral", 4:"positive", 5:"positive"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True)   # carpeta/archivo parquet (test)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-col", default="review_text")
    ap.add_argument("--id-col", default="review_id")
    ap.add_argument("--label-col", default="sentiment_category")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    svm_path = model_dir / "svm_model.joblib"
    vec_path = model_dir / "tfidf_vectorizer.joblib"
    preproc_cfg_path = model_dir / "preprocess_config.json"

    if not svm_path.exists() or not vec_path.exists():
        raise FileNotFoundError("Faltan artefactos del modelo: svm_model.joblib / tfidf_vectorizer.joblib")

    # Carga artefactos
    svm = joblib.load(svm_path)
    vec = joblib.load(vec_path)
    preproc_cfg = {"use_lemma": True, "language": "en", "features": {"add_neg": True, "add_contrast": True, "add_len": True}}
    if preproc_cfg_path.exists():
        preproc_cfg = json.loads(preproc_cfg_path.read_text(encoding="utf-8"))

    # Carga test
    df_te = read_parquet_folder_or_file(args.input)
    if args.text_col not in df_te.columns:
        raise KeyError(f"No existe columna de texto '{args.text_col}' en test")

    # Preprocesado texto
    cleaner = TextCleaner(
        use_lemma=preproc_cfg.get("use_lemma", True),
        lang=preproc_cfg.get("language", "en")
    ).fit(df_te[args.text_col])
    Xte_clean = cleaner.transform(df_te[args.text_col])

    # TF-IDF + extra feats (usando misma config)
    Xte_tfidf = vec.transform(Xte_clean)

    feats_cfg = preproc_cfg.get("features", {})
    extra = ExtraFeatures(
        use_neg=feats_cfg.get("add_neg", feats_cfg.get("add_neg_lexicon", True)),
        use_contrast=feats_cfg.get("add_contrast", feats_cfg.get("add_contrast_markers", True)),
        use_len=feats_cfg.get("add_len", True),
    )
    Xte = hstack([Xte_tfidf, extra.transform(Xte_clean)], format="csr")

    # Inferencia
    te_pred_idx = svm.predict(Xte)
    idx2label = {0:"negative", 1:"neutral", 2:"positive"}
    # LinearSVC suele devolver índices si entrenaste con y numérico; si entrenaste con strings, ya viene label
    if np.issubdtype(te_pred_idx.dtype, np.integer):
        te_pred = pd.Series(te_pred_idx).map(idx2label)
    else:
        te_pred = pd.Series(te_pred_idx).astype(str)

    # Construir salida
    out = pd.DataFrame({
        args.id_col: df_te[args.id_col] if args.id_col in df_te.columns else np.arange(len(df_te)),
        "prediction": te_pred
    })

    # Añadir label si existe o derivarlo desde rating (útil para métricas y CM)
    if args.label_col in df_te.columns:
        out[args.label_col] = df_te[args.label_col].astype(str)
    elif "rating" in df_te.columns:
        out[args.label_col] = derive_label_from_rating(df_te["rating"])

    # Guardar
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_predictions.parquet"
    out.to_parquet(out_path, index=False)

    print("[ok] SVM inferido →", out_path)
    print(out.head())

if __name__ == "__main__":
    main()

