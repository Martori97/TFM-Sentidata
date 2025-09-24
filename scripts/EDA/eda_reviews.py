#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA de reviews (full o sample) con:
- Resumen básico (summary.json)
- Distribución de sentimiento (dist_sentiment.csv) — derivando desde rating si falta
- Distribución de categoría (dist_category.csv)
- Estadísticos de longitudes en caracteres (len_stats.json)
- Tokenización con HF: token_stats.json (+ overflow >512), tokens_hist.csv
- Figuras en outdir/figs
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Parámetros CLI ----
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Ruta parquet (carpeta o archivo)")
    p.add_argument("--outdir", required=True, help="Carpeta de salida")
    p.add_argument("--sent-col", default="sentiment_category")
    p.add_argument("--cat-col", default="secondary_category")
    p.add_argument("--text-col", default="review_text")
    p.add_argument("--tokenizer", default="albert-base-v2", help="HF tokenizer")
    p.add_argument("--batch-size", type=int, default=4096, help="filas por lote para tokenización")
    p.add_argument("--max-cats", type=int, default=30, help="Top categorías para la figura")
    p.add_argument("--token-bins", type=int, default=60, help="bins del hist de tokens")
    p.add_argument("--model-max", type=int, default=512, help="límite de tokens del modelo (ALBERT=512)")
    return p.parse_args()

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    os.makedirs(os.path.join(p, "figs"), exist_ok=True)

# ---- Tokenización por lotes ----
def compute_token_lengths(texts: pd.Series, tokenizer, batch_size=4096, include_special=True):
    n = len(texts)
    out = np.zeros(n, dtype=np.int32)
    # Procesar por lotes para ahorrar RAM
    for i in range(0, n, batch_size):
        chunk = texts.iloc[i:i+batch_size].fillna("")
        enc = tokenizer(
            chunk.tolist(),
            add_special_tokens=include_special,
            truncation=False,
            return_attention_mask=False
        )["input_ids"]
        out[i:i+batch_size] = [len(ids) for ids in enc]
    return out

def main():
    args = parse_args()
    safe_mkdir(args.outdir)

    # ---- Carga parquet (archivo o carpeta) ----
    # pd.read_parquet soporta carpeta si es dataset de parquet
    df = pd.read_parquet(args.input)

    sent_col, cat_col, text_col = args.sent_col, args.cat_col, args.text_col

    # ---- Resumen básico ----
    summary = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "sentiment_na": int(df[sent_col].isna().sum()) if sent_col in df.columns else None,
        "category_na": int(df[cat_col].isna().sum()) if cat_col in df.columns else None,
        "text_na": int(df[text_col].isna().sum()) if text_col in df.columns else None,
        "columns": list(df.columns),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Distribución de sentimiento (derivar desde rating si falta) ----
    rating_col = "rating"
    if sent_col not in df.columns and rating_col in df.columns:
        def _to_label3(r):
            try:
                r = float(r)
            except Exception:
                return None
            if 1 <= r <= 2: return "neg"
            if r == 3:      return "neu"
            if 4 <= r <= 5: return "pos"
            return None
        df[sent_col] = df[rating_col].apply(_to_label3)

    # Crear SIEMPRE el CSV (vacío si no hay columna o todo NAs)
    if sent_col in df.columns:
        dist_sent = (df[sent_col]
                     .dropna()
                     .value_counts(normalize=True) * 100).rename("pct").reset_index()
        dist_sent.columns = [sent_col, "pct"]
    else:
        dist_sent = pd.DataFrame(columns=[sent_col, "pct"])

    dist_sent.to_csv(os.path.join(args.outdir, "dist_sentiment.csv"), index=False)

    # Figura si hay datos
    if sent_col in df.columns and not dist_sent.empty:
        ax = (df[sent_col].dropna().value_counts(normalize=True).sort_index() * 100).plot(kind="bar")
        ax.set_ylabel("%")
        ax.set_title("Distribución de Sentimiento (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "figs", "dist_sentiment.png"))
        plt.close()

    # ---- Distribución de categoría (CSV siempre; figura solo si hay datos) ----
    if cat_col in df.columns:
        dist_cat_full = df[cat_col].dropna().value_counts(normalize=True) * 100
        dist_cat = dist_cat_full.rename("pct").reset_index()
        dist_cat.columns = [cat_col, "pct"]
    else:
        dist_cat = pd.DataFrame(columns=[cat_col, "pct"])

    dist_cat.to_csv(os.path.join(args.outdir, "dist_category.csv"), index=False)

    if cat_col in df.columns and not dist_cat.empty:
        topN = dist_cat.head(args.max_cats).set_index(cat_col)["pct"]
        ax = topN.plot(kind="bar")
        ax.set_ylabel("%")
        ax.set_title(f"Top {args.max_cats} categorías (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "figs", "top_categories.png"))
        plt.close()

    # ---- Longitud en caracteres (stats + hist) ----
    if text_col in df.columns:
        lens = df[text_col].fillna("").str.len()
        len_stats = {
            "count": int(lens.shape[0]),
            "mean": float(lens.mean()),
            "std": float(lens.std()),
            "min": int(lens.min()),
            "p50": float(lens.quantile(0.50)),
            "p90": float(lens.quantile(0.90)),
            "p95": float(lens.quantile(0.95)),
            "p99": float(lens.quantile(0.99)),
            "max": int(lens.max()),
        }
        with open(os.path.join(args.outdir, "len_stats.json"), "w") as f:
            json.dump(len_stats, f, indent=2)

        plt.hist(lens, bins=50)
        plt.title("Distribución de longitudes (caracteres)")
        plt.xlabel("caracteres")
        plt.ylabel("frecuencia")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "figs", "len_hist.png"))
        plt.close()
    else:
        with open(os.path.join(args.outdir, "len_stats.json"), "w") as f:
            json.dump({"note": f"columna {text_col} no encontrada"}, f, indent=2)

    # ---- TOKENS (HF) ----
    # Silenciar warnings de transformers y paralelismo de tokenizers
    try:
        import os as _os
        _os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # Evitar warning de >512 al contar tokens (aquí no inferimos)
        tokenizer.model_max_length = int(1e7)

        if text_col in df.columns:
            n_tokens = compute_token_lengths(
                df[text_col].fillna(""),
                tokenizer,
                batch_size=args.batch_size,
                include_special=True
            )

            token_stats = {
                "count": int(n_tokens.shape[0]),
                "mean": float(np.mean(n_tokens)),
                "std": float(np.std(n_tokens)),
                "min": int(np.min(n_tokens)),
                "p50": float(np.percentile(n_tokens, 50)),
                "p90": float(np.percentile(n_tokens, 90)),
                "p95": float(np.percentile(n_tokens, 95)),
                "p99": float(np.percentile(n_tokens, 99)),
                "max": int(np.max(n_tokens)),
                "tokenizer": args.tokenizer,
                "include_special_tokens": True,
                "model_max_length": int(args.model_max),
            }

            # Overflow > model_max
            overflow_mask = n_tokens > args.model_max
            token_stats.update({
                "overflow_count": int(overflow_mask.sum()),
                "overflow_pct": float(overflow_mask.mean() * 100.0),
            })

            with open(os.path.join(args.outdir, "token_stats.json"), "w") as f:
                json.dump(token_stats, f, indent=2)

            # Histograma (para comparativa Chi² después)
            counts, bin_edges = np.histogram(n_tokens, bins=args.token_bins)
            tokens_hist_df = pd.DataFrame({
                "bin_left": bin_edges[:-1],
                "bin_right": bin_edges[1:],
                "count": counts
            })
            tokens_hist_df["pct"] = tokens_hist_df["count"] / tokens_hist_df["count"].sum() * 100.0
            tokens_hist_df.to_csv(os.path.join(args.outdir, "tokens_hist.csv"), index=False)

            # Figura: tokens (reales)
            plt.hist(n_tokens, bins=args.token_bins)
            plt.xlabel("tokens por review")
            plt.ylabel("frecuencia")
            plt.title(f"Distribución de tokens ({args.tokenizer})")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "figs", "tokens_hist.png"))
            plt.close()

            # Figura: tokens truncados a model_max (visión entrenamiento)
            n_tokens_trunc = np.minimum(n_tokens, args.model_max)
            plt.hist(n_tokens_trunc, bins=args.token_bins)
            plt.xlabel(f"tokens por review (cap a {args.model_max})")
            plt.ylabel("frecuencia")
            plt.title(f"Distribución de tokens truncados a {args.model_max}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "figs", f"tokens_hist_trunc{args.model_max}.png"))
            plt.close()
        else:
            with open(os.path.join(args.outdir, "token_stats.json"), "w") as f:
                json.dump({"error": f"columna {text_col} no encontrada"}, f, indent=2)
            # también dejamos un CSV vacío para DVC si fuera necesario
            pd.DataFrame(columns=["bin_left","bin_right","count","pct"]).to_csv(
                os.path.join(args.outdir, "tokens_hist.csv"), index=False
            )

    except Exception as e:
        # Nunca romper el EDA por tokenización
        with open(os.path.join(args.outdir, "token_stats.json"), "w") as f:
            json.dump({"error": str(e)}, f, indent=2)
        # CSV vacío para que DVC no falle si se esperaba
        pd.DataFrame(columns=["bin_left","bin_right","count","pct"]).to_csv(
            os.path.join(args.outdir, "tokens_hist.csv"), index=False
        )

    print(f"✅ EDA guardado en {args.outdir}")

if __name__ == "__main__":
    main()
