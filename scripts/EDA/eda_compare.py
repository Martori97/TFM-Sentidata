#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os
import pandas as pd
import numpy as np
from scipy.stats import chisquare

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full", required=True, help="carpeta EDA full (report/eda/full)")
    p.add_argument("--sample", required=True, help="carpeta EDA sample (report/eda/sample_XX)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--chiN", type=int, default=100_000,
                   help="N sintético para convertir % a cuentas enteras en Chi²")
    return p.parse_args()

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def pct_to_counts(pcts, N):
    """
    Convierte un vector de porcentajes a cuentas enteras que suman exactamente N.
    Maneja NaNs (los trata como 0). Si hay diferencia por redondeo, la ajusta
    en la categoría de mayor peso absoluto.
    """
    p = np.nan_to_num(np.asarray(pcts, dtype=float), nan=0.0)
    counts = np.rint(p / 100.0 * N).astype(int)
    diff = int(N - counts.sum())
    if diff != 0 and counts.size > 0:
        # ajustamos en la categoría más grande (o la primera si hay empate)
        idx = int(np.argmax(counts))
        counts[idx] += diff
        # evitar negativos por redondeos extremos
        if counts[idx] < 0:
            # si se fue negativo (muy raro), redistribuir de forma segura
            deficit = -counts[idx]
            counts[idx] = 0
            for j in np.argsort(-counts):  # de mayor a menor
                if deficit == 0: break
                take = min(counts[j], deficit)
                counts[j] -= take
                deficit -= take
    return counts

def chi2_from_pct(p_full, p_sample, N=100_000):
    """
    Aplica Chi² sobre cuentas sintéticas derivadas de %,
    garantizando mismas sumas y evitando problemas numéricos.
    """
    if len(p_full) == 0 or len(p_sample) == 0:
        return {"chi2": None, "p_value": None, "N": int(N), "note": "distribuciones vacías"}

    obs = pct_to_counts(p_sample, N)
    exp = pct_to_counts(p_full,  N)

    mask = (exp > 0) | (obs > 0)
    if mask.sum() < 2:
        return {"chi2": None, "p_value": None, "N": int(N), "note": "insuficiente soporte"}

    chi, p = chisquare(f_obs=obs[mask], f_exp=exp[mask])
    return {"chi2": float(chi), "p_value": float(p), "N": int(N)}

def diff_distribution(full_csv, sample_csv, key_col, out_csv):
    f = pd.read_csv(full_csv)
    s = pd.read_csv(sample_csv)
    merged = pd.merge(f, s, on=key_col, how="outer", suffixes=("_full", "_sample")).fillna(0.0)
    merged["diff_pct"] = merged["pct_sample"] - merged["pct_full"]
    merged.to_csv(out_csv, index=False)
    return merged

def main():
    args = parse_args()
    safe_mkdir(args.outdir)

    # 1) Sentimiento
    sent_full = os.path.join(args.full, "dist_sentiment.csv")
    sent_samp = os.path.join(args.sample, "dist_sentiment.csv")
    if os.path.exists(sent_full) and os.path.exists(sent_samp):
        diff_sent = diff_distribution(sent_full, sent_samp, "sentiment_category",
                                      os.path.join(args.outdir, "diff_sentiment.csv"))
        chi_res = chi2_from_pct(diff_sent["pct_full"].values, diff_sent["pct_sample"].values, N=args.chiN)
        with open(os.path.join(args.outdir, "chi2_sentiment.json"), "w") as f:
            json.dump(chi_res, f, indent=2)

    # 2) Categorías
    cat_full = os.path.join(args.full, "dist_category.csv")
    cat_samp = os.path.join(args.sample, "dist_category.csv")
    if os.path.exists(cat_full) and os.path.exists(cat_samp):
        diff_distribution(cat_full, cat_samp, "secondary_category",
                          os.path.join(args.outdir, "diff_category.csv"))

    # 3) Longitud caracteres (stats agregadas)
    len_full = os.path.join(args.full, "len_stats.json")
    len_samp = os.path.join(args.sample, "len_stats.json")
    if os.path.exists(len_full) and os.path.exists(len_samp):
        lf = json.load(open(len_full))
        ls = json.load(open(len_samp))
        with open(os.path.join(args.outdir, "diff_len.json"), "w") as f:
            json.dump({
                "mean_diff": (ls.get("mean") - lf.get("mean")) if ("mean" in ls and "mean" in lf) else None,
                "p50_diff":  (ls.get("p50")  - lf.get("p50"))  if ("p50"  in ls and "p50"  in lf) else None,
                "p95_diff":  (ls.get("p95")  - lf.get("p95"))  if ("p95"  in ls and "p95"  in lf) else None,
                "p99_diff":  (ls.get("p99")  - lf.get("p99"))  if ("p99"  in ls and "p99"  in lf) else None,
            }, f, indent=2)

    # 4) TOKENS (hist Chi² + diffs de percentiles)
    tok_full_stats = os.path.join(args.full, "token_stats.json")
    tok_samp_stats = os.path.join(args.sample, "token_stats.json")
    hist_full = os.path.join(args.full, "tokens_hist.csv")
    hist_samp = os.path.join(args.sample, "tokens_hist.csv")

    if os.path.exists(hist_full) and os.path.exists(hist_samp):
        hf = pd.read_csv(hist_full)
        hs = pd.read_csv(hist_samp)
        # Alinear por bins (asumimos mismos edges por mismo --token-bins)
        merged = hf.merge(hs, on=["bin_left", "bin_right"], how="outer",
                          suffixes=("_full", "_sample")).fillna(0.0)
        merged.to_csv(os.path.join(args.outdir, "tokens_hist_merged.csv"), index=False)

        chi_tok = chi2_from_pct(merged["pct_full"].values, merged["pct_sample"].values, N=args.chiN)
        with open(os.path.join(args.outdir, "chi2_tokens.json"), "w") as f:
            json.dump(chi_tok, f, indent=2)

    if os.path.exists(tok_full_stats) and os.path.exists(tok_samp_stats):
        tf = json.load(open(tok_full_stats))
        ts = json.load(open(tok_samp_stats))
        with open(os.path.join(args.outdir, "diff_tokens.json"), "w") as f:
            json.dump({
                "mean_diff": (ts.get("mean") - tf.get("mean")) if ("mean" in ts and "mean" in tf) else None,
                "p50_diff":  (ts.get("p50")  - tf.get("p50"))  if ("p50"  in ts and "p50"  in tf) else None,
                "p95_diff":  (ts.get("p95")  - tf.get("p95"))  if ("p95"  in ts and "p95"  in tf) else None,
                "p99_diff":  (ts.get("p99")  - tf.get("p99"))  if ("p99"  in ts and "p99"  in tf) else None,
                "max_diff":  (ts.get("max")  - tf.get("max"))  if ("max"  in ts and "max"  in tf) else None,
                "tokenizer_full": tf.get("tokenizer"),
                "tokenizer_sample": ts.get("tokenizer")
            }, f, indent=2)

    print(f"✅ Comparativa EDA guardada en {args.outdir}")

if __name__ == "__main__":
    main()
