#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build ATE trusted dataset (BIO) from ontology aliases over English reviews.

- Lee trusted (Delta/Parquet) y ontología JSON (aspect -> aliases).
- Tokeniza por espacios (whitespace) y etiqueta BIO al detectar alias (case-insensitive, word-level).
- Genera JSONL con columnas: {"tokens": [...], "labels": [...]}.
- Divide en train/valid y escribe dos ficheros JSONL bajo data/trusted/ate/.

Ejemplo:
python scripts/absa/build_ate_trusted_from_ontology.py \
  --input_path data/trusted/reviews_product_info_clean_full \
  --input_format delta \
  --text_col review_text_clean \
  --id_col review_id \
  --ontology_json configs/aspects_en.json \
  --output_dir data/trusted/ate \
  --train_filename train.jsonl \
  --valid_filename valid.jsonl \
  --sample_fraction 0.2 \
  --max_reviews 300000 \
  --train_valid_split 0.9 \
  --keep_neg_ratio 0.2 \
  --seed 13
"""
import argparse, os, json, random, re, sys
from typing import List, Dict
from pyspark.sql import SparkSession

# -------- Spark con Delta --------
def build_spark_with_delta(app="build_ate_trusted"):
    """
    Crea SparkSession con Delta (extensión + catálogo).
    Usa configure_spark_with_delta_pip si está disponible (paquete delta-spark).
    """
    builder = (
        SparkSession.builder
        .appName(app)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.files.ignoreCorruptFiles", "true")
        .config("spark.sql.shuffle.partitions", "200")
    )
    try:
        from delta import configure_spark_with_delta_pip
        return configure_spark_with_delta_pip(builder).getOrCreate()
    except Exception as e:
        sys.stderr.write(f"[warn] delta-spark via pip no disponible ({repr(e)}). Intentando Spark sin helper...\n")
        return builder.getOrCreate()

def is_delta_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, "_delta_log"))

def read_any(spark, path, fmt="auto"):
    if fmt == "auto":
        fmt = "delta" if is_delta_dir(path) else "parquet"
    if fmt == "delta":
        return spark.read.format("delta").load(path)
    else:
        return spark.read.parquet(path)

# -------- Utilidades de etiquetado --------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def compile_alias_patterns(ontology: Dict[str, list]) -> Dict[str, list]:
    """Para cada aspecto, lista de alias tokenizados (lowercase)."""
    compiled = {}
    for asp, aliases in ontology.items():
        toks = []
        seen = set()
        for a in aliases:
            a_norm = normalize(a)
            if not a_norm or a_norm in seen:
                continue
            seen.add(a_norm)
            toks.append(a_norm.split())
        if toks:
            compiled[asp] = toks
    return compiled

def label_tokens_with_aliases(tokens: List[str], alias_token_lists: List[List[str]]) -> List[str]:
    """
    Marca BIO si cualquier alias (lista de tokens) aparece en 'tokens' (case-insensitive).
    Si se solapan alias, se prioriza el más largo (greedy).
    """
    n = len(tokens)
    labels = ["O"] * n
    low = [t.lower() for t in tokens]
    alias_sorted = sorted(alias_token_lists, key=len, reverse=True)
    i = 0
    while i < n:
        matched = False
        for alias in alias_sorted:
            L = len(alias)
            if L == 0 or i + L > n:
                continue
            if low[i:i+L] == alias:
                labels[i] = "B-ASP"
                for k in range(1, L):
                    labels[i+k] = "I-ASP"
                i += L
                matched = True
                break
        if not matched:
            i += 1
    return labels

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--input_format", default="auto", choices=["auto","delta","parquet"])
    ap.add_argument("--text_col", default="review_text_clean")
    ap.add_argument("--id_col", default="review_id")
    ap.add_argument("--ontology_json", required=True)
    ap.add_argument("--output_dir", default="data/trusted/ate")
    ap.add_argument("--train_filename", default="train.jsonl")
    ap.add_argument("--valid_filename", default="valid.jsonl")
    ap.add_argument("--sample_fraction", type=float, default=0.2)
    ap.add_argument("--max_reviews", type=int, default=300000)
    ap.add_argument("--train_valid_split", type=float, default=0.9)
    ap.add_argument("--keep_neg_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, args.train_filename)
    valid_path = os.path.join(args.output_dir, args.valid_filename)

    # Ontología
    with open(args.ontology_json, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    alias_dict = compile_alias_patterns(ontology)
    alias_compiled = []
    for _, alias_lists in alias_dict.items():
        for al in alias_lists:
            tup = tuple(al)
            if tup not in alias_compiled:
                alias_compiled.append(tup)
    alias_compiled = [list(t) for t in alias_compiled]
    if not alias_compiled:
        raise ValueError("La ontología no contiene alias válidos.")

    # Spark con Delta
    spark = build_spark_with_delta()

    # Carga dataset
    df = read_any(spark, args.input_path, args.input_format).select(args.id_col, args.text_col)

    # Muestreo
    frac = max(0.0, min(1.0, args.sample_fraction))
    if 0 < frac < 1.0:
        df = df.sample(withReplacement=False, fraction=frac, seed=args.seed)
    if args.max_reviews and args.max_reviews > 0:
        df = df.limit(args.max_reviews)

    # Escritura incremental de JSONL
    n_total = n_pos = n_neg_kept = n_written = 0
    tmp_train = train_path + ".tmp"
    tmp_valid = valid_path + ".tmp"
    tf = open(tmp_train, "w", encoding="utf-8")
    vf = open(tmp_valid, "w", encoding="utf-8")

    try:
        for row in df.toLocalIterator():
            n_total += 1
            text = row[args.text_col]
            if not text:
                continue
            text = text.strip()
            if not text:
                continue

            tokens = text.split()
            if not tokens:
                continue

            labels = label_tokens_with_aliases(tokens, alias_compiled)
            has_aspect = any(l != "O" for l in labels)
            out_line = {"tokens": tokens, "labels": labels}

            if has_aspect:
                n_pos += 1
                if random.random() < args.train_valid_split:
                    tf.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                else:
                    vf.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                n_written += 1
            else:
                if random.random() < args.keep_neg_ratio:
                    n_neg_kept += 1
                    if random.random() < args.train_valid_split:
                        tf.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                    else:
                        vf.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                    n_written += 1
    finally:
        tf.close()
        vf.close()

    os.replace(tmp_train, train_path)
    os.replace(tmp_valid, valid_path)

    print(f"[trusted-ate] total_seen={n_total} | written={n_written} | positives={n_pos} | negatives_kept={n_neg_kept}")
    print(f"[trusted-ate] train: {train_path} | valid: {valid_path}")
    spark.stop()

if __name__ == "__main__":
    main()

