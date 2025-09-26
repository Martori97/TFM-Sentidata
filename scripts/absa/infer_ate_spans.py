#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infer ATE (BIO) spans over English reviews and write Parquet incrementally.

- Lee Delta/Parquet (auto o forzado).
- Tokeniza por palabras (whitespace) y usa modelo HF de token-classification.
- Evita .toPandas(): procesa con toLocalIterator() y escribe Parquet por lotes.
- Robusto ante Ctrl+C: al recibir SIGINT/SIGTERM, hace flush, cierra y sale.
- Muestra progreso periódico: reviews procesadas, %, spans, velocidad y ETA.

Output columns:
- review_id (str), aspect_span (str), start (int), end (int), prob (float), text_len (int)
"""

import argparse, os, sys, signal, time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pyspark.sql import SparkSession

# ------------- Señales -------------
_stop = {"flag": False}
def _handle_stop(signum, frame):
    print("[signal] Interrupt received, will flush & close...", flush=True)
    _stop["flag"] = True

signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)

# ------------- Spark + Delta -------------
def build_spark_with_delta(app="infer_ate_spans"):
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
        sys.stderr.write(f"[warn] delta-spark via pip no disponible ({repr(e)}). Intentando sin helper...\n")
        return builder.getOrCreate()

def is_delta_dir(path:str)->bool:
    return os.path.exists(os.path.join(path, "_delta_log"))

def read_any(spark, path, fmt="auto"):
    if fmt == "auto":
        fmt = "delta" if is_delta_dir(path) else "parquet"
    return spark.read.format(fmt).load(path) if fmt == "delta" else spark.read.parquet(path)

# ------------- NLP utils -------------
LABELS = ["O","B-ASP","I-ASP"]

def decode_spans(word_labels: List[str]) -> List[Tuple[int,int]]:
    spans, i, n = [], 0, len(word_labels)
    while i < n:
        if word_labels[i] == "B-ASP":
            j = i + 1
            while j < n and word_labels[j] == "I-ASP":
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans

def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--input_format", default="auto", choices=["auto","delta","parquet"])
    ap.add_argument("--text_col", default="review_text_clean")
    ap.add_argument("--id_col", default="review_id")
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--max_length", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--flush_rows", type=int, default=10000)
    ap.add_argument("--log_every_n_reviews", type=int, default=5000, help="frecuencia de logs de progreso")
    args = ap.parse_args()

    # ---- device log
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    # ---- tokenizer (Roberta needs add_prefix_space with pretokenized words)
    tok_kwargs = {"use_fast": True}
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True, **tok_kwargs)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, **tok_kwargs)

    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    # ---- Spark (Delta ready)
    spark = build_spark_with_delta()
    df = read_any(spark, args.input_path, args.input_format).select(args.id_col, args.text_col)

    # Contar total para % y ETA
    total_reviews = df.count()
    print(f"[info] total_reviews={total_reviews}", flush=True)

    # ---- incremental writer
    import pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    writer = None
    buffer: List[Dict] = []

    def flush(reason: str = "periodic"):
        nonlocal writer, buffer, produced_spans_total
        if not buffer:
            return
        pdf = pd.DataFrame(buffer)
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.output_parquet, table.schema)
        writer.write_table(table)
        wrote = len(buffer)
        produced_spans_total += wrote
        buffer.clear()
        print(f"[flush] wrote={wrote} | total_spans={produced_spans_total} | reason={reason}", flush=True)

    def whitespace_tokenize(text: str) -> List[str]:
        return text.split()

    # ---- progreso
    start_ts = time.time()
    processed_reviews = 0
    produced_spans_total = 0
    last_log_reviews = 0

    batch_ids, batch_texts, batch_word_lists = [], [], []

    try:
        for row in df.toLocalIterator():
            rid = str(row[args.id_col])
            text = (row[args.text_col] or "").strip()

            batch_ids.append(rid)
            batch_texts.append(text)
            batch_word_lists.append(whitespace_tokenize(text))

            if len(batch_texts) >= args.batch_size:
                # ---- process batch
                enc = tokenizer(batch_word_lists, is_split_into_words=True,
                                return_tensors="pt", truncation=True, padding=True,
                                max_length=args.max_length).to(device)
                with torch.no_grad():
                    logits = model(**enc).logits.detach().cpu().numpy()
                probs = softmax_np(logits)
                pred_ids = probs.argmax(axis=-1)

                # map back to word-level
                spans_in_batch = 0
                for b_idx in range(len(batch_texts)):
                    wid_map = enc.word_ids(batch_index=b_idx)
                    word_labels, word_probs, seen = [], [], set()
                    for t_idx, w_id in enumerate(wid_map):
                        if w_id is None or w_id in seen:
                            continue
                        seen.add(w_id)
                        lid = int(pred_ids[b_idx, t_idx])
                        word_labels.append(LABELS[lid])
                        word_probs.append(float(probs[b_idx, t_idx, lid]))

                    spans_idx = decode_spans(word_labels)
                    text_b = batch_texts[b_idx]
                    words_b = batch_word_lists[b_idx]

                    # char offsets
                    starts = []
                    pos = 0
                    for w in words_b:
                        s = text_b.find(w, pos)
                        starts.append(s)
                        pos = s + len(w) if s >= 0 else pos

                    for (ws, we) in spans_idx:
                        cs = starts[ws]
                        ce = starts[we-1] + len(words_b[we-1]) if we-1 < len(words_b) and starts[we-1] >= 0 else -1
                        span_text = text_b[cs:ce] if cs is not None and cs >= 0 and ce >= 0 else " ".join(words_b[ws:we])
                        conf = float(np.mean(word_probs[ws:we])) if ws < len(word_probs) and we <= len(word_probs) else 0.0
                        buffer.append({
                            "review_id": batch_ids[b_idx],
                            "aspect_span": span_text,
                            "start": int(cs) if cs is not None and cs >= 0 else None,
                            "end": int(ce) if ce is not None and ce >= 0 else None,
                            "prob": conf,
                            "text_len": len(text_b)
                        })
                        spans_in_batch += 1

                # clear batch and update progress
                processed_reviews += len(batch_texts)
                batch_ids, batch_texts, batch_word_lists = [], [], []

                # progreso periódico
                if (processed_reviews - last_log_reviews) >= max(1, args.log_every_n_reviews):
                    elapsed = time.time() - start_ts
                    rate = processed_reviews / elapsed if elapsed > 0 else 0.0
                    pct = (processed_reviews / total_reviews * 100.0) if total_reviews > 0 else 0.0
                    eta = (total_reviews - processed_reviews) / rate if rate > 0 else float("inf")
                    print(f"[progress] reviews={processed_reviews}/{total_reviews} "
                          f"({pct:.2f}%) | spans_buf={len(buffer)} | spans_total={produced_spans_total} "
                          f"| rate={rate:.1f} r/s | eta={fmt_eta(eta)}", flush=True)
                    last_log_reviews = processed_reviews

                # flush si grande
                if len(buffer) >= args.flush_rows:
                    flush(reason="size")

                # salida segura si pidieron parar
                if _stop["flag"]:
                    flush(reason="signal")
                    if writer is not None:
                        writer.close()
                    elapsed = time.time() - start_ts
                    print(f"[exit] interrupted. processed_reviews={processed_reviews}/{total_reviews} | "
                          f"total_spans={produced_spans_total} | elapsed={fmt_eta(elapsed)}", flush=True)
                    return

        # ---- procesa cola final si quedó algo sin cerrar
        if batch_texts:
            enc = tokenizer(batch_word_lists, is_split_into_words=True,
                            return_tensors="pt", truncation=True, padding=True,
                            max_length=args.max_length).to(device)
            with torch.no_grad():
                logits = model(**enc).logits.detach().cpu().numpy()
            probs = softmax_np(logits)
            pred_ids = probs.argmax(axis=-1)

            spans_in_batch = 0
            for b_idx in range(len(batch_texts)):
                wid_map = enc.word_ids(batch_index=b_idx)
                word_labels, word_probs, seen = [], [], set()
                for t_idx, w_id in enumerate(wid_map):
                    if w_id is None or w_id in seen:
                        continue
                    seen.add(w_id)
                    lid = int(pred_ids[b_idx, t_idx])
                    word_labels.append(LABELS[lid])
                    word_probs.append(float(probs[b_idx, t_idx, lid]))

                spans_idx = decode_spans(word_labels)
                text_b = batch_texts[b_idx]
                words_b = batch_word_lists[b_idx]

                starts = []
                pos = 0
                for w in words_b:
                    s = text_b.find(w, pos)
                    starts.append(s)
                    pos = s + len(w) if s >= 0 else pos

                for (ws, we) in spans_idx:
                    cs = starts[ws]
                    ce = starts[we-1] + len(words_b[we-1]) if we-1 < len(words_b) and starts[we-1] >= 0 else -1
                    span_text = text_b[cs:ce] if cs is not None and cs >= 0 and ce >= 0 else " ".join(words_b[ws:we])
                    conf = float(np.mean(word_probs[ws:we])) if ws < len(word_probs) and we <= len(word_probs) else 0.0
                    buffer.append({
                        "review_id": batch_ids[b_idx],
                        "aspect_span": span_text,
                        "start": int(cs) if cs is not None and cs >= 0 else None,
                        "end": int(ce) if ce is not None and ce >= 0 else None,
                        "prob": conf,
                        "text_len": len(text_b)
                    })
                    spans_in_batch += 1

            processed_reviews += len(batch_texts)

        # flush final
        flush(reason="final")
    finally:
        if writer is not None:
            writer.close()
        elapsed = time.time() - start_ts
        rate = processed_reviews / elapsed if elapsed > 0 else 0.0
        print(f"[saved] {args.output_parquet} | processed_reviews={processed_reviews}/{total_reviews} "
              f"| total_spans={produced_spans_total} | elapsed={fmt_eta(elapsed)} | rate={rate:.1f} r/s", flush=True)

if __name__ == "__main__":
    main()


