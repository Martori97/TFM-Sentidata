#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, json, warnings, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import confusion_matrix
import pyarrow as pa, pyarrow.dataset as ds, pyarrow.parquet as pq
from tqdm import tqdm

CLASS_ID_TO_NAME = {0:"negative",1:"neutral",2:"positive"}
POSSIBLE_LABEL_COLS = ["label","label3","target"]

DEFAULTS = {
  "text_col":"review_text_clean","id_col":"review_id","max_len":192,
  "batch_size":256,"stream_batch":20000,"no_probs":False,"pad_to_multiple_of":8,
  "bf16":True,"fp16":False,"enable_sdp_flash":True,
  "torch_compile":False,"torch_compile_mode":"reduce-overhead",
  "torch_compile_backend":"inductor","tokenizers_parallelism":True,
}

def load_params_yaml(path="params.yaml"):
    try:
        import yaml
    except Exception:
        return {}
    if not os.path.exists(path): return {}
    with open(path,"r") as f: return (yaml.safe_load(f) or {})

def map_rating_to_label3(r):
    try: r = int(r)
    except: return None
    if r in (1,2): return 0
    if r == 3: return 1
    if r in (4,5): return 2
    return None

def ensure_parent(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

def load_model_robust(model_dir, device, num_labels=None):
    try:
        m = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=num_labels if num_labels is not None else None
        ).to(device).eval()
        return m
    except Exception as e:
        print(f"[warn] carga directa falló: {e}", file=sys.stderr)
        ckpt_bin = os.path.join(model_dir,"pytorch_model.bin")
        ckpt_safe= os.path.join(model_dir,"model.safetensors")
        if not (os.path.exists(ckpt_bin) or os.path.exists(ckpt_safe)):
            raise RuntimeError(f"Sin pesos en {model_dir}")
        cfg = AutoConfig.from_pretrained(model_dir)
        if num_labels is not None: cfg.num_labels = num_labels
        m = AutoModelForSequenceClassification.from_config(cfg).to(device).eval()
        if os.path.exists(ckpt_safe):
            from safetensors.torch import load_file as sload
            sd = sload(ckpt_safe, device="cpu")
        else:
            sd = torch.load(ckpt_bin, map_location="cpu")
        new_sd = {k.replace("albert._orig_mod.","albert."): v for k,v in sd.items()}
        miss, unexp = m.load_state_dict(new_sd, strict=False)
        print(f"[fix] remap _orig_mod → albert. | missing={len(miss)} unexpected={len(unexp)}", file=sys.stderr)
        return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--eval_output_dir", default=None)
    args = ap.parse_args()

    params = load_params_yaml("params.yaml")
    infer = (params.get("infer") or {})
    model_cfg = (params.get("model") or {}).get("albert", {})

    text_col  = infer.get("text_col", model_cfg.get("text_col", DEFAULTS["text_col"]))
    id_col    = infer.get("id_col", DEFAULTS["id_col"])
    max_len   = int(infer.get("max_len", DEFAULTS["max_len"]))
    bsz       = int(infer.get("batch_size", DEFAULTS["batch_size"]))
    stream_bs = int(infer.get("stream_batch", DEFAULTS["stream_batch"]))
    no_probs  = bool(infer.get("no_probs", DEFAULTS["no_probs"]))
    pad_m8    = infer.get("pad_to_multiple_of", DEFAULTS["pad_to_multiple_of"])
    bf16      = bool(infer.get("bf16", DEFAULTS["bf16"]))
    fp16      = bool(infer.get("fp16", DEFAULTS["fp16"])) if not bf16 else False
    sdp_flash = bool(infer.get("enable_sdp_flash", DEFAULTS["enable_sdp_flash"]))
    tcompile  = bool(infer.get("torch_compile", DEFAULTS["torch_compile"]))
    tmode     = str(infer.get("torch_compile_mode", DEFAULTS["torch_compile_mode"]))
    tbackend  = str(infer.get("torch_compile_backend", DEFAULTS["torch_compile_backend"]))
    tok_par   = bool(infer.get("tokenizers_parallelism", DEFAULTS["tokenizers_parallelism"]))

    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tok_par else "false"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    num_labels = model_cfg.get("num_labels", None)
    model = load_model_robust(args.model_dir, device, num_labels=num_labels)

    if sdp_flash and device=="cuda":
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except Exception:
            pass

    if tcompile and hasattr(torch,"compile") and device=="cuda":
        try:
            model = torch.compile(model, backend=tbackend, mode=tmode, fullgraph=True)
            print(f"[torch.compile] enabled backend={tbackend} mode={tmode}")
        except Exception as e:
            print(f"[torch.compile] disabled ({e})")

    ds_in = ds.dataset(args.input_parquet, format="parquet")
    cols = set(ds_in.schema.names)
    if text_col not in cols:
        raise SystemExit(f"No existe la columna '{text_col}'. Columnas: {sorted(cols)}")

    read_cols = [text_col] + ([id_col] if id_col in cols else [])
    for c in ["rating", *POSSIBLE_LABEL_COLS]:
        if c in cols: read_cols.append(c)

    ensure_parent(args.output_parquet)
    writer, wrote = None, 0
    cm_sum = np.zeros((3,3), dtype=np.int64)
    have_truth = False

    def write_block(df_out):
        nonlocal writer, wrote
        table = pa.Table.from_pandas(df_out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.output_parquet, table.schema)
        writer.write_table(table)
        wrote += len(df_out)

    def predict_batch_texts(texts):
        preds, probs = [], None if no_probs else []
        ac_dtype = torch.bfloat16 if (device=="cuda" and bf16) else (torch.float16 if (device=="cuda" and fp16) else None)
        for i in range(0, len(texts), bsz):
            enc = tok(texts[i:i+bsz], padding=True, truncation=True, max_length=max_len,
                      pad_to_multiple_of=pad_m8, return_tensors="pt")
            enc = {k: v.to(device, non_blocking=True) for k,v in enc.items()}
            ctx = torch.autocast("cuda", dtype=ac_dtype) if ac_dtype is not None else torch.inference_mode()
            with torch.inference_mode(), ctx:
                logits = model(**enc).logits
            pred = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            preds.extend(pred)
            if not no_probs:
                p = torch.softmax(logits.to(torch.float32), dim=-1).detach().cpu().numpy().tolist()
                probs.extend(p)
        return preds, probs

    scanner = ds_in.scanner(columns=read_cols, batch_size=stream_bs)
    for rb in tqdm(scanner.to_batches(), desc="Infer", unit="batch"):
        df = rb.to_pandas()
        texts = df[text_col].astype(str).fillna("").tolist()

        pred_ids, prob_list = predict_batch_texts(texts)

        out = {
            id_col: df[id_col].values if id_col in df.columns else np.arange(wrote, wrote+len(df)),
            "pred_label_id": np.array(pred_ids, dtype=np.int64),
            "pred_label": np.array([CLASS_ID_TO_NAME.get(i, f"class_{i}") for i in pred_ids], dtype=object),
        }
        if not no_probs and prob_list is not None:
            prob_arr = np.array(prob_list, dtype=np.float32)
            out["prob_neg"], out["prob_neu"], out["prob_pos"] = prob_arr[:,0], prob_arr[:,1], prob_arr[:,2]
        for c in [*POSSIBLE_LABEL_COLS, "rating"]:
            if c in df.columns: out[c] = df[c].values

        write_block(pd.DataFrame(out))

        truth = None
        for c in POSSIBLE_LABEL_COLS:
            if c in df.columns: truth = df[c].to_numpy(); break
        if truth is None and "rating" in df.columns:
            truth = np.array([map_rating_to_label3(x) for x in df["rating"].tolist()])
        if truth is not None:
            y = np.array(pred_ids, dtype=int)[:len(truth)]
            t = np.array(truth)[:len(y)]
            m = pd.Series(t).notna().to_numpy()
            if m.any():
                cm_sum += confusion_matrix(t[m].astype(int), y[m], labels=[0,1,2])
                have_truth = True

    if writer: writer.close()

    if have_truth:
        eval_dir = args.eval_output_dir or os.path.join(os.path.dirname(args.output_parquet), "infer_eval")
        os.makedirs(eval_dir, exist_ok=True)
        total = cm_sum.sum()
        acc = float(np.diag(cm_sum).sum() / total) if total>0 else 0.0
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump({"accuracy": acc}, f, indent=2)
        pd.DataFrame(cm_sum, index=["negative","neutral","positive"],
                     columns=["negative","neutral","positive"]).to_csv(os.path.join(eval_dir,"confusion.csv"))
        print(f"[infer+eval] Done: wrote={wrote} | acc={acc:.4f} | eval_dir={eval_dir}")
    else:
        print(f"[infer] Done: wrote={wrote}")

if __name__ == "__main__":
    main()



