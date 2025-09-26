#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train ATE (Aspect Term Extraction) as token-classification (BIO) on English reviews.
Model: roberta-base. Labels: B-ASP, I-ASP, O.

Usage:
python scripts/absa/train_ate_bio.py \
  --train_path data/trusted/ate/train.jsonl \
  --valid_path data/trusted/ate/valid.jsonl \
  --model_name roberta-base \
  --output_dir models/ate_roberta_base \
  --epochs 3 --lr 2e-5 --batch_size 16 --grad_accum 2 --max_length 160 \
  --fp16 true --bf16 false --grad_checkpoint false --dataloader_workers 4
"""
import argparse, os, json, inspect
from typing import List, Dict
import numpy as np
import pandas as pd
import datasets as hfds
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)

# ----------------- labels -----------------
LABELS = ["O","B-ASP","I-ASP"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

# ----------------- io utils -----------------
def load_seq_tag(path: str) -> Dataset:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jsonl",".json"]:
        rows = []
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    elif ext == ".csv":
        df = pd.read_csv(path)
        for c in ["tokens","labels"]:
            df[c] = df[c].apply(lambda x: json.loads(x) if isinstance(x,str) else x)
    else:
        raise ValueError("Use .jsonl/.json or .csv")
    assert "tokens" in df.columns and "labels" in df.columns, "Missing tokens/labels"
    return Dataset.from_pandas(df[["tokens","labels"]], preserve_index=False)

def align_labels_with_tokens(tokenizer, tokens: List[str], labels: List[str], label2id: Dict[str,int], max_len: int):
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_len)
    word_ids = enc.word_ids()
    label_ids, prev_word_id = [], None
    for w_id in word_ids:
        if w_id is None:
            label_ids.append(-100)
        else:
            lab = labels[w_id]
            if w_id != prev_word_id:
                label_ids.append(label2id.get(lab, 0))
            else:
                if lab.startswith("B-"):
                    lab = "I-" + lab[2:]
                label_ids.append(label2id.get(lab, 0))
        prev_word_id = w_id
    return enc, label_ids

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    true_labels, pred_labels = [], []
    for pl, ll in zip(preds, labels):
        for p_i, l_i in zip(pl, ll):
            if l_i == -100: 
                continue
            true_labels.append(l_i)
            pred_labels.append(p_i)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)
    prec_asp, rec_asp, f1_asp, _ = precision_recall_fscore_support(
        [1 if l in (1,2) else 0 for l in true_labels],
        [1 if p in (1,2) else 0 for p in pred_labels],
        average="binary", zero_division=0
    )
    return {"precision_macro":float(prec),"recall_macro":float(rec),"f1_macro":float(f1),
            "precision_aspect":float(prec_asp),"recall_aspect":float(rec_asp),"f1_aspect":float(f1_asp)}

def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--valid_path", required=True)
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--output_dir", default="models/ate_roberta_base")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=160)
    # perf / precision flags
    ap.add_argument("--fp16", type=str, default="false")  # true|false
    ap.add_argument("--bf16", type=str, default="false")  # true|false (Ampere+)
    ap.add_argument("--grad_checkpoint", type=str, default="false")  # true|false
    ap.add_argument("--dataloader_workers", type=int, default=None)
    args = ap.parse_args()

    # ---------- device log ----------
    use_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    cc = torch.cuda.get_device_capability(0) if use_cuda else (0,0)
    print(f"[device] cuda={use_cuda} | name={dev_name} | cc={cc} | torch={torch.__version__}")
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    # ---------- tokenizer ----------
    tok_kwargs = {"use_fast": True}
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True, **tok_kwargs)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)

    # ---------- data ----------
    train_ds = load_seq_tag(args.train_path)
    valid_ds = load_seq_tag(args.valid_path)

    def map_enc(batch):
        out = {"input_ids":[], "attention_mask":[], "labels":[]}
        for tokens, labels in zip(batch["tokens"], batch["labels"]):
            enc, lab_ids = align_labels_with_tokens(tok, tokens, labels, LABEL2ID, args.max_length)
            out["input_ids"].append(enc["input_ids"])
            out["attention_mask"].append(enc["attention_mask"])
            out["labels"].append(lab_ids)
        return out

    # paraleliza el preprocesado si la versión de datasets lo soporta
    num_proc = args.dataloader_workers if args.dataloader_workers is not None else min(8, os.cpu_count() or 1)
    try:
        train_enc = train_ds.map(map_enc, batched=True, num_proc=num_proc, remove_columns=train_ds.column_names)
        valid_enc = valid_ds.map(map_enc, batched=True, num_proc=num_proc, remove_columns=valid_ds.column_names)
    except TypeError:
        train_enc = train_ds.map(map_enc, batched=True, remove_columns=train_ds.column_names)
        valid_enc = valid_ds.map(map_enc, batched=True, remove_columns=valid_ds.column_names)

    data_collator = DataCollatorForTokenClassification(tok)

    # ---------- model ----------
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    # gradient checkpointing (reduce VRAM, algo más lento)
    if str(args.grad_checkpoint).lower() == "true":
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("[opt] gradient checkpointing ENABLED")
        else:
            print("[opt] gradient checkpointing NOT supported in this version")

    # ---------- TrainingArguments (compatibles con versiones antiguas) ----------
    ta_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        logging_steps=50
    )
    sig = inspect.signature(TrainingArguments.__init__)
    steps_per_epoch = max(1, int(len(train_enc) / max(1, args.batch_size*args.grad_accum)))
    if "warmup_steps" in sig.parameters:
        ta_kwargs["warmup_steps"] = int(0.1 * steps_per_epoch * args.epochs)
    if "report_to" in sig.parameters:
        ta_kwargs["report_to"] = "none"
    # workers / pin memory
    if "dataloader_num_workers" in sig.parameters:
        ta_kwargs["dataloader_num_workers"] = num_proc
    if "dataloader_pin_memory" in sig.parameters:
        ta_kwargs["dataloader_pin_memory"] = True
    # precision
    want_bf16 = (str(args.bf16).lower() == "true") and use_cuda and (cc[0] >= 8)
    want_fp16 = (str(args.fp16).lower() == "true") and use_cuda and not want_bf16
    if "bf16" in sig.parameters:
        ta_kwargs["bf16"] = bool(want_bf16)
    if "fp16" in sig.parameters:
        ta_kwargs["fp16"] = bool(want_fp16)
    if want_bf16: print("[precision] using BF16")
    elif want_fp16: print("[precision] using FP16")
    else: print("[precision] using FP32")

    # optim más rápido si está
    if "optim" in sig.parameters:
        ta_kwargs["optim"] = "adamw_torch"

    # guarda poco para no llenar disco
    for k, v in {"save_total_limit": 2, "save_steps": 0}.items():
        if k in sig.parameters:
            ta_kwargs[k] = v

    tr_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_enc,
        eval_dataset=valid_enc,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("[eval]", metrics)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()



