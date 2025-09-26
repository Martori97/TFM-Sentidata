import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_parquet", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, required=True)
    parser.add_argument("--params_file", type=str, default="params.yaml")
    return parser.parse_args()

def load_params(params_file):
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    infer_params = params.get("infer_sentiment_full_pandas", {})
    return infer_params

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] using {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("[device] using CPU")
    return device

def load_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

def run_inference(df, tokenizer, model, device, batch_size, max_length, id_col, text_col):
    results = []
    batch = []
    batch_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inferencia"):
        review_id = row[id_col]
        review_text = row[text_col]
        batch.append(review_text)
        batch_ids.append(review_id)

        if len(batch) >= batch_size:
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

            for i in range(len(batch)):
                results.append({
                    id_col: batch_ids[i],
                    "pred_3": probs[i].argmax(),
                    "p_neg": probs[i][0],
                    "p_neu": probs[i][1],
                    "p_pos": probs[i][2],
                })

            batch = []
            batch_ids = []

    # Último batch si queda algo
    if batch:
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

        for i in range(len(batch)):
            results.append({
                id_col: batch_ids[i],
                "pred_3": probs[i].argmax(),
                "p_neg": probs[i][0],
                "p_neu": probs[i][1],
                "p_pos": probs[i][2],
            })

    return pd.DataFrame(results)

def main():
    args = parse_args()
    params = load_params(args.params_file)
    batch_size = params.get("batch_size", 128)
    max_length = params.get("max_length", 192)
    id_col = params.get("id_col", "review_id")
    text_col = params.get("text_col", "review_text_clean")

    print(f"[load] Cargando .parquet con solo columnas: {id_col}, {text_col}")
    df = pd.read_parquet(args.input_parquet, columns=[id_col, text_col])

    device = get_device()
    tokenizer, model = load_model(args.model_dir, device)

    print(f"[infer] batch_size={batch_size} | max_length={max_length}")
    df_pred = run_inference(df, tokenizer, model, device, batch_size, max_length, id_col, text_col)

    df_pred.to_parquet(args.output_parquet, index=False)
    print(f"[✅ done] Guardado: {args.output_parquet}")

if __name__ == "__main__":
    main()



