# fix_compiled_checkpoint.py
import os, json, sys

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/albert_sample_30pct"

def _rewrite_key(k: str) -> str:
    # Quita envoltorios de torch.compile y colapsa 'albert.albert.'
    nk = k.replace("._orig_mod.", ".")
    nk = nk.replace("albert._orig_mod.", "albert.")
    nk = nk.replace("albert.albert.", "albert.")
    return nk

def _fix_bin(path: str):
    import torch
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.4
    except TypeError:
        sd = torch.load(path, map_location="cpu")                      # fallback
    if not isinstance(sd, dict):
        raise ValueError(f"{path} no parece un state_dict")
    new_sd = { _rewrite_key(k): v for k, v in sd.items() }
    torch.save(new_sd, path)
    print(f"✅ Reescrito .bin → {os.path.basename(path)}")

def _fix_safetensors(path: str):
    try:
        from safetensors.torch import load_file, save_file
    except Exception:
        raise SystemExit("Necesitas instalar safetensors: pip install safetensors")
    sd = load_file(path)  # dict[str, tensor]
    new_sd = { _rewrite_key(k): v for k, v in sd.items() }
    save_file(new_sd, path)
    print(f"✅ Reescrito .safetensors → {os.path.basename(path)}")

def _gather_weights_files(model_dir: str):
    files = []
    # casos single-file
    for name in ["pytorch_model.bin", "model.safetensors"]:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            files.append(p)
    # casos sharded (+ detecta shards sin index)
    for ext in (".bin", ".safetensors"):
        # index.json → recoge nombres
        idx = os.path.join(model_dir, f"pytorch_model{ext}.index.json")
        if os.path.isfile(idx):
            with open(idx, "r", encoding="utf-8") as f:
                j = json.load(f)
            shard_names = sorted(set(j.get("weight_map", {}).values()))
            for s in shard_names:
                p = os.path.join(model_dir, s)
                if os.path.isfile(p):
                    files.append(p)
        # shards sueltos
        for fn in os.listdir(model_dir):
            if fn.endswith(ext) and "-of-" in fn:  # ej. pytorch_model-00001-of-00004.bin
                files.append(os.path.join(model_dir, fn))
    # dedup
    files = sorted(set(files))
    return files

def main():
    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(f"No existe el directorio: {MODEL_DIR}")
    files = _gather_weights_files(MODEL_DIR)
    if not files:
        raise SystemExit("No se encontraron pesos (.bin/.safetensors).")
    for p in files:
        if p.endswith(".bin"):
            _fix_bin(p)
        elif p.endswith(".safetensors"):
            _fix_safetensors(p)
    print("🎉 Listo. Vuelve a evaluar el modelo.")

if __name__ == "__main__":
    main()

