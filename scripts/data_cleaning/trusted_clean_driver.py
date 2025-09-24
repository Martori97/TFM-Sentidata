import os
import subprocess

# Configura datasets y sus rutas
datasets = {
    "sephora": "data/landing/sephora/delta",
    }

for dataset, folder in datasets.items():
    print(f"Lanzando limpieza de: {dataset}")

    for tabla in os.listdir(folder):
        delta_path = os.path.join(folder, tabla)
        if not os.path.isdir(delta_path):
            continue
        print(f"[RUN] Limpieza Sephora: {name}")
        subprocess.run(
            ["python", "scripts/data_cleaning/trusted_clean_single.py", in_path, out_path],
            check=False,
        )

if __name__ == "__main__":
    run()
