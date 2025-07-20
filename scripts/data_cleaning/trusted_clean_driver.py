import os
import subprocess

# Configura datasets y sus rutas
datasets = {
    "sephora": "data/landing/sephora/delta",
    "ulta": "data/landing/ulta/delta"
}

for dataset, folder in datasets.items():
    print(f"Lanzando limpieza de: {dataset}")

    for tabla in os.listdir(folder):
        delta_path = os.path.join(folder, tabla)
        if not os.path.isdir(delta_path):
            continue

        print(f"Procesando tabla: {tabla}")

        comando = [
            "python",
            "scripts/data_cleaning/trusted_clean_single.py",
            dataset,
            tabla
        ]

        result = subprocess.run(comando, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"{tabla} procesada con éxito.")
        else:
            print(f"Error procesando {tabla}:\n{result.stderr}")

