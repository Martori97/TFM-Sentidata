import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Inicializar API
api = KaggleApi()
api.authenticate()
print("API autenticada correctamente")

# Lista de datasets a descargar
datasets = {
    "sephora": {
        "kaggle_id": "nadyinky/sephora-products-and-skincare-reviews",
        "dest": "data/landing/sephora/raw"
    },
    "ulta": {
        "kaggle_id": "nenamalikah/nlp-ulta-skincare-reviews",
        "dest": "data/landing/ulta/raw"
    }
}

# Descargar cada uno
for nombre, info in datasets.items():
    print(f"Descargando dataset de {nombre}...")
    os.makedirs(info["dest"], exist_ok=True)
    api.dataset_download_files(info["kaggle_id"], path=info["dest"], unzip=True)
    print(f"Dataset {nombre} descargado en {info['dest']}")

