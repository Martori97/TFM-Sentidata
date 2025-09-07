#!/bin/bash
set -e

echo "=== Ejecutando pipeline con DVC (actualizado) ==="

# 1) Descarga Kaggle -> data/landing/*/raw
echo "1. Stage: descarga_kaggle_reviews"
dvc stage add --force -n descarga_kaggle_reviews \
  -d scripts/data_ingestion/descarga_kaggle_reviews.py \
  -o data/landing/sephora/raw \
  -o data/landing/ulta/raw \
  python scripts/data_ingestion/descarga_kaggle_reviews.py

# 2) CSV -> Delta -> data/landing/*/delta
echo "2. Stage: conversion_a_delta"
dvc stage add --force -n conversion_a_delta \
  -d scripts/data_cleaning/conversion_a_delta.py \
  -d data/landing/sephora/raw \
  -d data/landing/ulta/raw \
  -o data/landing/sephora/delta \
  -o data/landing/ulta/delta \
  python scripts/data_cleaning/conversion_a_delta.py

# 3) Limpieza avanzada (trusted) con Spark + UDF
#    Añadimos utils_text.py como dependencia explícita
echo "3. Stage: trusted_cleaning"
dvc stage add --force -n trusted_cleaning \
  -d scripts/data_cleaning/trusted_clean_driver.py \
  -d scripts/data_cleaning/trusted_clean_single.py \
  -d scripts/data_cleaning/utils_text.py \
  -d data/landing/sephora/delta \
  -d data/landing/ulta/delta \
  -o data/trusted/sephora_clean \
  -o data/trusted/ulta_clean \
  python scripts/data_cleaning/trusted_clean_driver.py

# 4) Construcción del dataset base para modelos (Delta + CSV)
#    Esta etapa lee por defecto sephora (DATASET=sephora). Si quieres ulta también,
#    crea un stage paralelo con DATASET=ulta y otras carpetas de salida.
echo "4. Stage: build_model_input"
dvc stage add --force -n build_model_input \
  -d scripts/data_cleaning/filtrar_reviews_base.py \
  -d data/trusted/sephora_clean \
  -o data/exploitation/modelos_input/reviews_base \
  -o data/exploitation/modelos_input/reviews_base_csv \
  python scripts/data_cleaning/filtrar_reviews_base.py

echo "5. Ejecutando dvc repro para construir el pipeline completo..."
dvc repro

echo "6. Añadiendo metadatos a Git..."
git add dvc.yaml dvc.lock .gitignore
git commit -m "Pipeline DVC actualizado: descarga, conversión, trusted cleaning, build_model_input"

echo "Pipeline DVC ejecutado y versionado correctamente."
