#!/bin/bash

echo "=== Ejecutando pipeline con DVC ==="

echo "1. Stage: descarga_kaggle_reviews"
dvc stage add --force -n descarga_kaggle_reviews \
  -d scripts/data_ingestion/descarga_kaggle_reviews.py \
  -o data/landing/sephora/raw \
  -o data/landing/ulta/raw \
  python scripts/data_ingestion/descarga_kaggle_reviews.py

echo "2. Stage: conversion_a_delta"
dvc stage add --force -n conversion_a_delta \
  -d scripts/data_cleaning/conversion_a_delta.py \
  -d data/landing/sephora/raw \
  -d data/landing/ulta/raw \
  -o data/landing/sephora/delta \
  -o data/landing/ulta/delta \
  python scripts/data_cleaning/conversion_a_delta.py

echo "3. Stage: trusted_cleaning"
dvc stage add --force -n trusted_cleaning \
  -d scripts/data_cleaning/trusted_clean_driver.py \
  -d scripts/data_cleaning/trusted_clean_single.py \
  -d data/landing/sephora/delta \
  -d data/landing/ulta/delta \
  -o data/trusted/sephora_clean \
  -o data/trusted/ulta_clean \
  python scripts/data_cleaning/trusted_clean_driver.py

echo "4. Ejecutando dvc repro para construir el pipeline completo..."
dvc repro

echo "5. Añadiendo metadatos a Git..."
git add dvc.yaml dvc.lock .gitignore
git commit -m "Pipeline completo: descarga, conversión y limpieza con DVC"

echo "Pipeline ejecutado y versionado correctamente."
