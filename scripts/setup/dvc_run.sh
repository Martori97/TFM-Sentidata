#!/usr/bin/env bash
set -euo pipefail

echo "=== Ejecutando/Definiendo pipeline DVC (actualizado) ==="

# Limpia stages anteriores para rehacerlos con la nueva realidad
# (Si no quieres tocar stages antiguos, comenta estas líneas)
dvc stage remove -n build_model_input      >/dev/null 2>&1 || true
dvc stage remove -n trusted_cleaning       >/dev/null 2>&1 || true
dvc stage remove -n conversion_a_delta     >/dev/null 2>&1 || true
dvc stage remove -n descarga_kaggle_reviews >/dev/null 2>&1 || true

# 1) (Opcional) Descarga Kaggle → landing/*/raw
echo "1. Stage: descarga_kaggle_reviews"
dvc stage add --force -n descarga_kaggle_reviews \
  -d scripts/data_ingestion/descarga_kaggle_reviews.py \
  -o data/landing/sephora/raw \
  -o data/landing/ulta/raw \
  python scripts/data_ingestion/descarga_kaggle_reviews.py

# 2) CSV -> (Sephora: DELTA) | (Ulta: PARQUET)
echo "2. Stage: conversion_a_delta"
dvc stage add --force -n conversion_a_delta \
  -d scripts/data_cleaning/conversion_a_delta.py \
  -d data/landing/sephora/raw \
  -d data/landing/ulta/raw \
  -o data/landing/sephora/delta \
  -o data/landing/ulta/parquet \
  python scripts/data_cleaning/conversion_a_delta.py

# 3) Limpieza avanzada (TRUSTED) SOLO Sephora
echo "3. Stage: trusted_cleaning (solo Sephora)"
dvc stage add --force -n trusted_cleaning \
  -d scripts/data_cleaning/trusted_clean_driver.py \
  -d scripts/data_cleaning/trusted_clean_single.py \
  -d scripts/data_cleaning/utils_text.py \
  -d data/landing/sephora/delta \
  -o data/trusted/sephora_clean \
  python scripts/data_cleaning/trusted_clean_driver.py

# 4) Construcción dataset base para modelos (Sephora)
#    Genera: Delta + Parquet + CSV (CSV sin columnas array)
echo "4. Stage: build_model_input (Delta + Parquet + CSV)"
dvc stage add --force -n build_model_input \
  -d scripts/data_cleaning/filtrar_reviews_base.py \
  -d data/trusted/sephora_clean \
  -o data/exploitation/modelos_input/reviews_base_delta \
  -o data/exploitation/modelos_input/reviews_base_parquet \
  -o data/exploitation/modelos_input/reviews_base_csv \
  python scripts/data_cleaning/filtrar_reviews_base.py

echo "5. Ejecutando 'dvc repro'..."
dvc repro

echo "6. Versionando metadatos de pipeline en Git..."
git add dvc.yaml dvc.lock .gitignore
git commit -m "DVC actualizado: Kaggle -> (Sephora: Delta / Ulta: Parquet) -> Trusted Sephora -> Base (Delta/Parquet/CSV)"

echo "Listo. Usa 'dvc push' para subir datos al remote y 'dvc pull' para recuperarlos."
