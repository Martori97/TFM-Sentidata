#!/usr/bin/env bash
set -euo pipefail

echo "=== Verificación de DVC / Estructura del proyecto ==="

# 1) dvc.yaml
if [[ -f "dvc.yaml" ]]; then
  echo "[OK] dvc.yaml encontrado"
else
  echo "[ERROR] dvc.yaml no encontrado"
  exit 1
fi

# 2) Landing procesado
echo -e "\n[→] Verificando landing procesado:"
SEPH_DELTA="data/landing/sephora/delta"
ULTA_PARQ="data/landing/ulta/parquet"

[[ -d "$SEPH_DELTA" ]] && echo "[OK] Sephora Delta: $SEPH_DELTA" || { echo "[ERROR] Falta $SEPH_DELTA"; exit 1; }
[[ -d "$ULTA_PARQ"  ]] && echo "[OK] Ulta Parquet: $ULTA_PARQ"   || echo "[WARN] Falta $ULTA_PARQ (no crítico)"

find "$SEPH_DELTA" -type f | head -n 3 || true
[[ -d "$ULTA_PARQ" ]] && find "$ULTA_PARQ" -type f | head -n 3 || true

# 3) Trusted (solo Sephora)
echo -e "\n[→] Verificando trusted (solo Sephora):"
TRUSTED_SEPH="data/trusted/sephora_clean"
if [[ -d "$TRUSTED_SEPH" ]]; then
  echo "[OK] $TRUSTED_SEPH"
  find "$TRUSTED_SEPH" -type f | head -n 3 || true
else
  echo "[ERROR] Falta $TRUSTED_SEPH"
  exit 1
fi

# 4) Exploitation (Delta + Parquet + CSV)
echo -e "\n[→] Verificando exploitation (Delta + Parquet + CSV):"
OUT_DELTA="data/exploitation/modelos_input/reviews_base_delta"
OUT_PARQ="data/exploitation/modelos_input/reviews_base_parquet"
OUT_CSV="data/exploitation/modelos_input/reviews_base_csv"

[[ -d "$OUT_DELTA" ]] && echo "[OK] Delta base: $OUT_DELTA" || echo "[WARN] Aún no existe $OUT_DELTA"
[[ -d "$OUT_PARQ"  ]] && echo "[OK] Parquet base: $OUT_PARQ" || echo "[WARN] Aún no existe $OUT_PARQ"
[[ -d "$OUT_CSV"   ]] && echo "[OK] CSV base: $OUT_CSV" || echo "[WARN] Aún no existe $OUT_CSV"

# 5) .gitignore (data y models deben ignorarse)
echo -e "\n[→] Verificando .gitignore:"
if grep -qE "^data/$" .gitignore && grep -qE "^models/$" .gitignore; then
  echo "[OK] data/ y models/ ignorados"
else
  echo "[WARN] Asegúrate de ignorar data/ y models/ en .gitignore"
fi

# 6) DVC status y remotes
echo -e "\n[→] dvc status:"
dvc status || true

echo -e "\n[→] dvc remote list:"
dvc remote list || true

echo -e "\n[FIN] Verificación completada."
