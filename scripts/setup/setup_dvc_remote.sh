#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME="modelstore"
# Usa SOLO el ID de la carpeta de Drive (no la URL completa)
FOLDER_ID="1dxDU51WC26xP5UwdeopM10PsLNFrDVJW"

echo "=== Configurando remote DVC en Google Drive ==="

# Instalar DVC con soporte GDrive
python -m pip install -q --upgrade "dvc[gdrive]>=3.0.0"

# Añadir/forzar remote por si ya existe
dvc remote add -d --force "$REMOTE_NAME" "gdrive://$FOLDER_ID"

# Autenticación por OAuth de usuario (no service account)
dvc remote modify "$REMOTE_NAME" gdrive_use_service_account false

# (Opcional) Evitar bloqueos por ficheros grandes
dvc remote modify "$REMOTE_NAME" gdrive_acknowledge_abuse true || true

echo "Remote '$REMOTE_NAME' configurado en carpeta $FOLDER_ID"
echo
echo "Comandos útiles:"
echo "  dvc remote list"
echo "  dvc push   # subir datos al remote"
echo "  dvc pull   # bajar datos del remote"
