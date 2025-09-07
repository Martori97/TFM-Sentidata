#!/bin/bash
set -e

REMOTE_NAME="modelstore"
FOLDER_ID="https://drive.google.com/drive/folders/1dxDU51WC26xP5UwdeopM10PsLNFrDVJW?dmr=1&ec=wgc-drive-globalnav-goto"

echo "=== Configurando remote DVC en Google Drive ==="

# Instala extra de DVC por si no estuviera
pip install "dvc[gdrive]"

# Añadir remote
dvc remote add -d $REMOTE_NAME gdrive://$FOLDER_ID || true

# Configuración: OAuth por usuario (no service account)
dvc remote modify $REMOTE_NAME gdrive_use_service_account false

echo "Remote $REMOTE_NAME configurado con carpeta $FOLDER_ID"

echo "Ejemplos de uso:"
echo " - Subir modelos: dvc push"
echo " - Descargar modelos: dvc pull"
echo " - Ver remotes: dvc remote list"
