#!/bin/bash

echo "=== Verificación de DVC en el proyecto ==="

# Verificar existencia de dvc.yaml
if [ -f "dvc.yaml" ]; then
    echo "[✔] dvc.yaml encontrado"
else
    echo "[✘] dvc.yaml no encontrado"
    exit 1
fi

# Ver grafo del pipeline
echo -e "\n[→] Grafo del pipeline (dvc dag):"
dvc dag || { echo "[✘] Error al generar el DAG"; exit 1; }

# Comprobar que los outputs existen
echo -e "\n[→] Archivos descargados desde Kaggle:"
if [ -d "data/landing/sephora/delta" ] && [ -d "data/landing/ulta/delta" ]; then
    ls -l data/landing/sephora/delta | grep .csv || echo "No hay archivos CSV"
    ls -l data/landing/ulta/delta | grep .csv || echo "No hay archivos CSV"
    echo "[✔] Directorios de datos existen"
else
    echo "[✘] No se encuentran los datos descargados"
    exit 1
fi

# Verificar si los datos están ignorados por Git
echo -e "\n[→] Verificando .gitignore:"
if grep -q "data/landing/" .gitignore; then
    echo "[✔] data/landing está en .gitignore"
else
    echo "[✘] data/landing no está ignorado por Git"
fi

# Verificar estado de DVC
echo -e "\n[→] dvc status:"
dvc status

# Prueba de reproducibilidad (comentado por seguridad)
# echo -e "\n[→] Eliminando CSV de Sephora para probar repro..."
# rm data/landing/sephora/delta/*.csv
# dvc repro

echo -e "\n [✔] Verificación completa."
