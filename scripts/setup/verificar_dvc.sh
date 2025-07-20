#!/bin/bash

echo "=== Verificación de DVC en el proyecto ==="

# Verificar existencia de dvc.yaml
if [ -f "dvc.yaml" ]; then
    echo "[OK] dvc.yaml encontrado"
else
    echo "[ERROR] dvc.yaml no encontrado"
    exit 1
fi

# Mostrar grafo del pipeline (opcional)
#echo -e "\n[→] Grafo del pipeline (dvc dag):"
#dvc dag || { echo "[ERROR] Error al generar el DAG"; exit 1; }

# Comprobar que los datos en landing existen
echo -e "\n[→] Verificando datos convertidos a Delta en zona landing:"
if [ -d "data/landing/sephora/delta" ] && [ -d "data/landing/ulta/delta" ]; then
    find data/landing/sephora/delta -type f -name "*.parquet" | head -n 3
    find data/landing/ulta/delta -type f -name "*.parquet" | head -n 3
    echo "[OK] Archivos Delta encontrados en landing"
else
    echo "[ERROR] Faltan carpetas en data/landing"
    exit 1
fi

# Comprobar que los datos en trusted existen
echo -e "\n[→] Verificando datos en zona trusted:"
if [ -d "data/trusted/sephora_clean" ] && [ -d "data/trusted/ulta_clean" ]; then
    find data/trusted/sephora_clean -type f | head -n 3
    find data/trusted/ulta_clean -type f | head -n 3
    echo "[OK] Archivos encontrados en trusted"
else
    echo "[ERROR] Faltan carpetas en data/trusted"
    exit 1
fi

# Verificar si están ignorados por Git
echo -e "\n[→] Verificando .gitignore:"
if grep -q "data/landing/" .gitignore && grep -q "data/trusted/" .gitignore; then
    echo "[OK] data/landing y data/trusted están en .gitignore"
else
    echo "[ADVERTENCIA] Revisa que data/landing y data/trusted estén correctamente ignorados"
fi

# Verificar estado del pipeline
echo -e "\n[→] Estado del pipeline (dvc status):"
dvc status

echo -e "\n[FIN] Verificación completa."
