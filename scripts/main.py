# scripts/main.py
"""
Pipeline principal de data_cleaning del TFM.

Orden:
0) Asegurar dependencias (pip install -r requirements.txt)
1) Descarga Kaggle -> data/landing/*/raw           (saltable con SKIP_KAGGLE=1)
2) CSV -> Delta     -> data/landing/*/delta
3) Limpieza trusted -> data/trusted/*_clean
4) Dataset base     -> data/exploitation/modelos_input
"""

import os
import subprocess
import sys
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = ROOT / "requirements.txt"

# Tareas del pipeline
STEPS = [
    ("Descarga Kaggle", "scripts/data_ingestion/descarga_kaggle_reviews.py"),
    ("Conversión a Delta", "scripts/data_cleaning/conversion_a_delta.py"),
    ("Limpieza trusted", "scripts/data_cleaning/trusted_clean_driver.py"),
    ("Dataset base", "scripts/data_cleaning/filtrar_reviews_base.py"),
]

def run_cmd(cmd, title=None):
    title = title or " ".join(cmd)
    print(f"\n=== {title} ===")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(f"ERROR en: {title}")
    if result.stdout.strip():
        print(result.stdout)

def ensure_requirements():
    if not REQ_FILE.exists():
        print(f"Aviso: no existe {REQ_FILE}, se omite instalación de dependencias.")
        return
    # Instala/actualiza dependencias del proyecto en el entorno actual
    run_cmd([sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)], title="Instalando dependencias (requirements.txt)")

def check_java():
    java_bin = which("java")
    if not java_bin:
        print("ADVERTENCIA: No se ha encontrado 'java' en PATH. Spark requiere JDK 11/17.")
        print("Instala Java y define JAVA_HOME antes de ejecutar etapas de Spark/Delta.\n")
    else:
        print(f"Java detectado: {java_bin}")

def find_script(rel_path: str) -> Path:
    p = ROOT / rel_path
    if p.exists():
        return p
    # Fallback: buscar por nombre dentro de scripts/
    name = Path(rel_path).name
    for cand in (ROOT / "scripts").rglob(name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No se encontró {rel_path} ni {name} dentro de scripts/")

def run_step(title: str, rel_path: str):
    # Opción para saltar Kaggle si no hay credenciales o se desea omitir
    if title.lower().startswith("descarga kaggle") and os.getenv("SKIP_KAGGLE", "0") == "1":
        print("Saltando 'Descarga Kaggle' por SKIP_KAGGLE=1")
        return
    script_path = find_script(rel_path)
    run_cmd([sys.executable, str(script_path)], title=f"{title}: {script_path}")

def main():
    print("Iniciando pipeline de data_cleaning")
    ensure_requirements()
    check_java()
    for title, rel in STEPS:
        run_step(title, rel)
    print("\nPipeline de data_cleaning finalizado correctamente")

if __name__ == "__main__":
    main()
