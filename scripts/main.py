# -*- coding: utf-8 -*-
"""
MAIN ORQUESTADOR TFM-Sentidata

Flujo:
  1) Conversión CSV -> (Sephora: DELTA) | (Ulta: PARQUET)
  2) Limpieza TRUSTED (solo Sephora)
  3) Generación base de modelos (Delta canónico + Parquet y CSV deduplicados)
  4) (Opcional) Reporte HERA -> activar con RUN_HERA=1

Ejecución:
    python scripts/main.py

Opciones por variables de entorno:
    RUN_HERA=1            -> ejecuta el reporte HERA al final (por defecto 0)
    HALT_ON_FAIL=0        -> si pones 0, continúa aunque falle una etapa (por defecto 1)
"""

import os
import sys
import time
import subprocess
from datetime import datetime

# --------- Config ----------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable  # respeta el venv actual
RUN_HERA = os.environ.get("RUN_HERA", "0") == "1"
HALT_ON_FAIL = os.environ.get("HALT_ON_FAIL", "1") != "0"

STEPS = [
    {
        "title": "1) CSV -> (Sephora: DELTA) | (Ulta: PARQUET)",
        "cmd": [PYTHON, "scripts/data_cleaning/conversion_a_delta.py"],
    },
    {
        "title": "2) Limpieza TRUSTED (solo Sephora)",
        "cmd": [PYTHON, "scripts/data_cleaning/trusted_clean_driver.py"],
    },
    {
        "title": "3) Base modelos (Delta canónico + Parquet/CSV deduplicados)",
        "cmd": [PYTHON, "scripts/data_cleaning/filtrar_reviews_base.py"],
    },
]

HERA_STEP = {
    "title": "4) Reporte HERA (opcional)",
    "cmd": [PYTHON, "scripts/data_cleaning/hera.py"],
}


# --------- Helpers ----------
def banner(text):
    print("=" * 80)
    print(text)
    print("=" * 80)


def run_step(title, cmd, cwd):
    banner(title)
    print(">>>", " ".join(cmd))
    start = time.perf_counter()
    try:
        # heredamos el entorno actual (venv, etc.)
        subprocess.run(cmd, cwd=cwd, check=True)
        secs = time.perf_counter() - start
        print(f"[OK] {title} en {secs:0.1f}s\n")
        return True
    except subprocess.CalledProcessError as e:
        secs = time.perf_counter() - start
        print(f"[ERROR] {title} falló tras {secs:0.1f}s (returncode={e.returncode})\n")
        if HALT_ON_FAIL:
            sys.exit(e.returncode)
        return False


def main():
    start_all = time.perf_counter()
    print()
    print("TFM-Sentidata :: Orquestación del pipeline")
    print("Fecha/Hora:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Repo root :", REPO_ROOT)
    print()

    # 1..3
    for step in STEPS:
        run_step(step["title"], step["cmd"], REPO_ROOT)

    # 4 (opcional)
    if RUN_HERA:
        run_step(HERA_STEP["title"], HERA_STEP["cmd"], REPO_ROOT)
    else:
        print("[INFO] HERA desactivado (RUN_HERA=1 para activarlo)\n")

    total = time.perf_counter() - start_all
    banner(f"Pipeline COMPLETADO en {total/60.0:0.1f} min")
    print("Salidas de interés:")
    print("  - Delta canónico :", "data/exploitation/modelos_input/reviews_base_delta")
    print("  - Parquet dedup  :", "data/exploitation/modelos_input/reviews_base_parquet")
    print("  - CSV dedup      :", "data/exploitation/modelos_input/reviews_base_csv")
    print()


if __name__ == "__main__":
    main()
