# -*- coding: utf-8 -*-
# scripts/data_cleaning/_bootstrap.py
# Asegura que utils_text.py sea visible en driver y executors.

import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "scripts", "data_cleaning")

def setup_path_and_distribute(spark=None):
    # Para el driver (import local)
    if UTILS_DIR not in sys.path:
        sys.path.insert(0, UTILS_DIR)
    # Para los executors (distribuye el archivo)
    if spark is not None:
        utils_file = os.path.join(UTILS_DIR, "utils_text.py")
        if os.path.isfile(utils_file):
            spark.sparkContext.addPyFile(utils_file)
