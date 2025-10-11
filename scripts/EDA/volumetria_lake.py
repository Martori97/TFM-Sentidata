#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Volumetría del lake (landing/trusted/exploitation) + FIX opcional de Parquet con TIMESTAMP(NANOS).
- Delta: detecta _delta_log
- Parquet: ficheros .parquet en nivel superior
- CSV: ficheros .csv en nivel superior
- Si hay TIMESTAMP(NANOS) y --fix-nanos:
  * convierte a microsegundos (PyArrow)
  * registra la nota en `fix_note`
  * deja `error=None`
"""

import argparse, os, sys, json, shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from pyspark.sql import SparkSession
from pyspark.sql import types as T

# Helpers FS
def human_size(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} EB"

def folder_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except FileNotFoundError:
                pass
    return total

def is_delta_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, "_delta_log"))

def has_files_top(path: str, exts: List[str]) -> bool:
    if not os.path.isdir(path):
        return any(path.endswith(ext) for ext in exts) and os.path.isfile(path)
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)) and any(f.endswith(ext) for ext in exts):
            return True
    return False

def detect_kind_strict(path: str) -> Optional[str]:
    if is_delta_dir(path):
        return "delta"
    if has_files_top(path, [".parquet"]):
        return "parquet"
    if has_files_top(path, [".csv"]):
        return "csv"
    return None

def find_datasets(roots: List[str]) -> List[str]:
    found = set()
    for root in roots:
        if not os.path.exists(root):
            continue
        k = detect_kind_strict(root)
        if k:
            found.add(root)
        for dirpath, dirnames, _ in os.walk(root):
            k = detect_kind_strict(dirpath)
            if k:
                found.add(dirpath)
            if is_delta_dir(dirpath):
                dirnames[:] = []
    return sorted(found)

# Spark
def build_spark(app_name="VolumetriaLake"):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

def try_read_dataset(spark, kind: str, path: str):
    if kind == "delta":
        return spark.read.format("delta").load(path)
    elif kind == "parquet":
        return spark.read.parquet(path)
    elif kind == "csv":
        return (spark.read.option("header", True).option("inferSchema", True).csv(path))
    else:
        raise ValueError(f"Tipo no soportado: {kind}")

def delta_versions(spark, path: str) -> Optional[int]:
    try:
        from delta.tables import DeltaTable
        dt = DeltaTable.forPath(spark, path)
        return dt.history().count()
    except Exception:
        return None

# PyArrow fix nanos → us
def _lazy_import_pyarrow():
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq

def _parquet_dest_path(src_path: str, inplace: bool, suffix: str) -> str:
    if inplace:
        return src_path
    if os.path.isdir(src_path):
        return src_path.rstrip("/") + suffix
    if src_path.endswith(".parquet"):
        base, ext = os.path.splitext(src_path)
        return f"{base}{suffix}{ext}"
    return src_path + suffix

def _convert_file_to_us(src_file: str, dest_file: str) -> bool:
    pa, pq = _lazy_import_pyarrow()
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    tbl = pq.read_table(src_file)
    needs_conv = any(pa.types.is_timestamp(f.type) and f.type.unit == "ns" for f in tbl.schema)
    if not needs_conv:
        if src_file != dest_file:
            shutil.copy2(src_file, dest_file)
        return False
    new_fields = []
    for f in tbl.schema:
        t = f.type
        if pa.types.is_timestamp(t) and t.unit == "ns":
            new_fields.append(pa.field(f.name, pa.timestamp("us", tz=t.tz)))
        else:
            new_fields.append(f)
    new_schema = pa.schema(new_fields)
    tbl2 = tbl.cast(new_schema, safe=False)
    pq.write_table(tbl2, dest_file, coerce_timestamps="us", allow_truncated_timestamps=True)
    return True

def _convert_dir_to_us(src_dir: str, dest_dir: str) -> bool:
    converted_any = False
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        out_root = os.path.join(dest_dir, rel) if rel != "." else dest_dir
        for f in files:
            if f.endswith(".parquet"):
                src_file = os.path.join(root, f)
                dest_file = os.path.join(out_root, f)
                ok = _convert_file_to_us(src_file, dest_file)
                converted_any = converted_any or ok
    return converted_any

def fix_parquet_path_to_us(src_path: str, inplace: bool=False, suffix: str="_us") -> Tuple[str,str]:
    pa, pq = _lazy_import_pyarrow()
    dest_path = _parquet_dest_path(src_path, inplace=inplace, suffix=suffix)
    if os.path.isfile(src_path) and src_path.endswith(".parquet"):
        ok = _convert_file_to_us(src_path, dest_path)
        return dest_path, ("converted nanos→us" if ok else "no nanos found")
    elif os.path.isdir(src_path):
        os.makedirs(dest_path, exist_ok=True)
        ok = _convert_dir_to_us(src_path, dest_path)
        return dest_path, ("converted nanos→us" if ok else "no nanos found")
    else:
        return src_path, "not parquet"

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["data/landing","data/trusted","data/exploitation"])
    parser.add_argument("--outdir", default="reports/volumetria")
    parser.add_argument("--no-count", action="store_true")
    parser.add_argument("--fix-nanos", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--suffix", default="_us")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    spark = build_spark()

    datasets = find_datasets(args.roots)
    results = []
    for path in datasets:
        kind = detect_kind_strict(path)
        size_b = folder_size_bytes(path)
        size_h = human_size(size_b)
        n_rows = n_cols = n_versions = None
        schema_json = None
        error = None
        fix_note = None
        used_path = path

        try:
            df = try_read_dataset(spark, kind, path)
        except Exception as e1:
            if kind=="parquet" and args.fix_nanos and "TIMESTAMP(NANOS" in str(e1):
                try:
                    fixed_path, note = fix_parquet_path_to_us(path, inplace=args.inplace, suffix=args.suffix)
                    df = try_read_dataset(spark, kind, fixed_path)
                    used_path = fixed_path
                    fix_note = note
                except Exception as e2:
                    error = f"fix failed: {e2}"
                    df = None
            else:
                error = str(e1)[:600]
                df = None

        if df is not None:
            try:
                n_cols = len(df.columns)
                if not args.no_count:
                    n_rows = df.count()
                if kind=="delta":
                    n_versions = delta_versions(spark, used_path)
                schema_json = json.dumps(df.schema.jsonValue())
            except Exception as e:
                error = str(e)[:600]

        results.append(dict(
            path=used_path,
            kind=kind,
            size_bytes=size_b,
            size_h=size_h,
            n_rows=n_rows,
            n_cols=n_cols,
            delta_versions=n_versions,
            schema_json=schema_json,
            error=error,
            fix_note=fix_note
        ))

    import pandas as pd
    pdf = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_out = os.path.join(args.outdir, f"volumetria_{ts}.csv")
    parquet_out = os.path.join(args.outdir, f"volumetria_{ts}.parquet")
    pdf.to_csv(csv_out, index=False)
    pdf.to_parquet(parquet_out, index=False)
    print(f"[ok] CSV: {csv_out}")
    print(f"[ok] Parquet: {parquet_out}")

if __name__=="__main__":
    main()

