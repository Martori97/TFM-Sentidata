# scripts/data_prep/convert_delta_to_parquet_snapshot.py

import argparse
from pyspark.sql import SparkSession

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .appName("DeltaToParquetSnapshot")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

    print(f"[🔄] Leyendo Delta desde {args.input_path}")
    df = spark.read.format("delta").load(args.input_path)

    print(f"[💾] Guardando snapshot como Parquet en {args.output_path}")
    df.write.mode("overwrite").parquet(args.output_path)

    print("[✅] Conversión completada con éxito.")

if __name__ == "__main__":
    main()
