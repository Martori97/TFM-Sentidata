# scripts/data_cleaning/filtrar_reviews_base.py
import os
from pathlib import Path
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.sql.functions import col

# -----------------------
# Configuración
# -----------------------
DATASET = os.getenv("DATASET", "sephora")  # sephora | ulta
TRUSTED_BASE = Path(f"data/trusted/{DATASET}_clean")

SUBCARPETAS_DEFAULT = [
    "reviews_0_250",
    "reviews_250_500",
    "reviews_500_750",
    "reviews_750_1250",
    "reviews_1250_end",
]

ONLY_0_250 = os.getenv("ONLY_0_250", "0") == "1"

MAX_LEN_ENV = os.getenv("MAX_LEN")  # si quieres filtrar por longitud limpia
MAX_LEN = int(MAX_LEN_ENV) if MAX_LEN_ENV and MAX_LEN_ENV.isdigit() else None

DELTA_OUT = Path("data/exploitation/modelos_input/reviews_base")
CSV_OUT = Path("data/exploitation/modelos_input/reviews_base_csv")

# -----------------------
# Crear SparkSession
# -----------------------
builder = (
    SparkSession.builder.appName("FiltrarReviewsBase")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .master("local[*]")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

def main():
    subcarpetas = ["reviews_0_250"] if ONLY_0_250 else SUBCARPETAS_DEFAULT
    folders = [TRUSTED_BASE / sc for sc in subcarpetas if (TRUSTED_BASE / sc).exists()]
    assert folders, f"No se encontraron subcarpetas en {TRUSTED_BASE}"

    df_union = None
    for p in folders:
        df_tmp = spark.read.format("delta").load(str(p))
        # columnas disponibles
        cols_avail = [c for c in ["review_text", "review_text_clean", "text_len", "rating"] if c in df_tmp.columns]
        df_tmp = df_tmp.select(*cols_avail)
        df_union = df_tmp if df_union is None else df_union.unionByName(df_tmp, allowMissingColumns=True)

    # filtro por longitud limpia si se pide
    if MAX_LEN and "text_len" in df_union.columns:
        df_union = df_union.filter(col("text_len") <= MAX_LEN)

    # columnas mínimas
    cols_out = [c for c in ["review_text", "review_text_clean", "text_len", "rating"] if c in df_union.columns]
    df_union = df_union.select(*cols_out)

    # salida Delta
    DELTA_OUT.mkdir(parents=True, exist_ok=True)
    df_union.write.format("delta").mode("overwrite").save(str(DELTA_OUT))
    print(f"[filtrar_reviews_base] Guardado Delta en {DELTA_OUT} ({df_union.count()} filas)")

    # salida CSV espejo
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    (
        df_union.coalesce(1)
        .write.option("header", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", True)
        .option("encoding", "UTF-8")
        .option("delimiter", ",")
        .mode("overwrite")
        .csv(str(CSV_OUT))
    )
    print(f"[filtrar_reviews_base] Guardado CSV en {CSV_OUT}")

if __name__ == "__main__":
    try:
        main()
    finally:
        spark.stop()
