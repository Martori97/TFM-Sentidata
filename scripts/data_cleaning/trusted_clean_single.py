# scripts/data_cleaning/trusted_clean_single.py
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.sql.functions import col, length, udf
from pyspark.sql.types import StringType

# Validación de argumentos
if len(sys.argv) != 3:
    print("Uso: python trusted_clean_single.py <dataset> <tabla>")
    sys.exit(1)

dataset = sys.argv[1]
tabla = sys.argv[2]

input_path = f"data/landing/{dataset}/delta/{tabla}"
output_path = f"data/trusted/{dataset}_clean/{tabla}"

# Spark con Delta
builder = (
    SparkSession.builder
    .appName(f"Cleaning {dataset}/{tabla}")
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Import local del utilitario (ruta del repo)
from scripts.data_cleaning.utils_text import clean_text  # noqa: E402

clean_udf = udf(clean_text, StringType())

try:
    print(f"[trusted_clean_single] Leyendo: {input_path}")
    df = spark.read.format("delta").load(input_path)

    print("[trusted_clean_single] Limpieza base: dropna + dropDuplicates")
    df = df.dropna(how="all").dropDuplicates()

    # Si existe columna review_text, creamos versión limpia + longitud
    if "review_text" in df.columns:
        print("[trusted_clean_single] Aplicando limpieza avanzada a 'review_text'...")
        df = df.withColumn("review_text_clean", clean_udf(col("review_text").cast("string")))
        df = df.withColumn("text_len", length(col("review_text_clean")))
        # filtros mínimos de calidad
        df = df.filter(col("review_text_clean").isNotNull() & (length(col("review_text_clean")) >= 3))
    else:
        print("[trusted_clean_single] Aviso: no existe columna 'review_text' en esta tabla.")

    print(f"[trusted_clean_single] Guardando en: {output_path}")
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .save(output_path)
    )

    total = df.count()
    print(f"[trusted_clean_single] Limpieza completada para {tabla} ({total} filas).")

except Exception as e:
    print(f"[trusted_clean_single] Error procesando {tabla}: {e}")
    raise
finally:
    spark.stop()
