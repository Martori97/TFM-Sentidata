from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import os

# -----------------------
# 1. Crear SparkSession
# -----------------------
builder = (
    SparkSession.builder.appName("FiltrarReviewsBase")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# -----------------------
# 2. Definir paths
# -----------------------
base_path = "data/trusted/sephora_clean"
delta_output_path = "data/exploitation/modelos_input/reviews_base"
csv_output_path = "data/exploitation/modelos_input/reviews_base_csv"

subcarpetas_reviews = [
    "reviews_0_250",
    "reviews_250_500",
    "reviews_500_750",
    "reviews_750_1250",
    "reviews_1250_end"
]

# -----------------------
# 3. Leer y unir todas las particiones
# -----------------------
df_union = None

for folder in subcarpetas_reviews:
    full_path = os.path.join(base_path, folder)
    df_tmp = spark.read.format("delta").load(full_path).select("review_text", "rating")
    df_union = df_tmp if df_union is None else df_union.unionByName(df_tmp)

# -----------------------
# 4. Guardar en formato Delta
# -----------------------
df_union.write.format("delta").mode("overwrite").save(delta_output_path)
print(f"Reviews combinadas guardadas en formato Delta: {delta_output_path}")

# -----------------------
# 5. Guardar también en formato CSV
# -----------------------
df_union.coalesce(1) \
    .write \
    .option("header", True) \
    .option("quote", '"') \
    .option("escape", '"') \
    .option("multiLine", True) \
    .option("encoding", "UTF-8") \
    .option("delimiter", ",") \
    .mode("overwrite") \
    .csv(csv_output_path)

print(f"Reviews combinadas guardadas en formato CSV: {csv_output_path}")

# -----------------------
# 6. Finalizar sesión Spark
# -----------------------
spark.stop()
