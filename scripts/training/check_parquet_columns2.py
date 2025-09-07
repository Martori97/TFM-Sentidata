from pyspark.sql import SparkSession

# Crear SparkSession con soporte Delta
spark = (
    SparkSession.builder
    .appName("PreviewTrusted")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)

# Cargar el dataset trusted
df = spark.read.format("delta").load("data/trusted/sephora_clean/reviews_0_250")

# Mostrar las primeras 15 filas sin truncar columnas
df.show(15, truncate=False)

# Si quieres también ver las columnas disponibles
print("Columnas disponibles:", df.columns)

df.printSchema()


