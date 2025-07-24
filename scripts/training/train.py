import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configurar MLflow para guardar experimentos localmente en models/
mlflow.set_tracking_uri("file:./models/mlflow_runs")
mlflow.set_experiment("sentidata_regresion_lineal")

# Cargar los datos desde carpeta Delta exportada
df = pd.read_parquet("data/trusted/sephora_clean/reviews_0_250")

# Filtrar columnas relevantes y eliminar nulos
df = df[["price_usd", "rating"]].dropna()

# Forzar conversión a valores numéricos (descarta texto no convertible)
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Eliminar filas que quedaron con NaN tras la conversión
df = df.dropna(subset=["price_usd", "rating"])

# Definir variables de entrada y salida
X = df[["price_usd"]]
y = df["rating"]

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Registrar experimento en MLflow
with mlflow.start_run():

    mlflow.log_param("modelo", "LinearRegression")
    mlflow.log_param("feature", "price_usd")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Guardar el modelo como archivo .pkl
    output_path = "models/model_lr_price.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    # Registrar el modelo también en MLflow
    mlflow.sklearn.log_model(model, "modelo_lr_price")

    print(f"Modelo entrenado y guardado en: {output_path}")
    print(f"MSE: {mse:.4f} | R2: {r2:.4f}")
