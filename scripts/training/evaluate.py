import pandas as pd
import pickle
import mlflow
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ruta al modelo entrenado
modelo_path = "models/model_lr_price.pkl"

# Cargar el modelo
with open(modelo_path, "rb") as f:
    model = pickle.load(f)

# Cargar datos
df = pd.read_parquet("data/trusted/sephora_clean/reviews_0_250")
df = df[["price_usd", "rating"]].dropna()
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["price_usd", "rating"])

# Preparar conjunto de evaluación
X = df[["price_usd"]]
y = df["rating"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Registrar métricas en MLflow
mlflow.set_tracking_uri("file:./models/mlflow_runs")
mlflow.set_experiment("sentidata_regresion_lineal")

with mlflow.start_run(run_name="evaluacion_post_entrenamiento"):
    mlflow.log_param("evaluacion", "con_modelo_entrenado")
    mlflow.log_metric("mse_eval", mse)
    mlflow.log_metric("r2_score_eval", r2)

    print(f"Evaluación completada con modelo guardado en: {modelo_path}")
    print(f"MSE: {mse:.4f} | R2: {r2:.4f}")
