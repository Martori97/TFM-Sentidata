import os
import pandas as pd

def convertir_a_parquet(origen, destino):
    if not os.path.exists(origen):
        print(f"La ruta de origen no existe: {origen}")
        return

    os.makedirs(destino, exist_ok=True)

    archivos = [f for f in os.listdir(origen) if f.endswith(".csv")]

    if not archivos:
        print(f"No se encontraron archivos CSV en {origen}")
        return

    for archivo in archivos:
        nombre_base = os.path.splitext(archivo)[0]
        ruta_entrada = os.path.join(origen, archivo)
        ruta_salida = os.path.join(destino, f"{nombre_base}.parquet")

        print(f"Convirtiendo {ruta_entrada} a {ruta_salida}")
        try:
            df = pd.read_csv(ruta_entrada, low_memory=False, dtype=str)
            df.to_parquet(ruta_salida, index=False)
        except Exception as e:
            print(f"Error al procesar {archivo}: {e}")

if __name__ == "__main__":
    convertir_a_parquet(
        origen="data/landing/sephora/raw",
        destino="data/landing/sephora/delta"
    )

    convertir_a_parquet(
        origen="data/landing/ulta/raw",
        destino="data/landing/ulta/delta"
    )
