import pandas as pd
import os

def show_table(path, name, n=5):
    if not os.path.exists(path):
        print(f"⚠️ {name}: ruta no encontrada {path}")
        return
    df = pd.read_parquet(path)
    print(f"\n=== {name.upper()} ({path}) ===")
    print("N filas:", len(df))
    print("Columnas:", df.columns.tolist())
    print("\nPrimeras filas:")
    print(df.head(n).to_string())
    return df

if __name__ == "__main__":
    path_test = "data/exploitation/modelos_input/sample_tvt/test"
    path_full = "data/trusted/reviews_full"

    df_test = show_table(path_test, "test", n=5)
    df_full = show_table(path_full, "full", n=5)

    # Opcional: exportar a CSV para ver en Excel/VSCode
    out_dir = "reports/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    if df_test is not None:
        df_test.head(100).to_csv(os.path.join(out_dir, "test_preview.csv"), index=False)
    if df_full is not None:
        df_full.head(100).to_csv(os.path.join(out_dir, "full_preview.csv"), index=False)
    print(f"\n✅ Exportadas vistas previas a {out_dir}/test_preview.csv y full_preview.csv")