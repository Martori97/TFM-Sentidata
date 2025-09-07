import pandas as pd
import glob

# ---- Landing (CSV) ----
landing_files = glob.glob("data/landing/sephora/raw/reviews_*.csv")
df_landing = pd.concat([pd.read_csv(f) for f in landing_files], ignore_index=True)
print("Landing total filas:", len(df_landing))
if "review_id" in df_landing.columns:
    print("Landing review_id únicos:", df_landing["review_id"].nunique())

# ---- Trusted (Delta / Parquet) ----
df_trusted = pd.read_parquet("data/trusted/sephora_clean/reviews_0_250")
print("Trusted total filas (subset 0_250):", len(df_trusted))
print("Trusted review_id únicos (subset):", df_trusted["review_id"].nunique())

# Si quieres TODO trusted concatenado (no solo 0_250):
trusted_paths = glob.glob("data/trusted/sephora_clean/reviews_*")
df_trusted_all = pd.concat([pd.read_parquet(p) for p in trusted_paths], ignore_index=True)
print("Trusted total filas (ALL):", len(df_trusted_all))
print("Trusted review_id únicos (ALL):", df_trusted_all["review_id"].nunique())


