# TFM-Sentidata

Arquitectura Delta Lake para análisis de productos cosméticos.

CAMBIOS

en trusted_clean_single.py update:
    Resumen para tu equipo
    Validamos y tipamos rating (1–5).
    Normalizamos review_text (trim + lower + sin saltos de línea/tabs).
    Creamos review_id único y reproducible.
    Guardamos todas las columnas originales + review_id.
    Reporte de calidad con distribución de ratings y duplicados.
    root
    |-- _c0: string (nullable = true)
    |-- author_id: string (nullable = true)
    |-- rating: integer (nullable = true)
    |-- is_recommended: string (nullable = true)
    |-- helpfulness: string (nullable = true)
    |-- total_feedback_count: string (nullable = true)
    |-- total_neg_feedback_count: string (nullable = true)
    |-- total_pos_feedback_count: string (nullable = true)
    |-- submission_time: string (nullable = true)
    |-- review_text: string (nullable = true)
    |-- review_title: string (nullable = true)
    |-- skin_tone: string (nullable = true)
    |-- eye_color: string (nullable = true)
    |-- skin_type: string (nullable = true)
    |-- hair_color: string (nullable = true)
    |-- product_id: string (nullable = true)
    |-- product_name: string (nullable = true)
    |-- brand_name: string (nullable = true)
    |-- price_usd: string (nullable = true)
    |-- review_id: string (nullable = true)

├── README.md
├── data
│   ├── exploitation
│   │   ├── analisis_final
│   │   ├── dashboards_data
│   │   ├── modelos_input
│   │   └── products
│   ├── landing
│   │   ├── sephora
│   │   │   ├── delta
│   │   │   └── raw
│   │   └── ulta
│   │       ├── delta
│   │       └── raw
│   └── trusted
│       ├── sephora_clean
│       │   ├── product_info
│       │   ├── reviews_0_250
│       │   ├── reviews_1250_end
│       │   ├── reviews_250_500
│       │   ├── reviews_500_750
│       │   └── reviews_750_1250
│       └── ulta_clean
│           └── Ulta Skincare Reviews
├── dvc.lock
├── dvc.yaml
├── dvc.yaml:Zone.Identifier
├── mlruns
│   └── 0
│       ├── 8854a58292f04b77bef57111634c51b0
│       │   ├── artifacts
│       │   ├── meta.yaml
│       │   ├── metrics
│       │   ├── params
│       │   └── tags
│       └── meta.yaml
├── models
│   ├── albert_subset
│   ├── albert_subset_0_250
│   │   ├── checkpoint-11967
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── optimizer.pt
│   │   │   ├── rng_state.pth
│   │   │   ├── scheduler.pt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── spiece.model
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── trainer_state.json
│   │   │   └── training_args.bin
│   │   ├── checkpoint-6000
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── optimizer.pt
│   │   │   ├── rng_state.pth
│   │   │   ├── scheduler.pt
│   │   │   ├── special_tokens_map.json
│   │   │   ├── spiece.model
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── trainer_state.json
│   │   │   └── training_args.bin
│   │   ├── config.json
│   │   ├── confusion.png
│   │   ├── metrics.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── spiece.model
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── training_args.bin
│   ├── mlflow_runs
│   │   └── 0
│   │       └── meta.yaml
│   └── model_lr_price.pkl
├── notebooks
├── params.yaml
├── reports
│   ├── albert_subset_0_250
│   │   ├── confusion.png
│   │   ├── metrics.json
│   │   ├── predictions.parquet
│   │   └── test.parquet
│   ├── albert_subset_all
│   │   ├── confusion_all.png
│   │   ├── metrics_overall.json
│   │   ├── metrics_partitions.csv
│   │   ├── predictions_all.parquet
│   │   ├── reviews_0_250_preds.parquet
│   │   ├── reviews_1250_end_preds.parquet
│   │   ├── reviews_250_500_preds.parquet
│   │   ├── reviews_500_750_preds.parquet
│   │   └── reviews_750_1250_preds.parquet
│   └── figures
├── requirements.txt
├── scripts
│   ├── data_cleaning
│   │   ├── conversion_a_delta.py
│   │   ├── trusted_clean_driver.py
│   │   ├── trusted_clean_driver.py:Zone.Identifier
│   │   ├── trusted_clean_single.py
│   │   └── trusted_clean_single.py:Zone.Identifier
│   ├── data_ingestion
│   │   └── descarga_kaggle_reviews.py
│   ├── data_integration
│   ├── evaluation
│   │   ├── merge_parquets.py
│   │   ├── metrics_overall.py
│   │   └── metrics_partitions.py
│   ├── main.py
│   ├── main.py:Zone.Identifier
│   ├── setup
│   │   ├── dvc_run.sh
│   │   └── verificar_dvc.sh
│   ├── training
│   │   ├── check_parquet_columns2.py
│   │   ├── conteo.py
│   │   ├── evaluate_albert.py
│   │   ├── infer_albert_all.py
│   │   ├── prepare_test_parquet.py
│   │   └── train_albert.py
│   └── visualization
├── tests
└── tmp_trainer
