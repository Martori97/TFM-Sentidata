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
