Sentidata — Trabajo de Fin de Máster (UPC Big Data, Machine Learning & AI)

Autores: Pedro Vidal, David Martori y Judith Martínez  
Máster: Big Data, Machine Learning & Artificial Intelligence — UPC School  
Año académico: 2025  
Título del proyecto: Sentidata: Análisis de sentimiento y aspectos en reseñas de productos de cosmética y fragancias mediante arquitectura Delta Lakehouse y modelos de lenguaje (BERT / ALBERT)


1. Objetivo del Proyecto

El proyecto Sentidata tiene como objetivo diseñar un sistema escalable de análisis de reseñas de productos cosméticos que permita:
- Analizar el sentimiento global de los consumidores.  
- Detectar los aspectos específicos (fragrance, texture, price, packaging, etc.) que influyen en ese sentimiento.  
- Extraer insights de negocio útiles para marketing e innovación de producto.

El enfoque combina técnicas de procesamiento del lenguaje natural (NLP) con una arquitectura Lakehouse basada en Delta Lake y Spark, asegurando reproducibilidad mediante DVC (Data Version Control).


2. Arquitectura Técnica

El proyecto se estructura en tres niveles del data lakehouse:

Capa: Landing (Bronze)
Carpeta: data/landing/
Descripción: Datos brutos descargados de Kaggle (Sephora y Ulta).

Capa: Trusted (Silver)
Carpeta: data/trusted/
Descripción: Datos limpiados, normalizados y convertidos a Delta Lake.

Capa: Exploitation (Gold)
Carpeta: data/exploitation/
Descripción: Datasets listos para modelado, visualización y análisis ABSA.

Tecnologías principales:
- Apache Spark 3.5.3 + Delta Lake 3.1.0  
- Python 3.10 + PySpark + Transformers (HuggingFace)  
- DVC para la orquestación de pipelines y versionado de datos  
- MLflow para trazabilidad de experimentos y modelos  
- Altair y HTML reports para visualización final


3. Fases Analíticas

Fase 1 — Sentiment Analysis
- Entrenamiento de clasificadores base (TF-IDF + SVM / Random Forest).  
- Fine-tuning de modelos ALBERT multiclase (positivo, neutro, negativo).  
- Inferencia sobre todo el corpus (1.1M reseñas).  
- Evaluación mediante matrices de confusión, F1-score y análisis por categoría.

Fase 2 — Aspect-Based Sentiment Analysis (ABSA)
- Extracción de aspect terms mediante reglas y modelos ATE.  
- Mapeo de aspectos a una ontología de producto (fragancia, textura, precio, etc.).  
- Cálculo del sentimiento por aspecto y generación de informes HTML.  
- Visualización de resultados mediante tarjetas de producto y dashboards comparativos.


4. Reproducibilidad y Ejecución

Requisitos previos:
- Ubuntu 22.04 (WSL o Linux nativo)  
- Python ≥ 3.10  
- Java ≥ 11  
- Spark 3.5.3 con Delta 3.1.0  
- DVC ≥ 3.0  
- MLflow ≥ 2.14  


Instalación básica:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

5. Estructura del Proyecto

TFM-Sentidata/
│
├── data/
│   ├── landing/            # Datos brutos (Kaggle, etc.)
│   ├── trusted/            # Datos limpios y en Delta Lake
│   └── exploitation/       # Datasets listos para modelado
│
├── scripts/
│   ├── ingest/             # Ingesta de datos
│   ├── cleaning/           # Limpieza y transformación
│   ├── training/           # Entrenamiento y evaluación
│   ├── infer/              # Inferencia y predicciones
│   ├── absa/               # Aspect-based sentiment analysis
│   └── viz/                # Visualizaciones y dashboards
│
├── reports/
│   ├── sentiment_albert/   # Resultados y métricas
│   ├── absa/               # Product cards, dashboards, etc.
│   └── eda/                # Exploratory Data Analysis
│
├── params.yaml             # Parámetros globales del pipeline
├── dvc.yaml                # Definición de etapas DVC
├── requirements.txt        # Dependencias Python
└── README.txt              # Este documento


6. Resultados y Conclusiones

- El modelo ALBERT logra un 89 % de accuracy global, con F1 ponderado de 0.88.  
- El sistema ABSA permite visualizar los atributos más valorados por los consumidores y detectar oportunidades de mejora en gamas específicas.  
- El enfoque Lakehouse y DVC garantiza reproducibilidad, escalabilidad y trazabilidad del proceso analítico.


7. Futuras Líneas de Trabajo

- Integración con fuentes sociales (Reddit, TikTok, Instagram) y e-retailers (Amazon).  
- Despliegue del pipeline como API o servicio SaaS para equipos de innovación.


8. Referencias

- Delta Lake Documentation — https://docs.delta.io  
- HuggingFace Transformers — https://huggingface.co/transformers  
- DVC Docs — https://dvc.org/doc  
- MLflow — https://mlflow.org  


© 2025 — Pedro Vidal, David Martori y Judith Martínez. Trabajo de Fin de Máster (UPC Big Data, Machine Learning & AI).