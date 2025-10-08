#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera catálogo HTML de product cards con:
- Radar SVG (5 ejes fijos por categoría, sin relleno: producto vs benchmark por loves)
- Imagen del producto: usa miniatura local si existe, si no usa image_url
- Métricas clave (incluido Rating calculado desde reviews) y top términos
- (Opcional) Barras de "peso del aspecto" si el JSONL incluye 'mentions'

Entradas por defecto (parametrizables por CLI):
  - --input_jsonl  reports/absa/cards/product_cards_with_images.jsonl
  - --params       params.yaml  (para cards.axes_by_category / cards.axes_global)
  - --thumbs_dir   data/cache/product_images/thumbs

NUEVO (opcional, recomendado):
  - --ratings_input   data/trusted/reviews_product_info_clean_full   (Delta o Parquet)
  - --ratings_format  delta | parquet  (por defecto 'delta')
  - --rating_col      rating           (columna de estrellas en reviews)
  - --id_col          product_id       (clave de producto en reviews/cards)
  - --round_decimals  2

Salida (parametrizable):
  - --output_html reports/absa/cards/product_cards.html

Requisitos:
  pip install jinja2 pyyaml
  (si usas Delta) pip install delta-spark==3.1.0
"""

import os
import json
import math
import argparse
from pathlib import Path
from jinja2 import Template

try:
    import yaml
except Exception:
    yaml = None

DEFAULT_INPUT_JSONL = "reports/absa/cards/product_cards_with_images.jsonl"
DEFAULT_OUTPUT_HTML = "reports/absa/cards/product_cards.html"
DEFAULT_PARAMS_YAML = "params.yaml"
DEFAULT_THUMBS_DIR = "data/cache/product_images/thumbs"

def parse_args():
    p = argparse.ArgumentParser(
        description="Genera catálogo HTML de product cards y recalcula Rating promedio por product_id desde reviews."
    )
    p.add_argument("--params", default=DEFAULT_PARAMS_YAML, help="Ruta a params.yaml")
    p.add_argument("--input_jsonl", default=DEFAULT_INPUT_JSONL, help="JSONL con cards enriquecidas")
    p.add_argument("--output_html", default=DEFAULT_OUTPUT_HTML, help="Ruta de salida del HTML")
    p.add_argument("--thumbs_dir", default=DEFAULT_THUMBS_DIR, help="Carpeta de thumbnails locales (pid.webp)")

    # NUEVO: orígenes de ratings desde reviews
    p.add_argument("--ratings_input", default=None,
                   help="Ruta a reviews (Delta o Parquet). Si se proporciona, se calcula avg(rating) por product_id.")
    p.add_argument("--ratings_format", choices=["delta", "parquet"], default="delta",
                   help="Formato de --ratings_input (por defecto delta).")
    p.add_argument("--rating_col", default="rating", help="Columna de estrellas en las reviews.")
    p.add_argument("--id_col", default="product_id", help="Columna clave de producto.")
    p.add_argument("--round_decimals", type=int, default=2, help="Decimales a mostrar en Rating (por defecto 2).")
    return p.parse_args()

def load_params(path):
    if not path or not os.path.exists(path) or yaml is None:
        return {}, ["fragrance", "longevity", "texture", "packaging", "price"]
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cards_cfg = cfg.get("cards", {}) or {}
    axes_by_cat = cards_cfg.get("axes_by_category", {}) or {}
    axes_global = cards_cfg.get("axes_global", ["fragrance", "longevity", "texture", "packaging", "price"])
    return axes_by_cat, axes_global

def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def load_cards(path):
    cards = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # rating "de entrada" (puede ser sobreescrito por el recálculo)
            raw_rating = _coalesce(obj.get("rating"), obj.get("rating_avg_2d"), obj.get("rating_media"))
            try:
                rating_media = float(raw_rating) if raw_rating is not None else None
            except Exception:
                rating_media = None

            def _to_float(x, default=None):
                try:
                    return float(x) if x is not None else default
                except Exception:
                    return default

            def _to_int(x, default=0):
                try:
                    return int(x or 0)
                except Exception:
                    return default

            cards.append({
                "product_id": str(obj.get("product_id", "")),
                "product_name": obj.get("product_name") or "(sin nombre)",
                "brand_name": _coalesce(obj.get("brand_name"), obj.get("brand")) or "(sin marca)",
                "category2": obj.get("category2") or "(sin categoría)",
                "rating_media": rating_media,                             # float o None (será sobrescrito si hay recálculo)
                "price_usd": _to_float(obj.get("price_usd"), 0.0) or 0.0,
                "loves": _to_int(obj.get("loves"), 0),
                "reviews_analizadas": _to_int(obj.get("reviews_analizadas"), 0),
                "opinions": obj.get("opinion_distribution", {}) or {},
                "radar": obj.get("radar", {}) or {},
                "mentions": obj.get("mentions", {}) or {},
                "top_positive": obj.get("top_positive", []) or [],
                "top_negative": obj.get("top_negative", []) or [],
                "image_url": obj.get("image_url", "") or "",
            })
    return cards

# ---------- Radar helpers ----------
import math
def axes_for_category(cat, axes_by_cat, axes_global):
    axes = list(axes_by_cat.get(cat, axes_global))
    return axes[:5] if len(axes) >= 5 else axes + [""] * (5 - len(axes))

def _polar_to_cart(cx, cy, r, angle_deg):
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)

def _polyline_points(values, cx, cy, R):
    pts = []
    for val, ang in values:
        v = max(0.0, min(1.0, float(val)))
        x, y = _polar_to_cart(cx, cy, R * v, ang)
        pts.append(f"{x:.1f},{y:.1f}")
    if pts:
        pts.append(pts[0])
    return " ".join(pts)

def make_radar_svg(product_radar: dict, bench_radar: dict | None, axes: list, size: int = 160) -> str:
    w = h = size
    cx = cy = size / 2
    R = size * 0.38

    n = len(axes)
    if n == 0:
        return ""

    angles = [(i * 360.0 / n) - 90.0 for i in range(n)]

    prod_vals = []
    for ax, ang in zip(axes, angles):
        v = float(product_radar.get(ax, 0.0) or 0.0) if ax else 0.0
        prod_vals.append((v, ang))

    bench_vals = []
    if bench_radar:
        for ax, ang in zip(axes, angles):
            v = float(bench_radar.get(ax, 0.0) or 0.0) if ax else 0.0
            bench_vals.append((v, ang))

    prod_points = _polyline_points(prod_vals, cx, cy, R)
    bench_points = _polyline_points(bench_vals, cx, cy, R) if bench_vals else ""

    rings = ""
    for frac in (0.25, 0.5, 0.75, 1.0):
        rings += f'<circle cx="{cx}" cy="{cy}" r="{R*frac:.1f}" fill="none" stroke="#bbb" stroke-dasharray="2,2" stroke-opacity="0.6"/>\n'

    axes_lines = ""
    labels = ""
    label_R = R * 1.15
    for ax, ang in zip(axes, angles):
        x2, y2 = _polar_to_cart(cx, cy, R, ang)
        axes_lines += f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#ddd"/>\n'
        lx, ly = _polar_to_cart(cx, cy, label_R, ang)
        text = ax if ax else ""
        labels += f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10" text-anchor="middle" dominant-baseline="middle" fill="#333">{text}</text>\n'

    poly_prod = f'<polyline points="{prod_points}" fill="none" stroke="#1f77b4" stroke-width="2"/>'
    poly_bench = ""
    if bench_points:
        poly_bench = f'<polyline points="{bench_points}" fill="none" stroke="#555" stroke-width="2" stroke-dasharray="6,4" stroke-opacity="0.9"/>'

    svg = f"""<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="radar">
{rings}{axes_lines}{labels}{poly_bench}{poly_prod}
</svg>"""
    return svg

# ---------- Weights ----------
def make_weight_rows(axes, mentions: dict, total_reviews: int):
    if not mentions:
        return ""
    total = max(1, int(total_reviews or 0))
    rows = []
    for ax in axes:
        if not ax:
            continue
        m = int(mentions.get(ax, 0) or 0)
        pct = 100.0 * m / total if total else 0.0
        rows.append(f"""
          <div class="wrow">
            <div class="wlabel">{ax}</div>
            <div class="wbar"><div class="wfill" style="width:{pct:.0f}%;"></div></div>
            <div class="wval">{m} <span class="wpct">({pct:.0f}%)</span></div>
          </div>
        """)
    return "\n".join(rows)

# ---------- Thumbs rel path ----------
def local_thumb_relpath(pid: str, thumbs_dir: str, html_dir: Path) -> str:
    if not pid:
        return ""
    p = Path(thumbs_dir) / f"{pid}.webp"
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    if not p.exists():
        return ""
    try:
        return os.path.relpath(p, start=html_dir)
    except Exception:
        return str(p).replace("\\", "/")

# ---------- Rating recalculation from reviews ----------
def recompute_ratings_map(args) -> dict | None:
    """
    Devuelve dict: { product_id: avg_rating_float } calculado desde --ratings_input.
    Soporta Delta (por defecto) o Parquet. Requiere delta-spark si formato=delta.
    """
    if not args.ratings_input:
        return None

    # Lazy import para no exigir Spark si no se usa el recálculo
    from pyspark.sql import SparkSession, functions as F

    builder = (
        SparkSession.builder.appName("cards_rating_recalc")
        .config("spark.sql.shuffle.partitions", "200")
    )

    if args.ratings_format == "delta":
        # Habilitar Delta
        try:
            from delta import configure_spark_with_delta_pip
            builder = (builder
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
            spark = configure_spark_with_delta_pip(builder).getOrCreate()
        except Exception as e:
            raise RuntimeError(
                "No se pudo habilitar Delta. Instala 'delta-spark==3.1.0' o usa --ratings_format parquet."
            ) from e
        df = spark.read.format("delta").load(args.ratings_input)
    else:
        spark = builder.getOrCreate()
        # Admite fichero o carpeta Parquet
        df = spark.read.parquet(args.ratings_input)

    id_col = args.id_col
    rating_col = args.rating_col

    # Filtra ratings nulos y calcula media
    agg = (
        df.where(f"{rating_col} IS NOT NULL")
          .groupBy(id_col)
          .avg(rating_col)
          .withColumnRenamed(f"avg({rating_col})", "rating_avg")
    )

    # Colectar a dict (número de productos << número de reviews, debería caber en driver)
    pdf = agg.toPandas()  # columnas: [product_id, rating_avg]
    spark.stop()
    # Convierte NaN a None y a float normal
    ratings_map = {}
    for _, row in pdf.iterrows():
        pid = str(row[id_col])
        try:
            val = float(row["rating_avg"]) if row["rating_avg"] is not None else None
        except Exception:
            val = None
        if pid and val is not None:
            ratings_map[pid] = val
    return ratings_map

# ---------- Build HTML ----------
def build_html(cards, input_jsonl, output_html, thumbs_dir, axes_by_cat, axes_global, rating_override_map, round_decimals):
    out_html_path = Path(output_html)
    html_dir = out_html_path.parent
    os.makedirs(html_dir, exist_ok=True)

    # Si hay mapa de medias, sobrescribe rating_media de cada card
    if rating_override_map:
        for c in cards:
            pid = c["product_id"]
            if pid in rating_override_map:
                c["rating_media"] = rating_override_map[pid]

    # benchmark por categoría (máximo loves)
    best_by_cat = {}
    for c in cards:
        cat = c["category2"]
        if cat not in best_by_cat or c["loves"] > best_by_cat[cat]["loves"]:
            best_by_cat[cat] = c

    # enriquecer por card
    for c in cards:
        axes = axes_for_category(c["category2"], axes_by_cat, axes_global)
        bench = best_by_cat.get(c["category2"])
        bench_radar = bench["radar"] if bench else None
        c["radar_svg"] = make_radar_svg(c["radar"], bench_radar, axes, size=160)
        c["weights_html"] = make_weight_rows(axes, c.get("mentions", {}), c.get("reviews_analizadas", 0))
        c["local_thumb"] = local_thumb_relpath(c["product_id"], thumbs_dir, html_dir)

    # Plantilla (rating SIEMPRE con 'round_decimals' decimales)
    fmt = f"%.{round_decimals}f"
    template_str = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <title>Product Cards (ABSA)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <style>
    body {{ padding: 24px; background:#faf9f7; }}
    .card {{ margin-bottom: 20px; border-radius: 14px; }}
    .metric {{ font-size: 0.95rem; }}
    .neg {{ color: #d9534f; }} .neu {{ color: #6c757d; }} .pos {{ color: #28a745; }}
    .badge-pill {{ border-radius: 999px; }}
    .prod-img {{ width: 84px; height: 84px; border-radius: 8px; object-fit: contain; border: 1px solid #eee; background:#fff; }}

    .vizgrid {{ display: grid; grid-template-columns: 180px 1fr; gap: 14px; align-items: center; }}
    .radar-wrap {{ width: 160px; height: 160px; margin: 0 auto; }}

    .weights {{ font-size: 12px; }}
    .wrow {{ display: grid; grid-template-columns: 88px 1fr auto; gap: 8px; align-items: center; margin: 4px 0; }}
    .wlabel {{ color:#444; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .wbar {{ position: relative; height: 10px; background:#eee; border-radius: 6px; }}
    .wfill {{ position:absolute; left:0; top:0; bottom:0; background:#7aa7c7; border-radius: 6px; }}
    .wval {{ color:#222; }}
    .wpct {{ color:#666; }}

    .radar-wrap text {{ font-size: 10px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1 class="mb-1">Catálogo de productos (ABSA)</h1>
    <p class="text-muted">
      Fuente: {{{{ input_jsonl }}}} — {{{{ cards|length }}}} productos.
      Radar: producto (línea) vs benchmark por loves (línea discontinua). Ejes fijos (5) por categoría.
      {{% if cards|length > 0 and cards[0].weights_html %}}A la derecha del radar, peso de cada aspecto (nº reviews y %).{{% endif %}}
    </p>

    <div class="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-4">
      {{% for c in cards %}}
      <div class="col">
        <div class="card h-100 shadow-sm">
          <div class="card-body">

            <div class="d-flex justify-content-between align-items-start">
              <div class="pe-3">
                <h5 class="card-title mb-0">{{{{ c.product_name }}}}</h5>
                <p class="text-muted mb-1">{{{{ c.brand_name }}}}</p>
                <div class="d-flex gap-2">
                  <span class="badge text-bg-secondary">{{{{ c.category2 }}}}</span>
                  <span class="badge text-bg-light">ID: {{{{ c.product_id }}}}</span>
                </div>
              </div>
              {{% if c.local_thumb %}}
                <img class="prod-img" src="{{{{ c.local_thumb }}}}" alt="product image" />
              {{% elif c.image_url %}}
                <img class="prod-img" src="{{{{ c.image_url }}}}" alt="product image" />
              {{% endif %}}
            </div>

            <div class="metric mt-2 mb-1">
              <strong>Rating:</strong> {{{{ "{fmt}"|format(c.rating_media) if c.rating_media is not none else "—" }}}} ·
              <strong>Precio:</strong> {{{{ "{fmt}"|format(c.price_usd) }}}} USD ·
              <strong>Loves:</strong> {{{{ c.loves }}}} ·
              <strong>Reviews:</strong> {{{{ c.reviews_analizadas }}}}
            </div>

            {{% set pos = (c.opinions.get('pos', 0.0) or 0.0) * 100.0 %}}
            {{% set neu = (c.opinions.get('neu', 0.0) or 0.0) * 100.0 %}}
            {{% set neg = (c.opinions.get('neg', 0.0) or 0.0) * 100.0 %}}
            <div class="metric mb-2">
              <strong>Opiniones:</strong>
              <span class="pos">{{{{ "%.1f"|format(pos) }}}}% pos</span> ·
              <span class="neu">{{{{ "%.1f"|format(neu) }}}}% neu</span> ·
              <span class="neg">{{{{ "%.1f"|format(neg) }}}}% neg</span>
            </div>

            <div class="vizgrid mb-2">
              <div class="radar-wrap">{{{{ c.radar_svg | safe }}}}</div>
              {{% if c.weights_html %}}
                <div class="weights">{{{{ c.weights_html | safe }}}}</div>
              {{% endif %}}
            </div>

            <div class="mb-1">
              <strong>Top términos positivos:</strong>
              {{% if c.top_positive %}}
                {{% for t in c.top_positive %}}
                  <span class="badge text-bg-success badge-pill me-1">{{{{ t }}}}</span>
                {{% endfor %}}
              {{% else %}}<span class="text-muted">—</span>{{% endif %}}
            </div>

            <div class="mb-1">
              <strong>Top términos negativos:</strong>
              {{% if c.top_negative %}}
                {{% for t in c.top_negative %}}
                  <span class="badge text-bg-danger badge-pill me-1">{{{{ t }}}}</span>
                {{% endfor %}}
              {{% else %}}<span class="text-muted">—</span>{{% endif %}}
            </div>

          </div>
        </div>
      </div>
      {{% endfor %}}
    </div>
  </div>
</body>
</html>
"""
    html = Template(template_str).render(cards=cards, input_jsonl=input_jsonl)

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[ok] catálogo generado en {out_html_path}")

def main():
    args = parse_args()

    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(
            f"No existe {args.input_jsonl}. Ajusta --input_jsonl o ejecuta antes el pipeline que lo genera."
        )

    axes_by_cat, axes_global = load_params(args.params)
    cards = load_cards(args.input_jsonl)

    # (Opcional) Recalcular avg(rating) por product_id desde reviews y sobrescribir
    rating_map = recompute_ratings_map(args) if args.ratings_input else None

    build_html(
        cards=cards,
        input_jsonl=args.input_jsonl,
        output_html=args.output_html,
        thumbs_dir=args.thumbs_dir,
        axes_by_cat=axes_by_cat,
        axes_global=axes_global,
        rating_override_map=rating_map,
        round_decimals=args.round_decimals,
    )

if __name__ == "__main__":
    main()








