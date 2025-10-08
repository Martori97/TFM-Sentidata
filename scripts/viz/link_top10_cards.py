#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera un índice HTML que une Top-10 (global y por secondary category) con las product cards.

Entradas:
- --per_category_csv: CSV con Top-10 por categoría (ej: reports/loves_top10/per_category_top10.csv)
- --global_csv: CSV con Top-10 global (ej: reports/loves_top10/global_top10.csv)
- --products_parquet: snapshot de productos para completar brand/nombre/rating/loves si faltan
- --cards_dir: carpeta donde están las cards HTML (ej: reports/absa/cards)
- --thumbs_dir: carpeta con thumbnails (ej: data/cache/product_images)
- --out_html: salida HTML (ej: reports/absa/cards/top10_index.html)
- --card_pattern: patrón del nombre de la card; por defecto "{product_id}.html"

El HTML resultante incluye:
- Selector de categoría (Global + cada secondary category con Top-10).
- Grid de 10 productos por categoría, cada uno enlazando a su card.
"""

import os
import argparse
import pandas as pd
from html import escape

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_thumb(thumbs_dir: str, product_id: str) -> str | None:
    if not thumbs_dir or not os.path.isdir(thumbs_dir):
        return None
    # nombre exacto
    for ext in IMG_EXTS:
        p = os.path.join(thumbs_dir, f"{product_id}{ext}")
        if os.path.exists(p):
            return p
    # comodín {product_id}_*.ext
    try:
        for f in os.listdir(thumbs_dir):
            fn = f.lower()
            if f.startswith(product_id) and fn.endswith(IMG_EXTS):
                return os.path.join(thumbs_dir, f)
    except Exception:
        pass
    return None


def load_top_csv(path: str, scope: str) -> pd.DataFrame:
    """
    Carga CSV robustamente y normaliza columnas:
    - product_id (obligatoria)
    - category (si falta y scope=="global", usa "GLOBAL")
    - brand, product_name, loves, rating (opcionales; si faltan, se crean vacías)
    """
    cols_target = ["product_id", "category", "brand", "product_name", "loves", "rating"]

    if not path or not os.path.exists(path):
        # devuelve todas las columnas esperadas vacías
        return pd.DataFrame(columns=cols_target)

    df = pd.read_csv(path)

    # normalización básica (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}

    def col_any(cands):
        for c in cands:
            if c in cols_lower:
                return cols_lower[c]
        return None

    pid = col_any(["product_id", "pid", "id"])
    if pid is None:
        raise ValueError(f"{path}: no encuentro columna product_id")
    df = df.rename(columns={pid: "product_id"})

    cat = col_any(["category", "secondary_category", "pinfo_secondary_category"])
    if cat:
        df = df.rename(columns={cat: "category"})
    else:
        df["category"] = "GLOBAL" if scope == "global" else None

    br = col_any(["brand", "brand_name"])
    if br:
        df = df.rename(columns={br: "brand"})
    pn = col_any(["product_name", "name", "title"])
    if pn:
        df = df.rename(columns={pn: "product_name"})
    lv = col_any(["loves", "loves_count", "pinfo_loves_count"])
    if lv:
        df = df.rename(columns={lv: "loves"})
    rt = col_any(["rating", "avg_rating"])
    if rt:
        df = df.rename(columns={rt: "rating"})

    # Asegura todas las columnas objetivo (si faltan, créalas vacías)
    for c in cols_target:
        if c not in df.columns:
            df[c] = None

    return df[cols_target]


def enrich_from_snapshot(df: pd.DataFrame, products_parquet: str) -> pd.DataFrame:
    if not products_parquet or not os.path.exists(products_parquet) or df.empty:
        return df

    snap = pd.read_parquet(products_parquet)

    # Mapea columnas útiles si existen
    map_cols = {}
    for src, dst in [
        ("pinfo_brand_name", "brand"),
        ("brand_name", "brand"),
        ("pinfo_product_name", "product_name"),
        ("product_name", "product_name"),
        ("pinfo_secondary_category", "category"),
        ("secondary_category", "category"),
        ("pinfo_loves_count", "loves"),
        ("rating", "rating"),
        ("pinfo_rating", "rating"),
    ]:
        if src in snap.columns and dst not in map_cols:
            map_cols[src] = dst

    use_cols = ["product_id"] + list(map_cols.keys())
    snap2 = (
        snap[[c for c in use_cols if c in snap.columns]]
        .drop_duplicates("product_id")
        .rename(columns=map_cols)
    )

    out = df.merge(snap2, on="product_id", how="left", suffixes=("", "_snap"))

    # completa nulos del csv con snapshot
    for c in ["brand", "product_name", "category", "loves", "rating"]:
        if c + "_snap" in out.columns:
            out[c] = out[c].fillna(out[c + "_snap"])
            out.drop(columns=[c + "_snap"], inplace=True)

    return out


def build_index_html(
    per_cat: pd.DataFrame,
    global_df: pd.DataFrame,
    cards_dir: str,
    thumbs_dir: str,
    card_pattern: str,
    out_html: str,
):
    # categorías en orden alfabético, Global primero si existe
    categories = []
    if not global_df.empty:
        categories.append("GLOBAL")
    if not per_cat.empty:
        cats = sorted(
            [
                c
                for c in per_cat["category"].dropna().unique().tolist()
                if c and c != "GLOBAL"
            ]
        )
        categories.extend(cats)

    def card_url(product_id: str) -> str | None:
        fname = card_pattern.format(product_id=product_id)
        path = os.path.join(cards_dir, fname)
        return path if os.path.exists(path) else None

    def row_to_tile(r) -> str:
        pid = str(r["product_id"])
        name = (
            escape(str(r["product_name"]))
            if pd.notna(r.get("product_name"))
            else pid
        )
        brand = escape(str(r["brand"])) if pd.notna(r.get("brand")) else ""
        loves = r.get("loves")
        rating = r.get("rating")

        img = find_thumb(thumbs_dir, pid)
        img_tag = (
            f'<img src="{img}" alt="{name}" />'
            if img
            else '<div class="ph">No image</div>'
        )

        href = card_url(pid)
        title = f"{brand} — {name}" if brand else name
        meta = []
        if pd.notna(rating):
            try:
                meta.append(f'⭐ {float(rating):.2f}')
            except Exception:
                pass
        if pd.notna(loves):
            try:
                meta.append(f'❤ {int(float(loves)):,}')
            except Exception:
                pass
        meta_txt = " · ".join(meta) if meta else ""

        a_attrs = (
            f'href="{href}" target="_blank"'
            if href
            else 'style="pointer-events:none; opacity:.7;"'
        )
        tile = f"""
        <a class="tile" {a_attrs} title="{title}">
          <div class="thumb">{img_tag}</div>
          <div class="txt">
            <div class="name">{title}</div>
            <div class="meta">{meta_txt}</div>
          </div>
        </a>"""
        return tile

    # HTML
    html_head = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Top-10 · Cards</title>
<style>
  :root { --gap: 14px; --tile-w: 220px; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
  header { position: sticky; top:0; background:#fff; z-index:10; border-bottom:1px solid #eee; padding:12px 16px; display:flex; gap:10px; align-items:center; }
  h1 { font-size:16px; margin:0 10px 0 0; }
  #catSel { padding:6px 10px; font-size:14px; }
  main { padding:16px; }
  section { margin-bottom:28px; }
  h2 { font-size:18px; margin:8px 0 12px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(var(--tile-w), 1fr)); gap: var(--gap); }
  .tile { display:block; border:1px solid #eee; border-radius:12px; overflow:hidden; text-decoration:none; color:#111; background:#fff; }
  .tile:hover { box-shadow:0 2px 10px rgba(0,0,0,.08); transform: translateY(-1px); }
  .thumb { width:100%; aspect-ratio: 1 / 1; display:grid; place-items:center; background:#fafafa; }
  .thumb img { max-width:100%; max-height:100%; object-fit:contain; }
  .thumb .ph { font-size:12px; color:#999; }
  .txt { padding:10px; }
  .name { font-weight:600; font-size:14px; line-height:1.3; margin-bottom:4px; }
  .meta { font-size:12px; color:#666; }
  .hidden { display:none; }
</style>
</head>
<body>
<header>
  <h1>Top-10 · Cards</h1>
  <label for="catSel">Secondary category:</label>
  <select id="catSel"></select>
</header>
<main>
"""

    html_sections = []

    # Global
    if not global_df.empty:
        tiles = "\n".join(
            global_df.head(10).apply(row_to_tile, axis=1).tolist()
        )
        html_sections.append(
            f"""
<section id="sec-GLOBAL">
  <h2>Global Top-10</h2>
  <div class="grid">{tiles}</div>
</section>
"""
        )

    # Por categoría
    if not per_cat.empty:
        for cat in [c for c in categories if c != "GLOBAL"]:
            d = per_cat[per_cat["category"] == cat].head(10)
            if d.empty:
                continue
            tiles = "\n".join(d.apply(row_to_tile, axis=1).tolist())
            html_sections.append(
                f"""
<section id="sec-{escape(cat)}" class="hidden">
  <h2>{escape(cat)}</h2>
  <div class="grid">{tiles}</div>
</section>
"""
            )

    html_tail = f"""
</main>
<script>
  const cats = {categories!r};
  const sel = document.getElementById('catSel');
  cats.forEach(c => {{
    const op = document.createElement('option');
    op.value = c; op.textContent = (c === "GLOBAL" ? "Global Top-10" : c);
    sel.appendChild(op);
  }});
  sel.value = cats[0];

  function show(cat) {{
    cats.forEach(c => {{
      const sec = document.getElementById('sec-'+c);
      if (!sec) return;
      sec.classList.toggle('hidden', c !== cat);
    }});
  }}
  sel.addEventListener('change', e => show(e.target.value));
  show(cats[0]);
</script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_head + "\n".join(html_sections) + html_tail)
    print(f"[OK] HTML -> {out_html}")


def main():
    ap = argparse.ArgumentParser("Top-10 index linking to product cards")
    ap.add_argument("--per_category_csv", required=True)
    ap.add_argument("--global_csv", required=True)
    ap.add_argument("--products_parquet", required=True)
    ap.add_argument("--cards_dir", required=True)
    ap.add_argument("--thumbs_dir", default="")
    ap.add_argument("--out_html", required=True)
    ap.add_argument("--card_pattern", default="{product_id}.html")
    args = ap.parse_args()

    per_cat = load_top_csv(args.per_category_csv, scope="percat")
    glob = load_top_csv(args.global_csv, scope="global")

    # Completa con snapshot (brand/product_name/category/loves/rating si faltan)
    per_cat = enrich_from_snapshot(per_cat, args.products_parquet)
    glob = enrich_from_snapshot(glob, args.products_parquet)

    # Garantiza columnas esperadas (por si snapshot tampoco las tiene)
    for df in (per_cat, glob):
        for c in ["brand", "product_name", "category", "loves", "rating"]:
            if c not in df.columns:
                df[c] = None

    build_index_html(
        per_cat, glob, args.cards_dir, args.thumbs_dir, args.card_pattern, args.out_html
    )


if __name__ == "__main__":
    main()


