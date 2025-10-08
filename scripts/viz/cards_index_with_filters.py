#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Índice HTML filtrable de product cards:
- Filtros: brand, secondary category, price buckets (<$25, $25–$49, $50–$74, $75+)
- Enlaza a reports/absa/cards/{product_id}.html (configurable con --card_pattern)
- Sin iconos
"""

import os
import re
import json
import argparse
import pandas as pd
from html import escape

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# ---------- utils numéricos ----------
def as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def as_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

# ---------- precio ----------
def parse_price(x):
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x)
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None

def pick_price_row(row):
    # Prioridad: sale -> base -> value
    for c in ["pinfo_sale_price_usd", "pinfo_price_usd", "pinfo_value_price_usd", "price_usd"]:
        if c in row and pd.notna(row[c]):
            v = parse_price(row[c])
            if v is not None:
                return v
    return None

def price_bucket(p):
    if p is None: return "Unknown"
    if p < 25: return "<$25"
    if p < 50: return "$25–$49"
    if p < 75: return "$50–$74"
    return "$75+"

# ---------- thumbs ----------
def find_thumb(thumbs_dir, product_id):
    if not thumbs_dir or not os.path.isdir(thumbs_dir): return None
    for ext in IMG_EXTS:
        p = os.path.join(thumbs_dir, f"{product_id}{ext}")
        if os.path.exists(p): return p
    # comodín {product_id}_*.ext
    try:
        for f in os.listdir(thumbs_dir):
            fn = f.lower()
            if f.startswith(product_id) and fn.endswith(IMG_EXTS):
                return os.path.join(thumbs_dir, f)
    except Exception:
        pass
    return None

HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Product Cards · Filtros</title>
<style>
  :root { --gap: 14px; --tile-w: 240px; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color:#111; }
  header { position: sticky; top:0; background:#fff; z-index:10; border-bottom:1px solid #eee; padding:10px 16px; }
  .filters { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .filters label { font-size:13px; color:#333; }
  .filters select, .filters input { padding:6px 8px; font-size:14px; }
  .summary { margin-top:8px; font-size:13px; color:#555; }
  main { padding:16px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(var(--tile-w), 1fr)); gap: var(--gap); }
  .tile { display:block; border:1px solid #eee; border-radius:12px; overflow:hidden; text-decoration:none; color:#111; background:#fff; }
  .tile:hover { box-shadow:0 2px 10px rgba(0,0,0,.08); transform: translateY(-1px); }
  .thumb { width:100%; aspect-ratio: 1 / 1; display:grid; place-items:center; background:#fafafa; }
  .thumb img { max-width:100%; max-height:100%; object-fit:contain; }
  .thumb .ph { font-size:12px; color:#999; }
  .txt { padding:10px; }
  .name { font-weight:600; font-size:14px; line-height:1.3; margin-bottom:6px; }
  .brand { font-size:12px; color:#444; margin-bottom:4px; }
  .meta { font-size:12px; color:#666; }
  .muted { color:#999; }
  .hidden { display:none; }
</style>
</head>
<body>
<header>
  <div class="filters">
    <label>Secondary category:
      <select id="secSel"></select>
    </label>
    <label>Brand:
      <select id="brandSel"></select>
    </label>
    <label>Price:
      <select id="priceSel">
        <option value="__ALL__">All</option>
        <option value="<$25"><$25</option>
        <option value="$25–$49">$25–$49</option>
        <option value="$50–$74">$50–$74</option>
        <option value="$75+">$75+</option>
      </select>
    </label>
    <label>Search:
      <input id="q" type="search" placeholder="product or brand..." />
    </label>
  </div>
  <div class="summary"><span id="count"></span></div>
</header>
<main>
  <div id="grid" class="grid"></div>
</main>

<script>
  const DATA = __DATA__;
  const cardsDir = __CARDS_DIR__;
  const cardPattern = __CARD_PATTERN__;

  function uniqueSorted(arr) {
    return Array.from(new Set(arr.filter(x => x && x !== "Unknown"))).sort((a,b)=> a.localeCompare(b));
  }

  const secSel = document.getElementById('secSel');
  const brandSel = document.getElementById('brandSel');
  const priceSel = document.getElementById('priceSel');
  const searchBox = document.getElementById('q');
  const grid = document.getElementById('grid');
  const countEl = document.getElementById('count');

  // options
  const cats = uniqueSorted(DATA.map(d => d.category));
  const brands = uniqueSorted(DATA.map(d => d.brand));
  secSel.innerHTML = '<option value="__ALL__">All</option>' + cats.map(c => `<option value="${c}">${c}</option>`).join('');
  brandSel.innerHTML = '<option value="__ALL__">All</option>' + brands.map(b => `<option value="${b}">${b}</option>`).join('');

  function fmt(n, decimals=0) {
    if (n == null || isNaN(n)) return '';
    return Number(n).toLocaleString(undefined, {maximumFractionDigits: decimals});
  }

  function cardURL(pid) {
    const fname = cardPattern.replace('{product_id}', pid);
    return cardsDir + '/' + fname;
  }

  function tile(d) {
    const hasImg = !!d.thumb;
    const href = d.card_exists ? `href="${cardURL(d.product_id)}" target="_blank"` : 'style="pointer-events:none; opacity:.7;"';
    const price = d.price != null ? `$${fmt(d.price, 2)}` : 'Price: –';
    const rating = d.rating != null ? `Rating: ${Number(d.rating).toFixed(2)}` : 'Rating: –';
    const loves = d.loves != null ? `Loves: ${fmt(d.loves)}` : 'Loves: –';
    return `
      <a class="tile" ${href} title="${d.brand ? d.brand+' — ' : ''}${d.product_name}">
        <div class="thumb">${hasImg ? `<img src="${d.thumb}" alt="${d.product_name}">` : `<div class="ph">No image</div>`}</div>
        <div class="txt">
          <div class="name">${d.product_name}</div>
          <div class="brand">${d.brand || ''}</div>
          <div class="meta">${price} · ${rating} · ${loves}</div>
          <div class="meta muted">${d.category} · ${d.bucket}</div>
        </div>
      </a>`;
  }

  function applyFilters() {
    const cat = secSel.value;
    const brand = brandSel.value;
    const bucket = priceSel.value;
    const q = (searchBox.value || '').toLowerCase().trim();

    let rows = DATA;
    if (cat !== '__ALL__') rows = rows.filter(d => d.category === cat);
    if (brand !== '__ALL__') rows = rows.filter(d => d.brand === brand);
    if (bucket !== '__ALL__') rows = rows.filter(d => d.bucket === bucket);
    if (q) {
      rows = rows.filter(d =>
        (d.product_name && d.product_name.toLowerCase().includes(q)) ||
        (d.brand && d.brand.toLowerCase().includes(q))
      );
    }

    countEl.textContent = `${rows.length} products`;
    grid.innerHTML = rows.map(tile).join('');
  }

  secSel.addEventListener('change', applyFilters);
  brandSel.addEventListener('change', applyFilters);
  priceSel.addEventListener('change', applyFilters);
  searchBox.addEventListener('input', applyFilters);

  applyFilters();
</script>
</body>
</html>
"""

def main():
    ap = argparse.ArgumentParser("Cards index with filters (brand, secondary category, price)")
    ap.add_argument("--products_parquet", required=True)
    ap.add_argument("--cards_dir", required=True)
    ap.add_argument("--thumbs_dir", default="")
    ap.add_argument("--out_html", required=True)
    ap.add_argument("--card_pattern", default="{product_id}.html")
    args = ap.parse_args()

    snap = pd.read_parquet(args.products_parquet)

    # --- Resolver duplicados con precedencia pinfo_* ---
    # Si existen ambas columnas (p.ej., pinfo_rating y rating), eliminamos la "no pinfo_"
    def drop_if_both(df, preferred, fallback):
        if preferred in df.columns and fallback in df.columns:
            return df.drop(columns=[fallback])
        return df

    for pref, fb in [
        ("pinfo_product_name", "product_name"),
        ("pinfo_brand_name", "brand_name"),
        ("pinfo_secondary_category", "secondary_category"),
        ("pinfo_loves_count", "loves"),   # por si existiera una columna 'loves'
        ("pinfo_rating", "rating"),
    ]:
        snap = drop_if_both(snap, pref, fb)

    # --- Selección y renombrado seguro (sin duplicar destinos) ---
    colmap = {
        "product_id": "product_id",
        "pinfo_product_name": "product_name",
        "product_name": "product_name",
        "pinfo_brand_name": "brand",
        "brand_name": "brand",
        "pinfo_secondary_category": "category",
        "secondary_category": "category",
        "pinfo_loves_count": "loves",
        "pinfo_rating": "rating",
        "rating": "rating",
        # precios se mantienen con sus nombres originales para pick_price_row
        "pinfo_sale_price_usd": "pinfo_sale_price_usd",
        "pinfo_price_usd": "pinfo_price_usd",
        "pinfo_value_price_usd": "pinfo_value_price_usd",
        "price_usd": "price_usd",
    }

    # De colmap, nos quedamos solo con columnas existentes
    src_cols = [c for c in colmap.keys() if c in snap.columns]
    df2 = snap[src_cols].copy()
    # Renombra, ahora sin conflictos en rating/brand/product_name/category
    df2 = df2.rename(columns=colmap)

    # Asegura mínimos
    for c in ["product_id","product_name","brand","category","loves","rating",
              "pinfo_sale_price_usd","pinfo_price_usd","pinfo_value_price_usd","price_usd"]:
        if c not in df2.columns:
            df2[c] = None

    # Precio y bucket
    df2["price"] = df2.apply(pick_price_row, axis=1)
    df2["bucket"] = df2["price"].apply(price_bucket)
    df2["product_name"] = df2["product_name"].fillna(df2["product_id"])
    df2["category"] = df2["category"].fillna("Skincare")

    # Thumbs + card_exists
    thumbs, card_exists = [], []
    for pid in df2["product_id"].astype(str):
        thumbs.append(find_thumb(args.thumbs_dir, pid))
        path = os.path.join(args.cards_dir, args.card_pattern.format(product_id=pid))
        card_exists.append(os.path.exists(path))
    df2["thumb"] = thumbs
    df2["card_exists"] = card_exists

    # Exporta data JSON embebida (con cast robusto)
    records = []
    for _, r in df2.drop_duplicates("product_id").iterrows():
        records.append({
            "product_id": str(r["product_id"]),
            "product_name": escape(str(r["product_name"])) if pd.notna(r["product_name"]) else str(r["product_id"]),
            "brand": escape(str(r["brand"])) if pd.notna(r["brand"]) else None,
            "category": escape(str(r["category"])) if pd.notna(r["category"]) else "Skincare",
            "price": as_float(r["price"]),
            "bucket": r["bucket"],
            "rating": as_float(r["rating"]),
            "loves": as_int(r["loves"]),
            "thumb": r["thumb"],
            "card_exists": bool(r["card_exists"]),
        })

    html = (HTML
            .replace("__DATA__", json.dumps(records))
            .replace("__CARDS_DIR__", json.dumps(args.cards_dir))
            .replace("__CARD_PATTERN__", json.dumps(args.card_pattern)))

    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] HTML -> {args.out_html}  (rows={len(records):,})")

if __name__ == "__main__":
    main()

