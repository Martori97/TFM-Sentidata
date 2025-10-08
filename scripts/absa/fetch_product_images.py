#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_product_images.py

Busca imágenes para cada producto usando DuckDuckGo, evitando dominios que
dan 403 (blacklist) y priorizando dominios “amables” (whitelist). Graba:
- CSV con mapping product_id -> image_url, image_source, image_query
- JSONL de cards enriquecidas con image_url (solo si no tenían ya una buena)

Lee configuración desde params.yaml -> images:
  images:
    max_per_run: 50          # tope de productos por ejecución
    sleep_sec: 0.6           # pausa entre llamadas DDG
    timeout_sec: 20
    user_agent: "Mozilla/5.0 ..."
    out_csv: "reports/absa/images/product_images.csv"
    out_cards_jsonl: "reports/absa/cards/product_cards_with_images.jsonl"
    blacklist_domains: [ ... ]
    whitelist_domains:  [ ... ]   # se priorizan, luego el resto no blacklist
    debug_sample: 0

Respetará proxies del entorno (Tor):
  export HTTP_PROXY="socks5h://127.0.0.1:9050"
  export HTTPS_PROXY="socks5h://127.0.0.1:9050"

Requisitos:
  pip install requests "requests[socks]" beautifulsoup4 pyyaml
"""

import os, sys, csv, json, time, argparse, warnings, re
from urllib.parse import urlparse, quote
import requests
from bs4 import BeautifulSoup
import yaml

warnings.filterwarnings("ignore")

PARAMS_PATH = "params.yaml"
CARDS_IN    = "reports/absa/cards/product_cards.jsonl"              # base sin imágenes
CARDS_OUT   = "reports/absa/cards/product_cards_with_images.jsonl"  # enriquecido
IMAGES_CSV  = "reports/absa/images/product_images.csv"

DEFAULT_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"

VQD_PATTERNS = [
    re.compile(r'vqd\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'vqd=([^\&]+)\&', re.IGNORECASE),
    re.compile(r'vqd\s*:\s*["\']([^"\']+)["\']', re.IGNORECASE),
]

def load_params(path=PARAMS_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p):
    if p: os.makedirs(p, exist_ok=True)

def load_cards(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def save_cards(path, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_existing_csv(path):
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = str(row.get("product_id") or "").strip()
            if pid:
                d[pid] = row
    return d

def append_csv_rows(path, rows):
    new_file = not os.path.exists(path)
    ensure_dir(os.path.dirname(path))
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["product_id","product_name","brand_name","image_url","image_source","image_query"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def ddg_get_vqd(session, q, timeout, ua):
    headers = {"User-Agent": ua, "Referer":"https://duckduckgo.com/"}
    url = "https://duckduckgo.com/?q=" + requests.utils.quote(q) + "&iax=images&ia=images"
    r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    html = r.text
    for rx in VQD_PATTERNS:
        m = rx.search(html)
        if m: return m.group(1)
    soup = BeautifulSoup(html, "html.parser")
    for s in soup.find_all("script"):
        if s.string:
            for rx in VQD_PATTERNS:
                m = rx.search(s.string)
                if m: return m.group(1)
    return None

def ddg_search_images(session, q, timeout, ua):
    vqd = ddg_get_vqd(session, q, timeout, ua)
    if not vqd:
        return []
    headers = {"User-Agent": ua, "Referer":"https://duckduckgo.com/"}
    api = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={requests.utils.quote(q)}&vqd={requests.utils.quote(vqd)}&f=,,,&p=1"
    r = session.get(api, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    data = r.json()
    return data.get("results") or []

def domain(host_or_url):
    host = host_or_url
    if host_or_url.startswith("http"):
        host = urlparse(host_or_url).netloc
    host = host.split(":")[0].lower()
    return host

def base_domain(host):
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host

def pick_image_url(results, blacklist, whitelist):
    """
    Estrategia: primero máxima coincidencia con whitelist; si no hay, primera no-blacklist.
    Campo preferente: 'image'; si no, 'thumbnail'.
    """
    # 1) whitelist
    for it in results:
        for key in ("image","thumbnail"):
            url = it.get(key) or ""
            if not url.startswith("http"):
                continue
            host = domain(url)
            if any(host.endswith(w) for w in whitelist):
                return url, "ddg:whitelist"
    # 2) no-blacklist
    for it in results:
        for key in ("image","thumbnail"):
            url = it.get(key) or ""
            if not url.startswith("http"):
                continue
            host = domain(url)
            if any(host.endswith(b) for b in blacklist):
                continue
            return url, "ddg:first"
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=PARAMS_PATH)
    args = ap.parse_args()

    params = load_params(args.params)
    imgs   = params.get("images", {}) or {}
    max_per_run = int(imgs.get("max_per_run", 50))
    sleep_sec   = float(imgs.get("sleep_sec", 0.6))
    timeout     = int(imgs.get("timeout_sec", 20))
    ua          = imgs.get("user_agent") or DEFAULT_UA
    debug_sample= int(imgs.get("debug_sample", 0))
    out_csv     = imgs.get("out_csv", IMAGES_CSV)
    out_cards   = imgs.get("out_cards_jsonl", CARDS_OUT)

    # listas
    blacklist = set((imgs.get("blacklist_domains") or []))
    env_bl = os.getenv("THUMBS_BLACKLIST","")
    if env_bl:
        blacklist |= {d.strip().lower() for d in env_bl.split(",") if d.strip()}
    whitelist = set((imgs.get("whitelist_domains") or [
        "m.media-amazon.com",
        "i.ebayimg.com",
        "i5.walmartimages.com",
        "cdn.shopify.com",
        "images.ctfassets.net",
        "static.thcdn.com",
        "images.asos-media.com",
        "images-static.nicecdn.com",
        "target.scene7.com",
        "johnlewis.scene7.com",
        "images-na.ssl-images-amazon.com"
    ]))

    # proxies de entorno (Tor u otros)
    proxies = {
        "http": os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
        "https": os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
    }
    proxies = {k:v for k,v in proxies.items() if v}

    # carga cards base
    assert os.path.exists(CARDS_IN), f"No existe {CARDS_IN}"
    cards = load_cards(CARDS_IN)

    # mappings ya existentes (no duplicar)
    existing = read_existing_csv(out_csv)

    session = requests.Session()
    if proxies:
        session.proxies.update(proxies)
    session.headers.update({"User-Agent": ua})

    new_rows = []
    updated_cards = []

    # recorta por debug_sample si se pide
    if debug_sample > 0:
        cards_iter = cards[:debug_sample]
    else:
        cards_iter = cards

    processed = 0
    found = 0

    for c in cards_iter:
        pid   = str(c.get("product_id") or "").strip()
        pname = (c.get("product_name") or "").strip()
        bname = (c.get("brand_name") or "").strip()

        # si ya existe mapping previo y el dominio NO está en blacklist, mantenlo
        prev = existing.get(pid)
        prev_url = prev.get("image_url") if prev else ""
        keep_prev = False
        if prev_url:
            h = domain(prev_url)
            if not any(h.endswith(b) for b in blacklist):
                keep_prev = True

        # si la card ya tiene image_url y no es de blacklist, mantenla
        card_url = (c.get("image_url") or "").strip()
        keep_card = False
        if card_url:
            h = domain(card_url)
            if not any(h.endswith(b) for b in blacklist):
                keep_card = True

        if keep_prev or keep_card:
            updated_cards.append(c)
            continue

        # límite por ejecución
        if processed >= max_per_run:
            updated_cards.append(c)
            continue

        q = f"\"{pname}\" \"{bname}\"" if bname else f"\"{pname}\""
        try:
            results = ddg_search_images(session, q, timeout, ua)
            url, src = pick_image_url(results, blacklist, whitelist)
        except Exception as e:
            url, src = None, None

        if url:
            # asigna a card y agrega a CSV batch
            c["image_url"]   = url
            c["image_source"]= src or "ddg"
            c["image_query"] = q
            new_rows.append({
                "product_id": pid,
                "product_name": pname,
                "brand_name": bname,
                "image_url": url,
                "image_source": src or "ddg",
                "image_query": q
            })
            found += 1
        updated_cards.append(c)
        processed += 1
        time.sleep(sleep_sec)

    # guarda resultados
    if new_rows:
        append_csv_rows(out_csv, new_rows)
    ensure_dir(os.path.dirname(out_cards))
    save_cards(out_cards, updated_cards)

    print(f"[ok] procesados: {processed} | nuevos URLs: {found}")
    print(f"[ok] CSV: {out_csv}")
    print(f"[ok] cards enriquecidas: {out_cards}")

if __name__ == "__main__":
    main()




