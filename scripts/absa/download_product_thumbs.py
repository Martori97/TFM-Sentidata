#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera miniaturas WebP (256px) desde image_url en el JSONL de cards.
Estrategia robusta anti-403:
  1) fetch directo (UA de navegador, proxy si está en env)
  2) reintento con Referer específico por host (Sephora, Clinique, etc.)
  3) fallback vía images.weserv.nl (proxy público) -> evita hotlink/CDN blocks

Entradas:
  - reports/absa/cards/product_cards_with_images.jsonl

Salidas:
  - data/cache/product_images/thumbs/{product_id}.webp
  - reports/absa/images/product_thumbs.csv (appends)

Requisitos:
  pip install pillow requests "requests[socks]"
"""

import os, io, csv, json, time, argparse, warnings
from urllib.parse import urlparse, quote
import requests
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

JSONL_IN   = "reports/absa/cards/product_cards_with_images.jsonl"
THUMBS_DIR = "data/cache/product_images/thumbs"
OUT_CSV    = "reports/absa/images/product_thumbs.csv"

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome Safari"
)

# Referers razonables por dominio (amplía si ves otros dominios en tus fails)
HOST_REF_MAP = {
    "sephora.com": "https://www.sephora.com/",
    "sephora.fr": "https://www.sephora.fr/",
    "clinique.com": "https://www.clinique.com/",
    "fresh.com": "https://www.fresh.com/",
    "qvc.com": "https://www.qvc.com/",
    "feelunique.com": "https://www.feelunique.com/",
    "ecosmetics.com": "https://www.ecosmetics.com/",
    "walmartimages.com": "https://www.walmart.com/",
}

def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)

def load_cards(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_csv_header_if_missing(path):
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["product_id","product_name","brand_name","image_url","status","local_path","error"])
            w.writeheader()

def append_rows(path, rows):
    if not rows: return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["product_id","product_name","brand_name","image_url","status","local_path","error"])
        for r in rows: w.writerow(r)

def open_image_bytes(data):
    """Abre imagen y normaliza modo/transparencia."""
    img = Image.open(io.BytesIO(data))
    if img.mode == "P" and ("transparency" in img.info or img.getbands() == ("P",)):
        try:
            img = img.convert("RGBA")
        except Exception:
            img = img.convert("RGB")
    if img.mode in ("RGBA", "LA"):
        # compón sobre blanco (si no quieres alfa en el thumb)
        bg = Image.new("RGB", img.size, (255, 255, 255))
        mask = img.split()[-1]
        bg.paste(img, mask=mask)
        img = bg.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img

def fetch_bytes(session, url, timeout, referer=None):
    """Descarga bytes con UA y referer opcional."""
    headers = {"User-Agent": DEFAULT_UA}
    if referer:
        headers["Referer"] = referer
    r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content

def weserv_proxy_url(raw_url, w=512, h=512):
    """
    Construye URL de proxy images.weserv.nl (gratis).
    - Se envía sin esquema y con parámetros de resize básicos.
    """
    # strip esquema
    parsed = urlparse(raw_url)
    host_path = f"{parsed.netloc}{parsed.path}"
    if parsed.query:
        host_path += f"?{parsed.query}"
    # codifica todo
    return f"https://images.weserv.nl/?url={quote(host_path, safe='')}&w={w}&h={h}&fit=inside&we&s=0"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")  # opcional
    args = ap.parse_args()

    # settings (admiten override por env)
    jsonl_in  = os.getenv("THUMBS_INPUT_JSONL", JSONL_IN)
    thumbs_dir= os.getenv("THUMBS_DIR", THUMBS_DIR)
    out_csv   = os.getenv("THUMBS_OUT_CSV", OUT_CSV)
    size      = int(os.getenv("THUMBS_SIZE", "256"))
    quality   = int(os.getenv("THUMBS_QUALITY", "80"))
    timeout   = int(os.getenv("THUMBS_TIMEOUT", "20"))
    sleep_sec = float(os.getenv("THUMBS_SLEEP", "0.2"))

    # proxies (Tor u otros), heredados del entorno
    proxies = {
        "http": os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
        "https": os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"),
    }
    proxies = {k:v for k,v in proxies.items() if v}

    assert os.path.exists(jsonl_in), f"No existe {jsonl_in}"
    cards = load_cards(jsonl_in)
    ensure_dir(thumbs_dir)
    write_csv_header_if_missing(out_csv)

    session = requests.Session()
    if proxies:
        session.proxies.update(proxies)

    ok = skip = fail = 0
    rows_log = []

    for c in cards:
        pid   = str(c.get("product_id") or "").strip()
        pname = (c.get("product_name") or "").strip()
        bname = (c.get("brand_name") or "").strip()
        url   = (c.get("image_url") or "").strip()

        dest = os.path.join(thumbs_dir, f"{pid}.webp") if pid else None

        if not pid or not dest:
            rows_log.append({"product_id": pid,"product_name":pname,"brand_name":bname,"image_url":url,
                             "status":"fail","local_path":"","error":"no_product_id"})
            fail += 1
            continue

        if os.path.exists(dest):
            rows_log.append({"product_id": pid,"product_name":pname,"brand_name":bname,"image_url":url,
                             "status":"skip","local_path":dest,"error":""})
            skip += 1
            continue

        if not url or not url.startswith("http"):
            rows_log.append({"product_id": pid,"product_name":pname,"brand_name":bname,"image_url":url,
                             "status":"fail","local_path":"","error":"no_url"})
            fail += 1
            continue

        err_msg = ""
        data = None

        # 1) intento directo
        try:
            data = fetch_bytes(session, url, timeout)
        except Exception as e1:
            err_msg = f"direct:{e1}"

        # 2) intento con Referer por host (si 1 falló)
        if data is None:
            try:
                host = urlparse(url).netloc.lower()
                # normaliza host base para mapping
                base = host.split(":")[0]
                referer = None
                for k,v in HOST_REF_MAP.items():
                    if base.endswith(k):
                        referer = v
                        break
                if referer:
                    data = fetch_bytes(session, url, timeout, referer=referer)
                else:
                    # reintento genérico con Google como referer
                    data = fetch_bytes(session, url, timeout, referer="https://www.google.com/")
            except Exception as e2:
                err_msg += f" | ref:{e2}"

        # 3) fallback vía images.weserv.nl (si 1 y 2 fallaron)
        if data is None:
            try:
                proxy = weserv_proxy_url(url, w=size*2, h=size*2)
                data = fetch_bytes(session, proxy, timeout)
            except Exception as e3:
                err_msg += f" | weserv:{e3}"

        if data is None:
            rows_log.append({"product_id": pid,"product_name":pname,"brand_name":bname,"image_url":url,
                             "status":"fail","local_path":"","error":err_msg[:500]})
            fail += 1
            time.sleep(sleep_sec)
            continue

        # Guardar thumb
        try:
            img = open_image_bytes(data)
            img.thumbnail((size, size))
            img.save(dest, "WEBP", quality=quality)
            rows_log.append({"product_id": pid,"product_name":pname,"brand_name":bname,"image_url":url,
                             "status":"ok","local_path":dest,"error":""})
            ok += 1
        except Exception as e:
            rows_log.append({"product_id": pid,"product_name":pname,"brand_name":bname,"image_url":url,
                             "status":"fail","local_path":"","error":f"save:{e}"[:500]})
            fail += 1

        time.sleep(sleep_sec)

    append_rows(out_csv, rows_log)
    print(f"[ok] thumbs -> {ok} | [skip] existentes -> {skip} | [fail] -> {fail}")
    if not os.listdir(thumbs_dir):
        print(f"WARNING: '{thumbs_dir}' is empty.")

if __name__ == "__main__":
    main()
