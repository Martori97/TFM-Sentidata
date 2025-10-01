# -*- coding: utf-8 -*-
"""
Funciones de limpieza + tokenización + "lematización" heurística (sin dependencias).
Pensado para velocidad y robustez en Spark UDFs.
"""

import re
import unicodedata

# Regex útiles
_URL_RE = re.compile(r"""https?://\S+|www\.\S+""", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_EMOJI_RE = re.compile(
    "[\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transport & map symbols
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251]+",
    flags=re.UNICODE
)

# Mapa de negaciones comunes a su forma expandida
_NEGATIONS = {
    "can't": "can not", "cant": "can not",
    "won't": "will not", "wont": "will not",
    "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not",
    "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "werent": "were not",
    "doesn't": "does not", "doesnt": "does not",
    "don't": "do not", "dont": "do not",
    "didn't": "did not", "didnt": "did not",
    "haven't": "have not", "havent": "have not",
    "hasn't": "has not", "hasnt": "has not",
    "hadn't": "had not", "hadnt": "had not",
    "shouldn't": "should not", "shouldnt": "should not",
    "wouldn't": "would not", "wouldnt": "would not",
    "couldn't": "could not", "couldnt": "could not",
    "ain't": "is not", "aint": "is not",
}

def _normalize_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _expand_negations(s: str) -> str:
    toks = s.split()
    out = []
    for t in toks:
        low = t.lower()
        out.append(_NEGATIONS.get(low, t))
    return " ".join(out)

def clean_text(text: str) -> str:
    """Limpieza robusta: minúsculas, URLs, HTML, emojis, negaciones, signos, espacios."""
    if text is None:
        return ""
    s = str(text)

    # Normalizaciones básicas
    s = s.replace("\n", " ").replace("\r", " ")

    # Minúsculas pronto para facilitar regex
    s = s.lower()

    # Quitar URLs y HTML
    s = _URL_RE.sub(" ", s)
    s = _HTML_RE.sub(" ", s)

    # Eliminar emojis
    s = _EMOJI_RE.sub(" ", s)

    # Normalizar acentos
    s = _normalize_accents(s)

    # Expandir negaciones
    s = _expand_negations(s)

    # Quitar signos de puntuación y símbolos (incluye ! ?)
    s = _NON_ALNUM_RE.sub(" ", s)

    # Colapsar espacios y recortar
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


# --------- Tokenización ligera (sin dependencias) ---------
def tokenize_text(text: str):
    """Devuelve lista de tokens [a-z0-9]+ en minúsculas."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


# --------- "Lematización" heurística (rápida) --------------
def _strip_suffixes(tok: str) -> str:
    """
    Heurística simple para inglés: elimina sufijos comunes.
    No es una lematización lingüística completa, pero funciona bien con modelos BOW/TF-IDF.
    """
    # Orden importa (más largos primero)
    for suf in ["'s", "ing", "ies", "ers", "ed", "es", "s"]:
        if tok.endswith(suf):
            if suf == "ies" and len(tok) > 3:
                return tok[:-3] + "y"
            base = tok[: -len(suf)]
            # Evitar devolver vacío
            return base if base else tok
    return tok

def lemmatize_text(text: str) -> str:
    """Devuelve string con 'lemmas' heurísticos (rápidos)."""
    if not text:
        return ""
    toks = tokenize_text(text)
    lemmas = [_strip_suffixes(t) for t in toks if t]
    lemmas = [l for l in lemmas if l]
    return " ".join(lemmas)
