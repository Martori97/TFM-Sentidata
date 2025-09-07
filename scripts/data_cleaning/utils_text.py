# scripts/data_cleaning/utils_text.py
import re
import unicodedata
from html import unescape

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TAG_RE = re.compile(r"<.*?>")
USER_RE = re.compile(r"@[A-Za-z0-9_]+")
NUM_RE = re.compile(r"\b\d+([.,]\d+)?\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")            # colapsar repeticiones 3+ a 2
PUNCT_RE = re.compile(r"[^\w\s'-]")             # conservar ' y -
MULTISPACE_RE = re.compile(r"\s+")

NEGATIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "won't": "will not", "can't": "can not",
    "cannot": "can not", "shouldn't": "should not", "n't": " not"
}
NEGATION_SCOPE = 3  # unir hasta 3 tokens tras "not" (heurística)

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def expand_negations(text: str) -> str:
    for k, v in NEGATIONS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text

def _mark_negation(tokens):
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        out.append(t)
        if t == "not":
            for j in range(1, NEGATION_SCOPE + 1):
                if i + j < len(tokens):
                    out.append("not_" + tokens[i + j])
            i += NEGATION_SCOPE
        i += 1
    # evita cascadas not_not_*
    return [t for t in out if not t.startswith("not_not_")]

def clean_text(text: str, keep_numbers: bool = False) -> str:
    """
    Limpieza robusta para reviews (EN/ES mixto):
    - lower, quitar/normalizar HTML/URLs/usuarios, tildes
    - colapsar repeticiones de caracteres
    - expandir negaciones y marcar not_*
    - quitar números (opcional), puntuación “dura”
    - espacios normalizados
    """
    if not isinstance(text, str):
        return ""
    text = unescape(text).lower()
    text = URL_RE.sub(" ", text)
    text = USER_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = expand_negations(text)
    text = strip_accents(text)
    text = REPEAT_RE.sub(r"\1\1", text)
    if not keep_numbers:
        text = NUM_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    toks = text.split()
    if not toks:
        return ""
    toks = _mark_negation(toks)
    return " ".join(toks)
