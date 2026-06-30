"""Vietnamese-friendly tokenisation for lexical retrieval.

Uses ``pyvi`` word segmentation when available (best for Vietnamese), and
falls back to a robust regex tokenizer otherwise so the pipeline always runs.
Legal cues like article numbers and document codes are preserved as tokens.
"""
from __future__ import annotations

import re
import unicodedata

try:  # Optional, much better Vietnamese segmentation.
    from pyvi import ViTokenizer  # type: ignore
    _HAS_PYVI = True
except Exception:  # pragma: no cover - optional dep
    _HAS_PYVI = False

# Vietnamese legal stopwords (light list — keep domain words like "điều").
_STOPWORDS = {
    "và", "của", "các", "có", "được", "cho", "trong", "là", "với", "theo",
    "tại", "này", "đó", "khi", "về", "đến", "từ", "một", "những", "hoặc",
    "thì", "để", "phải", "không", "đã", "sẽ", "bị", "ra", "vào", "như",
}

# Keep document codes like 04/2017/QH14 and numbers as single tokens.
_CODE_RE = re.compile(r"\d+/\d+/[A-Za-zĐđ\-]+")
_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return text.strip()


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Return a list of lowercase tokens for BM25 / lexical scoring."""
    text = normalize(text)
    codes = _CODE_RE.findall(text)
    if _HAS_PYVI:
        # ViTokenizer joins multi-syllable words with underscores.
        seg = ViTokenizer.tokenize(text).lower()
        toks = [t for t in re.split(r"\s+", seg) if t]
        toks = [re.sub(r"[^0-9a-zà-ỹđ_]", "", t) for t in toks]
    else:
        toks = [t.lower() for t in _WORD_RE.findall(text)]

    toks = [t for t in toks if t]
    if drop_stopwords:
        toks = [t for t in toks if t.replace("_", " ") not in _STOPWORDS]
    # Re-add document codes (lowercased) so exact code match always indexes.
    toks.extend(c.lower() for c in codes)
    return toks
