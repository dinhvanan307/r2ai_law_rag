"""Convert legal-document HTML into clean plain text.

Goal: preserve the line structure the legal chunker relies on — ``Điều N``,
khoản (``1.``), điểm (``a)``), ``Chương``, ``Mục`` — while stripping nav,
scripts, styles, and boilerplate. Works on both crawled pages and
user-saved HTML (any source).
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

# Tags that never contain article text.
_DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer",
              "form", "button", "input", "iframe", "svg", "aside"]

# Common vbpl.vn / legal-portal content containers (best-effort, ordered).
_CONTENT_SELECTORS = [
    "div.toanvancontent", "div.fulltext", "div.content1", "div#divContentDoc",
    "div.vbProperties", "div.cldivContentDocVn", "article", "main",
    "div#content", "div.content",
]


def _largest_text_block(soup: BeautifulSoup) -> str:
    """Fallback: pick the DOM node with the most 'Điều' occurrences / text."""
    best, best_score = None, -1
    for div in soup.find_all(["div", "article", "section", "td"]):
        txt = div.get_text("\n", strip=True)
        if not txt:
            continue
        score = len(re.findall(r"Điều\s+\d+", txt)) * 200 + len(txt)
        if score > best_score:
            best, best_score = txt, score
    return best or soup.get_text("\n", strip=True)


def html_to_text(html: str, content_selector: str | None = None) -> str:
    """Extract the main legal text from an HTML page."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_DROP_TAGS):
        tag.decompose()

    text = ""
    selectors = [content_selector] if content_selector else _CONTENT_SELECTORS
    for sel in selectors:
        if not sel:
            continue
        node = soup.select_one(sel)
        if node:
            cand = node.get_text("\n", strip=True)
            if len(re.findall(r"Điều\s+\d+", cand)) >= 1:
                text = cand
                break
    if not text:
        text = _largest_text_block(soup)

    return clean_text(text)


def clean_text(text: str) -> str:
    """Normalise whitespace and ensure article/khoản markers start a line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    # Ensure "Điều N", "Chương", "Mục" begin on their own line.
    text = re.sub(r"(?<!\n)\s*(Điều\s+\d+\s*[.:])", r"\n\1", text)
    text = re.sub(r"(?<!\n)\s*(CHƯƠNG|Chương)\s+([IVXLCDM\d]+)", r"\n\1 \2", text)
    text = re.sub(r"(?<!\n)\s*(MỤC|Mục)\s+(\d+)", r"\n\1 \2", text)
    # Collapse spaces and blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
