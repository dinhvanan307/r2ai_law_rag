"""Fetch full text of a legal document from vbpl.vn (or any direct URL).

vbpl.vn (CSDL quốc gia về VBPL) serves documents WITHOUT login. Typical
full-text URL pattern::

    https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=<id>

Because metadata (law_id / doc_type / title) comes from the seed list, this
module only needs to download the page and extract the article text. If a seed
has no ``url`` yet, ``search_keyword_url`` builds a ready-to-click search link
so you can grab the canonical full-text URL once and paste it back.
"""
from __future__ import annotations

import re
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup

from ...schema import LegalDoc
from .base import PoliteSession
from .normalize import html_to_text

VBPL_BASE = "https://vbpl.vn"
# Correct search endpoint is /Pages/vbpq-timkiem.aspx (no scope prefix).
VBPL_SEARCH = "https://vbpl.vn/Pages/vbpq-timkiem.aspx?type=0&Keyword={kw}"
# Full-text URL pattern: /{scope}/Pages/vbpq-toanvan.aspx?ItemID=<id>
_ITEMID_RE = re.compile(r"vbpq-toanvan\.aspx\?ItemID=(\d+)", re.IGNORECASE)
_ANY_ITEMID_RE = re.compile(r"ItemID=(\d+)", re.IGNORECASE)


def search_keyword_url(keyword: str) -> str:
    """A human-clickable vbpl.vn search URL for a document keyword."""
    return VBPL_SEARCH.format(kw=quote_plus(keyword))


def toanvan_url(item_id: str | int, scope: str = "TW") -> str:
    """Build a canonical full-text URL from an ItemID."""
    return f"{VBPL_BASE}/{scope}/Pages/vbpq-toanvan.aspx?ItemID={item_id}"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def resolve_url(seed: dict, session: PoliteSession,
                use_cache: bool = True) -> str | None:
    """Best-effort: search vbpl.vn and return the best full-text URL for a seed.

    Strategy: fetch the search page for the seed keyword, collect every anchor
    that points at a document (``ItemID``), then score candidates by how well
    the anchor text + nearby text match the seed's ``law_id`` and ``title``.
    The law_id (e.g. "04/2017/QH14") is a very strong, near-unique signal.
    """
    keyword = seed.get("keyword") or f"{seed.get('title','')} {seed.get('law_id','')}"
    search_url = search_keyword_url(keyword)
    html = session.get(search_url, use_cache=use_cache)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    law_id = _norm(seed.get("law_id", ""))
    title_tokens = set(_norm(seed.get("title", "")).split())

    best_url, best_score = None, -1.0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = _ITEMID_RE.search(href) or _ANY_ITEMID_RE.search(href)
        if not m:
            continue
        # Context = anchor text + parent block text (titles often near the link).
        ctx = _norm(a.get_text(" ") + " " +
                    (a.find_parent(["li", "tr", "div", "td"]).get_text(" ")
                     if a.find_parent(["li", "tr", "div", "td"]) else ""))
        score = 0.0
        if law_id and law_id in ctx:
            score += 10.0                      # exact code match → dominant
        if title_tokens:
            overlap = len(title_tokens & set(ctx.split())) / len(title_tokens)
            score += overlap                   # 0..1
        if "toanvan" in href.lower():
            score += 0.5                        # prefer full-text links
        if score > best_score:
            item_id = m.group(1)
            url = href if href.lower().startswith("http") else urljoin(VBPL_BASE, href)
            # Normalise to a toanvan URL so we always fetch full text.
            if "toanvan" not in url.lower():
                url = toanvan_url(item_id)
            best_url, best_score = url, score

    if best_url and best_score >= 1.0:
        return best_url
    if best_url:
        print(f"[vbpl] low-confidence match for {seed.get('law_id')} "
              f"(score={best_score:.1f}): {best_url} — verify manually.")
        return best_url
    return None


def fetch_document(
    seed: dict,
    session: PoliteSession,
    content_selector: str | None = None,
    use_cache: bool = True,
) -> LegalDoc | None:
    """Download + normalise one document described by a seed dict.

    Seed dict keys: ``law_id``, ``doc_type``, ``title``, ``url`` (required to
    crawl), optional ``keyword``.
    """
    url = (seed.get("url") or "").strip()
    # Convenience: allow pasting just the numeric ItemID instead of a full URL.
    if not url and str(seed.get("item_id") or "").strip():
        url = toanvan_url(str(seed["item_id"]).strip())
    if not url:
        print(f"[vbpl] no URL for {seed.get('law_id')} — open and copy the "
              f"full-text URL:\n        {search_keyword_url(seed.get('keyword', seed.get('title','')))}")
        return None

    html = session.get(url, use_cache=use_cache)
    if not html:
        return None

    text = html_to_text(html, content_selector=content_selector)
    if len(text) < 200:
        print(f"[vbpl] WARNING: extracted text very short for {seed.get('law_id')} "
              f"({len(text)} chars). Check content_selector or the saved page.")

    return LegalDoc(
        law_id=str(seed["law_id"]).strip(),
        doc_type=str(seed["doc_type"]).strip(),
        title=str(seed["title"]).strip(),
        raw_text=text,
        source_url=url,
        meta={"keyword": seed.get("keyword", ""), "note": seed.get("note", "")},
    )
