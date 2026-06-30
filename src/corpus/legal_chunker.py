"""Legal-aware chunker for Vietnamese legal documents.

Vietnamese legal texts are hierarchical:
    Phần → Chương → Mục → Điều → Khoản → Điểm

The competition matches at the **Điều (article)** level, so the atomic chunk is
one article. This module turns the raw full text of a document into a list of
``Article`` objects, preserving Chương/Mục context for richer retrieval.

Design notes
------------
* We split primarily on ``Điều N`` headers. Everything between two article
  headers (including its khoản/điểm) is the article body.
* Chương/Mục headers seen *before* an article are attached as context.
* Article numbers are normalised to ``"Điều N"`` (the scorer normalises
  ``Điều X`` the same way), tolerant of ``Điều 4.``, ``Điều 4:`` etc.
* Optional ``split_long_articles`` produces khoản-level sub-chunks for dense
  retrieval while every sub-chunk still points back to its parent ``Điều``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..schema import Article, Chunk, LegalDoc

# --- Header patterns -------------------------------------------------------
# Leading markdown/formatting tolerated: spaces, tabs, '#', '>', '*', '-'.
# This recovers documents whose article headers are bold/heading in Markdown
# (e.g. "**Điều 1.**", "### Điều 1", "> Điều 1") — common in vbpl Markdown dumps.
_MD_PREFIX = r"[ \t#>*\-]*"
# "Điều 1", "Điều 12.", "Điều 3:", "Điều 4 -", "**Điều 5. ...**", "**Điều****1.**"
# Separator after "Điều" may be spaces and/or markdown emphasis (* #) glued in.
_ARTICLE_RE = re.compile(rf"^{_MD_PREFIX}Điều[\s*#]+(\d+)\s*[.:\-–]?\s*(.*)$", re.MULTILINE)
_CHUONG_RE = re.compile(rf"^{_MD_PREFIX}(CHƯƠNG|Chương)\s+([IVXLCDM\d]+)\b.*$", re.MULTILINE)
_MUC_RE = re.compile(rf"^{_MD_PREFIX}(MỤC|Mục)\s+(\d+)\b.*$", re.MULTILINE)
# Khoản: "1.", "12." at line start (markdown-tolerant). Điểm: "a)", "b)".
_KHOAN_RE = re.compile(rf"^{_MD_PREFIX}(\d+)\.\s+", re.MULTILINE)
# Trailing markdown emphasis to strip from captured headings ("Title**").
_MD_TRAILING = re.compile(r"[\s*#]+$")


def normalize_article_no(raw_no: str | int) -> str:
    """Return canonical ``"Điều N"`` from a number or a messy string."""
    if isinstance(raw_no, int):
        return f"Điều {raw_no}"
    m = re.search(r"(\d+)", str(raw_no))
    return f"Điều {m.group(1)}" if m else f"Điều {str(raw_no).strip()}"


# --- Stored-article header parsing (legal_articles.jsonl path) --------------
# Rows in legal_articles.jsonl store the body as "**Điều N. Title**\n\nbody".
# Building the index *directly from the jsonl* (Kaggle/Colab path) previously
# kept ``article_title`` empty and embedded the raw "**" markdown. This helper
# recovers the heading as a clean ``article_title`` field — a strong retrieval
# signal that ~27% of rows otherwise drop — and strips "**" emphasis from the
# body (noise for the dense encoder; BM25 already discards it). Header-less
# prose rows (~16%) fall back to an empty title and the unchanged body, so the
# transform is strictly additive: worst case equals the previous behaviour.
_STORED_HDR_RE = re.compile(
    r"^[ \t#>*\-]*Điều[\s*#]+\d+\s*[.:\-–]?[ \t]*(.*?)[ \t]*\**\s*$",
    re.MULTILINE,
)


def split_stored_article(raw: str) -> tuple[str, str]:
    """Return ``(article_title, clean_body)`` from a stored jsonl article."""
    t = (raw or "").replace("\r\n", "\n").lstrip("\n")
    m = _STORED_HDR_RE.match(t)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip().strip("*").strip()
        body = t[m.end():]
    else:
        title, body = "", t
    body = re.sub(r"\n{3,}", "\n\n", body.replace("**", "")).strip()
    return title, body


def _clean(text: str) -> str:
    # Normalise newlines / non-breaking spaces / excessive blank lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_context_headers(text: str) -> list[tuple[int, str, str]]:
    """Return sorted (position, kind, label) for Chương/Mục headers."""
    headers: list[tuple[int, str, str]] = []
    for m in _CHUONG_RE.finditer(text):
        headers.append((m.start(), "chuong", re.sub(r"\s+", " ", m.group(0).strip())))
    for m in _MUC_RE.finditer(text):
        headers.append((m.start(), "muc", re.sub(r"\s+", " ", m.group(0).strip())))
    headers.sort(key=lambda x: x[0])
    return headers


def _context_at(pos: int, headers: list[tuple[int, str, str]]) -> tuple[str, str]:
    """Latest Chương/Mục label that appears before ``pos``."""
    chuong = muc = ""
    for hpos, kind, label in headers:
        if hpos > pos:
            break
        if kind == "chuong":
            chuong, muc = label, ""   # new chương resets mục
        else:
            muc = label
    return chuong, muc


def parse_document(doc: LegalDoc) -> list[Article]:
    """Parse a ``LegalDoc`` full text into a list of ``Article`` objects."""
    text = _clean(doc.raw_text)
    if not text:
        return []

    headers = _find_context_headers(text)
    matches = list(_ARTICLE_RE.finditer(text))
    articles: list[Article] = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        num = m.group(1)
        # Heading is the remainder of the "Điều N ..." line (strip md emphasis).
        heading_line = _MD_TRAILING.sub("", m.group(2).strip()).strip("*").strip()
        body = text[m.end():end].strip()
        chuong, muc = _context_at(start, headers)

        articles.append(
            Article(
                law_id=doc.law_id,
                doc_type=doc.doc_type,
                doc_title=doc.title,
                article_no=normalize_article_no(num),
                article_title=heading_line,
                text=body,
                chuong=chuong,
                muc=muc,
                source_url=doc.source_url,
            )
        )
    return articles


def split_khoan(article: Article, max_chars: int = 1200) -> list[str]:
    """Split a long article into khoản-level pieces for dense retrieval.

    Short articles return a single piece. Sub-chunks always map back to the
    parent article via ``Chunk.article_uid`` in :func:`article_to_chunks`.
    """
    body = article.text
    if len(body) <= max_chars:
        return [body]

    # Split at "1." "2." khoản boundaries while keeping the marker.
    parts: list[str] = []
    idxs = [m.start() for m in _KHOAN_RE.finditer(body)]
    if not idxs:
        # Fallback: hard wrap by paragraphs.
        para, buf = body.split("\n"), ""
        for p in para:
            if len(buf) + len(p) > max_chars and buf:
                parts.append(buf.strip())
                buf = ""
            buf += p + "\n"
        if buf.strip():
            parts.append(buf.strip())
        return parts or [body]

    idxs.append(len(body))
    head = body[: idxs[0]].strip()
    for j in range(len(idxs) - 1):
        seg = body[idxs[j]: idxs[j + 1]].strip()
        parts.append(seg)
    if head:
        parts.insert(0, head)
    return [p for p in parts if p.strip()]


def article_to_chunks(article: Article, split_long: bool = True,
                      max_chars: int = 1200) -> list[Chunk]:
    """Convert an article into one or more retrievable ``Chunk`` objects."""
    pieces = split_khoan(article, max_chars) if split_long else [article.text]
    chunks: list[Chunk] = []
    for k, piece in enumerate(pieces):
        # Metadata Enrichment
        meta_lines = [f"Văn bản: {article.doc_name}"]
        if article.chuong:
            meta_lines.append(f"Chương: {article.chuong}")
        if article.muc:
            meta_lines.append(f"Mục: {article.muc}")
        
        title_line = f"Điều khoản: {article.article_no}"
        if article.article_title:
            title_line += f" {article.article_title}"
        meta_lines.append(title_line)
        
        context_str = "\n".join(meta_lines)
        enriched_piece = f"{context_str}\nNội dung:\n{piece.strip()}"
        
        chunks.append(
            Chunk(
                chunk_id=f"{article.article_uid}#${k}",
                article_uid=article.article_uid,
                doc_uid=article.doc_uid,
                law_id=article.law_id,
                article_no=article.article_no,
                text=enriched_piece,
                meta={
                    "doc_name": article.doc_name,
                    "article_title": article.article_title,
                    "chuong": article.chuong,
                    "muc": article.muc,
                },
            )
        )
    return chunks


def build_chunks(articles: Iterable[Article], split_long: bool = True,
                 max_chars: int = 1200) -> list[Chunk]:
    """Flatten many articles into chunks for indexing."""
    out: list[Chunk] = []
    for a in articles:
        out.extend(article_to_chunks(a, split_long=split_long, max_chars=max_chars))
    return out


# --- No-Điều handling (F2-aware index filtering) ---------------------------
# The competition grader matches at the **Điều (article)** level: every gold
# answer has the form ``law_id|tên văn bản|Điều X`` (verified: 100% of gold and
# dev_gold entries contain "Điều X"). A document with no parsable "Điều" header
# therefore can *never* be a gold hit — indexing it can only add false-positive
# candidates that depress Precision (and thus macro-F2). So such docs are kept
# OUT of the index.
#
# But "produced 0 articles" has two very different causes:
#   * EXPECTED — admin docs (Thông tư cũ, Chỉ thị, Lệnh…) that are legitimately
#     structured with Mục La Mã / Khoản instead of Điều. Dropping them is correct.
#   * SUSPECT — high-tier instruments (Luật/Bộ luật/Pháp lệnh/Nghị định, or any
#     QH-issued law_id) that SHOULD have Điều. Zero chunks here means the stored
#     text is corrupt or the wrong document was saved (e.g. 45/2019/QH14 Bộ luật
#     Lao động whose text got overwritten). These must be surfaced, not lost.
_NO_DIEU_OK_TYPES = {
    "Thông tư", "Thông tư liên tịch", "Thông tư liên bộ", "Thông tư liên ngành",
    "Chỉ thị", "Công văn", "Lệnh", "Nghị quyết", "Nghị quyết liên tịch",
    "Quyết định", "Chương trình", "Văn bản khác", "Thông báo", "Công điện",
}
_DIEU_EXPECTED_TYPES = {"Luật", "Bộ luật", "Pháp lệnh", "Nghị định", "Hiến pháp"}
_QH_LAWID_RE = re.compile(r"/QH\d+\b", re.IGNORECASE)


def _is_dieu_expected(doc: LegalDoc) -> bool:
    """True if this doc *should* contain Điều articles (so 0 chunks = alarm).

    High-tier instruments (Luật/Bộ luật/Pháp lệnh/Nghị định/Hiến pháp) are
    always Điều-structured. A QH-issued law_id is also Điều-structured **unless**
    its doc_type is an inherently article-less form (Nghị quyết, Lệnh…): QH
    Nghị quyết are resolutions written in Mục/Khoản, not Điều, so they must NOT
    raise a false corruption alarm.
    """
    dtype = (doc.doc_type or "").strip()
    if dtype in _DIEU_EXPECTED_TYPES:
        return True
    if _QH_LAWID_RE.search(doc.law_id or "") and dtype not in _NO_DIEU_OK_TYPES:
        return True
    return False


@dataclass
class ChunkBuildReport:
    """Summary of which docs made it into the index and which were dropped."""
    n_docs: int = 0
    n_docs_indexed: int = 0
    n_articles: int = 0
    n_chunks: int = 0
    dropped_expected: dict[str, int] = field(default_factory=dict)
    dropped_suspect: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def n_dropped(self) -> int:
        return sum(self.dropped_expected.values()) + len(self.dropped_suspect)

    def summary(self) -> str:
        lines = [
            f"[chunk] docs={self.n_docs:,} → indexed={self.n_docs_indexed:,} "
            f"(dropped {self.n_dropped:,} no-Điều) | "
            f"articles={self.n_articles:,} chunks={self.n_chunks:,}",
        ]
        if self.dropped_expected:
            top = sorted(self.dropped_expected.items(),
                         key=lambda x: -x[1])
            shown = ", ".join(f"{k}×{v}" for k, v in top[:8])
            lines.append(f"[chunk]   dropped (expected, no Điều by design): {shown}")
        if self.dropped_suspect:
            lines.append(
                f"[chunk]   ⚠ {len(self.dropped_suspect)} HIGH-TIER docs produced 0 "
                f"chunks but SHOULD have Điều — likely corrupt/wrong text, re-fetch:")
            for law_id, dtype, title in self.dropped_suspect[:25]:
                lines.append(f"[chunk]      {law_id:24s} [{dtype}] {title[:60]}")
            if len(self.dropped_suspect) > 25:
                lines.append(f"[chunk]      … +{len(self.dropped_suspect) - 25} more")
        return "\n".join(lines)


def build_indexable_chunks(
    docs: Iterable[LegalDoc], split_long: bool = True, max_chars: int = 1200,
) -> tuple[list[Article], list[Chunk], ChunkBuildReport]:
    """Parse docs → articles → chunks, excluding no-Điều docs from the index.

    Returns ``(articles, chunks, report)``. Docs that yield no parsable Điều are
    omitted from the index (they can never be a gold hit). The report classifies
    every dropped doc as *expected* (admin doc with no Điều by design) or
    *suspect* (a Luật/Nghị định/QH instrument that should have Điều — a
    data-quality red flag worth re-fetching).
    """
    report = ChunkBuildReport()
    all_articles: list[Article] = []
    for d in docs:
        report.n_docs += 1
        arts = parse_document(d)
        if arts:
            report.n_docs_indexed += 1
            all_articles.extend(arts)
        elif _is_dieu_expected(d):
            report.dropped_suspect.append(
                (d.law_id or "?", (d.doc_type or "?").strip(), d.title or ""))
        else:
            key = (d.doc_type or "Không rõ").strip() or "Không rõ"
            report.dropped_expected[key] = report.dropped_expected.get(key, 0) + 1
    report.n_articles = len(all_articles)
    chunks = build_chunks(all_articles, split_long=split_long, max_chars=max_chars)
    report.n_chunks = len(chunks)
    return all_articles, chunks, report
