"""Core data structures for the R2AI legal RAG pipeline.

The competition scores at the *article* (Điều) level. Every object here keeps a
stable ``article_uid`` of the form ``law_id|doc_name|Điều X`` so retrieval,
generation, and scoring all speak the same language.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class LegalDoc:
    """A whole legal document (Luật / Nghị định / Thông tư ...)."""

    law_id: str                 # Mã văn bản, e.g. "04/2017/QH14"
    doc_type: str               # Loại văn bản, e.g. "Luật", "Nghị định"
    title: str                  # Trích yếu, e.g. "Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
    raw_text: str = ""          # Full plain text (article bodies)
    source_url: str = ""        # For SOURCES.md / reproducibility
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_name(self) -> str:
        """``tên văn bản`` = Loại văn bản + Mã văn bản + Trích yếu."""
        from .corpus.doc_name import format_doc_name  # local import avoids cycle

        return format_doc_name(self.doc_type, self.law_id, self.title)


@dataclass
class Article:
    """One ``Điều`` — the atomic unit the competition matches against."""

    law_id: str
    doc_type: str
    doc_title: str              # Trích yếu (for doc_name building)
    article_no: str             # Normalised, e.g. "Điều 4"
    article_title: str = ""     # Heading after the article number
    text: str = ""              # Full article body (all khoản / điểm)
    chuong: str = ""            # Context: Chương ...
    muc: str = ""               # Context: Mục ...
    source_url: str = ""

    @property
    def doc_name(self) -> str:
        from .corpus.doc_name import format_doc_name

        return format_doc_name(self.doc_type, self.law_id, self.doc_title)

    @property
    def article_uid(self) -> str:
        """Stable id: ``law_id|doc_name|Điều X`` (matches scorer normalisation)."""
        return f"{self.law_id}|{self.doc_name}|{self.article_no}"

    @property
    def doc_uid(self) -> str:
        """``law_id|doc_name`` for the relevant_docs field."""
        return f"{self.law_id}|{self.doc_name}"

    def index_text(self) -> str:
        """Text used for BM25 / dense indexing (heading + body + ids)."""
        parts = [self.doc_name, self.article_no]
        if self.article_title:
            parts.append(self.article_title)
        if self.chuong:
            parts.append(self.chuong)
        if self.muc:
            parts.append(self.muc)
        parts.append(self.text)
        return "\n".join(p for p in parts if p)


@dataclass
class Chunk:
    """A retrievable unit. Usually 1 article; long articles may split by khoản.

    ``article_uid`` always points back to the parent Điều so IR results can be
    deduplicated to article granularity regardless of sub-chunking.
    """

    chunk_id: str
    article_uid: str
    doc_uid: str
    law_id: str
    article_no: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedArticle:
    """An article returned by retrieval, with score + provenance."""

    article_uid: str
    doc_uid: str
    law_id: str
    article_no: str
    doc_name: str
    article_title: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAItem:
    """Final per-question output object → serialised into results.json."""

    id: int
    question: str
    answer: str = ""
    relevant_docs: list[str] = field(default_factory=list)
    relevant_articles: list[str] = field(default_factory=list)

    def to_submission(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "relevant_docs": self.relevant_docs,
            "relevant_articles": self.relevant_articles,
        }
