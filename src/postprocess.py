"""Turn retrieved articles + answer into a competition ``QAItem``.

Builds ``relevant_docs`` / ``relevant_articles`` in the exact required format and
order (highest-ranked first), deduplicated, capped for F2 balance.
"""
from __future__ import annotations

from .schema import QAItem, RetrievedArticle


def build_qa_item(
    qid: int,
    question: str,
    answer: str,
    articles: list[RetrievedArticle],
    max_articles: int | None = None,
) -> QAItem:
    if max_articles is not None:
        articles = articles[:max_articles]

    relevant_articles: list[str] = []
    relevant_docs: list[str] = []
    seen_art: set[str] = set()
    seen_doc: set[str] = set()

    for a in articles:
        if a.article_uid not in seen_art:
            relevant_articles.append(a.article_uid)
            seen_art.add(a.article_uid)
        if a.doc_uid not in seen_doc:
            relevant_docs.append(a.doc_uid)
            seen_doc.add(a.doc_uid)

    return QAItem(
        id=qid,
        question=question,
        answer=answer,
        relevant_docs=relevant_docs,
        relevant_articles=relevant_articles,
    )
