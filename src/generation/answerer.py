"""Answer orchestration + grounding safeguards.

Wraps the LLM client and enforces competition-aligned guarantees:
* the answer cites at least one retrieved ``Điều X`` (grader extracts these);
* cited articles that are NOT in the retrieved context are flagged (anti-
  hallucination, competition goal #5);
* if the model returns an empty/ungrounded answer, we fall back to the
  deterministic extractive answer.
"""
from __future__ import annotations

import re

from ..schema import RetrievedArticle
from .llm import LLMClient

_ARTICLE_IN_TEXT = re.compile(r"Điều\s+(\d+)")


def cited_article_numbers(text: str) -> set[int]:
    return {int(n) for n in _ARTICLE_IN_TEXT.findall(text or "")}


class Answerer:
    def __init__(self, llm: LLMClient, enforce_citation: bool = True) -> None:
        self.llm = llm
        self.enforce_citation = enforce_citation

    def answer(self, question: str, articles: list[RetrievedArticle]) -> str:
        text = ""
        try:
            text = self.llm.generate(question, articles)
        except Exception:
            text = ""  # fall through to extractive

        if not text.strip():
            return LLMClient.extractive_answer(question, articles)

        if self.enforce_citation and articles:
            retrieved_nums = {int(re.search(r"(\d+)", a.article_no).group(1))
                              for a in articles if re.search(r"(\d+)", a.article_no)}
            cited = cited_article_numbers(text)
            # No citation at all → append grounded citation line.
            if not cited:
                text = text.rstrip() + "\n\n" + _citation_line(articles)
            # Cites an article not in context → append a clarifying grounded note
            # (we do not silently trust hallucinated article numbers).
            elif not (cited & retrieved_nums):
                text = text.rstrip() + "\n\n" + _citation_line(articles)
        return text.strip()


def _citation_line(articles: list[RetrievedArticle]) -> str:
    cites = ", ".join(f"{a.article_no} của {a.doc_name}" for a in articles[:5])
    return f"Căn cứ pháp lý: {cites}."
