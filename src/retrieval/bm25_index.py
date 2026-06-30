"""BM25 lexical index over legal chunks (Vietnamese-tokenised).

BM25 is essential for legal retrieval because exact lexical cues — document
codes ("04/2017/QH14"), article numbers, and rare legal terms — must match
precisely, which dense vectors alone often miss.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..schema import Chunk
from .text_utils import tokenize


class BM25Index:
    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []

    def build(self, chunks: list[Chunk]) -> "BM25Index":
        self.chunks = chunks
        self._corpus_tokens = [tokenize(c.text) for c in chunks]
        # rank_bm25 requires non-empty docs; guard empty tokenisations.
        safe = [toks or ["∅"] for toks in self._corpus_tokens]
        self.bm25 = BM25Okapi(safe)
        return self

    def search(self, query: str, top_n: int = 100) -> list[tuple[int, float]]:
        """Return ``(chunk_index, score)`` sorted by descending score."""
        if self.bm25 is None:
            return []
        q = tokenize(query) or ["∅"]
        scores = self.bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    # --- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks,
                         "corpus_tokens": self._corpus_tokens}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx.chunks = data["chunks"]
        idx._corpus_tokens = data["corpus_tokens"]
        safe = [toks or ["∅"] for toks in idx._corpus_tokens]
        idx.bm25 = BM25Okapi(safe)
        return idx
