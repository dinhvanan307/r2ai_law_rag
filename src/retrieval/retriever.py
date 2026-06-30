"""End-to-end retriever: BM25 + dense -> RRF -> rerank -> top-K articles.

Output is deduplicated to **article (Điều)** granularity because that is what
the competition scores. Selection is tuned for **F2** (recall weighted 2x):
we keep articles above a relative score threshold, bounded by [min_k, max_k].
"""
from __future__ import annotations

import math

from ..schema import Chunk, RetrievedArticle
from .bm25_index import BM25Index
from .dense_index import DenseIndex
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker
from typing import Callable


class Retriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        reranker: Reranker | None = None,
        *,
        bm25_top_n: int = 100,
        dense_top_n: int = 100,
        rrf_weights: tuple[float, float] = (1.0, 1.0),
        rrf_k: int = 60,
        rerank_top_n: int = 60,
        use_reranker: bool = True,
        min_k: int = 1,
        max_k: int = 10,
        rel_threshold: float = 0.5,
        rel_score_transform: str = "none",
        hyde_generator: Callable[[str], str] | None = None,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker or Reranker()
        self.bm25_top_n = bm25_top_n
        self.dense_top_n = dense_top_n
        self.rrf_weights = rrf_weights
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n
        self.use_reranker = use_reranker
        self.min_k = min_k
        self.max_k = max_k
        self.rel_threshold = rel_threshold
        self.rel_score_transform = rel_score_transform
        self.hyde_generator = hyde_generator
        # BM25 and dense share the same chunk list (same build order).
        self.chunks: list[Chunk] = bm25.chunks

    # --- core --------------------------------------------------------------
    def _rank(self, query: str, rerank_top_n: int | None = None
              ) -> list[tuple[str, Chunk, float]]:
        """BM25 + dense -> RRF -> (optional) rerank. Returns article-level
        ``(uid, chunk, score)`` sorted by score (no F2 selection applied)."""
        search_query = query
        if self.hyde_generator is not None:
            try:
                hyde_doc = self.hyde_generator(query)
                if hyde_doc:
                    search_query = f"{query}\n{hyde_doc}"
            except Exception as e:
                print(f"[HyDE] error generating: {e}")

        bm25_rank = self.bm25.search(search_query, self.bm25_top_n)
        dense_rank = self.dense.search(search_query, self.dense_top_n)
        fused = reciprocal_rank_fusion(
            [bm25_rank, dense_rank],
            weights=list(self.rrf_weights),
            rrf_k=self.rrf_k,
        )

        # Collapse chunk-level fusion to article-level (keep best chunk score).
        art_best: dict[str, tuple[float, Chunk]] = {}
        order: list[str] = []
        for chunk_idx, score in fused:
            ch = self.chunks[chunk_idx]
            uid = ch.article_uid
            if uid not in art_best or score > art_best[uid][0]:
                art_best[uid] = (score, ch)
            if uid not in order:
                order.append(uid)

        window = rerank_top_n or self.rerank_top_n
        candidate_uids = order[:window]
        candidates = [art_best[u][1] for u in candidate_uids]

        if self.use_reranker and candidates:
            texts = [c.text for c in candidates]
            scores = self.reranker.rerank(query, texts)
            scored = list(zip(candidate_uids, candidates, scores))
            scored.sort(key=lambda x: x[2], reverse=True)
        else:
            scored = [(u, art_best[u][1], art_best[u][0]) for u in candidate_uids]
        return scored

    @staticmethod
    def _to_article(uid: str, ch: Chunk, score: float) -> RetrievedArticle:
        return RetrievedArticle(
            article_uid=uid,
            doc_uid=ch.doc_uid,
            law_id=ch.law_id,
            article_no=ch.article_no,
            doc_name=ch.meta.get("doc_name", ""),
            article_title=ch.meta.get("article_title", ""),
            text=ch.text,
            score=float(score),
        )

    def retrieve(self, query: str) -> list[RetrievedArticle]:
        return self._select(self._rank(query))

    def candidates(self, query: str, top_n: int = 50) -> list[RetrievedArticle]:
        """Top-N article candidates WITHOUT F2 selection — for hard-negative
        mining and diagnostics."""
        scored = self._rank(query, rerank_top_n=max(top_n, self.rerank_top_n))
        return [self._to_article(u, c, s) for u, c, s in scored[:top_n]]

    @staticmethod
    def _sigmoid(x: float) -> float:
        # numerically stable two-branch sigmoid
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def _transform(self, score: float) -> float:
        """Map a raw retrieval/rerank score onto the scale used by the relative
        threshold. ``sigmoid`` stabilises the ratio when reranker scores are
        logits (possibly negative); ``none`` keeps raw scores (legacy)."""
        if self.rel_score_transform == "sigmoid":
            return self._sigmoid(score)
        return score

    # --- F2-oriented selection --------------------------------------------
    def _select(self, scored: list[tuple[str, Chunk, float]]) -> list[RetrievedArticle]:
        if not scored:
            return []
        # ``scored`` is sorted by raw score desc; transform is monotonic so the
        # ordering is preserved. Transform stabilises the relative threshold for
        # cross-encoder logits (sigmoid maps to (0,1) -> top_score always > 0).
        top_score = self._transform(scored[0][2])
        out: list[RetrievedArticle] = []
        for rank, (uid, ch, score) in enumerate(scored):
            s = self._transform(score)
            keep = rank < self.min_k
            if not keep and rank < self.max_k:
                # Relative threshold: keep while score stays near the top.
                if top_score > 0:
                    keep = (s / top_score) >= self.rel_threshold
                else:
                    keep = s >= 0
            if not keep:
                break
            out.append(self._to_article(uid, ch, score))
        return out[: self.max_k]
