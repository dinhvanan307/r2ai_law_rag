"""Reciprocal Rank Fusion (RRF) for combining BM25 + dense rankings."""
from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    weights: list[float] | None = None,
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Fuse multiple ``(doc_index, score)`` rankings into one.

    RRF score = Σ_i  w_i / (rrf_k + rank_i(doc)).  Robust to score-scale
    differences between BM25 and cosine, which is exactly our situation.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, (doc_idx, _score) in enumerate(ranking):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + w / (rrf_k + rank + 1)
    out = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return out[:top_n] if top_n else out
