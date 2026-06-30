"""Cross-encoder reranker (pluggable) with a lexical-overlap lite fallback.

Production: a Vietnamese cross-encoder such as ``AITeamVN/Vietnamese_Reranker``
or ``BAAI/bge-reranker-v2-m3`` scores each (query, article) pair for fine-grained
relevance — the single biggest precision lever after fusion.

Lite fallback: token-overlap (Jaccard-ish) scoring via rapidfuzz, so the
pipeline reranks sensibly even with no model available.
"""
from __future__ import annotations

from .text_utils import tokenize


class Reranker:
    def __init__(self, backend: str = "auto",
                 model_name: str = "AITeamVN/Vietnamese_Reranker",
                 device: str | None = None, batch_size: int = 16) -> None:
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        try:
            import sentence_transformers  # noqa: F401
            return "cross-encoder"
        except Exception:
            return "lexical"

    def _safe_device(self) -> str | None:
        """Khong cho 'cuda' lot xuong khi may KHONG co GPU.

        Truoc day: device='cuda' nhung Accelerator chua bat -> CrossEncoder
        hoac crash, hoac am tham roi ve CPU -> rerank 2000 cau cham 3-10h ma
        KHONG bao gi. Gio: phat hien som, ha xuong 'cpu' va IN canh bao.
        """
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("⚠️  [reranker] device=cuda nhung torch.cuda.is_available()"
                          "=False -> ha ve CPU (se CHAM). FIX: bat Accelerator GPU.",
                          flush=True)
                    return "cpu"
            except Exception as e:  # torch chua cai
                print(f"⚠️  [reranker] device=cuda nhung khong import duoc torch "
                      f"({e}) -> ha ve CPU.", flush=True)
                return "cpu"
        return self.device

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            dev = self._safe_device()
            print(f"[reranker] tai cross-encoder {self.model_name} tren device={dev!r} …",
                  flush=True)
            self._model = CrossEncoder(self.model_name, device=dev, max_length=512)
        return self._model

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Return a relevance score per candidate (same order)."""
        if not candidates:
            return []
        backend = self._resolve_backend()
        if backend == "cross-encoder":
            model = self._load()
            pairs = [(query, c) for c in candidates]
            scores = model.predict(pairs, batch_size=self.batch_size,
                                   show_progress_bar=False)
            return [float(s) for s in scores]
        return self._lexical_scores(query, candidates)

    @staticmethod
    def _lexical_scores(query: str, candidates: list[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(candidates)
        out: list[float] = []
        for c in candidates:
            ct = set(tokenize(c))
            if not ct:
                out.append(0.0)
                continue
            inter = len(q & ct)
            # Recall-leaning overlap: |q∩c| / |q|  (rewards covering the query).
            out.append(inter / len(q))
        return out
