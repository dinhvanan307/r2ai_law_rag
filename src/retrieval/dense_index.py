"""Dense (semantic) index over legal chunks.

Two backends, selected automatically:

* **bge-m3 / sentence-transformers** (production): set ``backend="st"`` with a
  model such as ``AITeamVN/Vietnamese_Embedding`` or ``BAAI/bge-m3``. Produces
  compact dense vectors (e.g. 1024-dim) — scales to hundreds of thousands of
  chunks (354k × 1024 float32 ≈ 1.4 GB).
* **tfidf** (lite fallback): scikit-learn TF-IDF + cosine, kept as a **sparse**
  matrix end-to-end. No model download, runs anywhere.

⚠️ Scaling note: the TF-IDF matrix is NEVER densified. Densifying a 354k×vocab
matrix would need terabytes of RAM; we keep CSR sparse and compute cosine via
sparse dot products, so the full national corpus builds on a laptop.

Both backends expose the same ``build`` / ``search`` API.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..schema import Chunk
from .text_utils import tokenize

# Bound TF-IDF vocabulary so fit() memory stays sane on a huge corpus.
_TFIDF_MAX_FEATURES = 300_000


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class DenseIndex:
    def __init__(self, backend: str = "auto", model_name: str = "BAAI/bge-m3",
                 device: str | None = None, batch_size: int = 16,
                 max_seq_length: int = 512, fp16: bool = False) -> None:
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        # Cap sequence length so a few pathologically long chunks can't blow up
        # the attention buffer (bge-m3 allows 8192 → 128GB attn on mps).
        self.max_seq_length = max_seq_length
        self.fp16 = fp16   # half precision → faster encode on mps/cuda
        self.chunks: list[Chunk] = []
        self._emb = None            # np.ndarray (st) OR scipy CSR (tfidf)
        self._sparse = False        # True when _emb is a sparse matrix
        self._model = None          # sentence-transformers model
        self._vectorizer = None     # sklearn TfidfVectorizer

    # --- backend resolution ------------------------------------------------
    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        try:
            import sentence_transformers  # noqa: F401
            return "st"
        except Exception:
            return "tfidf"

    def _load_st_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Hard cap input length → bounds attention memory + speeds encoding.
            self._model.max_seq_length = self.max_seq_length
            if self.fp16 and self.device in ("cuda", "mps"):
                try:
                    self._model = self._model.half()
                except Exception as e:  # pragma: no cover
                    print(f"[dense] fp16 not applied ({e}); using fp32.")
        return self._model

    # --- build -------------------------------------------------------------
    def build(self, chunks: list[Chunk]) -> "DenseIndex":
        self.chunks = chunks
        self.backend = self._resolve_backend()
        texts = [c.text for c in chunks]

        if self.backend == "st":
            model = self._load_st_model()
            emb = model.encode(texts, batch_size=self.batch_size,
                               show_progress_bar=True, normalize_embeddings=True)
            self._emb = np.asarray(emb, dtype=np.float32)
            self._sparse = False
        else:  # tfidf — keep sparse, never densify
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            # Adaptive min_df: prune rare terms on large corpora to bound memory.
            min_df = 2 if len(texts) > 5000 else 1
            self._vectorizer = TfidfVectorizer(
                tokenizer=tokenize, lowercase=False, token_pattern=None,
                ngram_range=(1, 2), min_df=min_df, max_features=_TFIDF_MAX_FEATURES,
                dtype=np.float32,
            )
            mat = self._vectorizer.fit_transform(texts)   # CSR sparse
            self._emb = normalize(mat, norm="l2", axis=1)  # still sparse CSR
            self._sparse = True
        return self

    # --- encode query ------------------------------------------------------
    def _encode_query(self, query: str):
        if self.backend == "st":
            model = self._load_st_model()
            q = model.encode([query], normalize_embeddings=True)
            return np.asarray(q, dtype=np.float32)          # (1, d) dense
        from sklearn.preprocessing import normalize
        mat = self._vectorizer.transform([query])           # (1, d) sparse
        return normalize(mat, norm="l2", axis=1)

    def search(self, query: str, top_n: int = 100) -> list[tuple[int, float]]:
        if self._emb is None:
            return []
        q = self._encode_query(query)
        if self._sparse:
            # cosine = normalized sparse dot product → dense (N,) score vector
            scores = (self._emb @ q.T).toarray().ravel()
        else:
            scores = self._emb @ q[0]
        if top_n < len(scores):
            idx = np.argpartition(-scores, top_n)[:top_n]
            order = idx[np.argsort(-scores[idx])]
        else:
            order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]

    # --- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "model_name": self.model_name,
            "chunks": self.chunks,
            "emb": self._emb,            # sparse CSR or dense ndarray (both picklable)
            "sparse": self._sparse,
        }
        if self.backend == "tfidf":
            payload["vectorizer"] = self._vectorizer
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None,
             batch_size: int = 16, max_seq_length: int = 512,
             fp16: bool = False) -> "DenseIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        # device/fp16 KHONG duoc luu trong .pkl -> phai truyen lai khi load,
        # neu khong query-encode se mac dinh CPU + fp32 (cham han nhieu).
        idx = cls(backend=data["backend"], model_name=data.get("model_name", ""),
                  device=device, batch_size=batch_size,
                  max_seq_length=max_seq_length, fp16=fp16)
        idx.chunks = data["chunks"]
        idx._emb = data["emb"]
        idx._sparse = data.get("sparse", False)
        if data["backend"] == "tfidf":
            idx._vectorizer = data["vectorizer"]
        return idx
