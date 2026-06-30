"""End-to-end pipeline wiring: index build + per-question inference."""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .corpus.legal_chunker import (build_chunks, build_indexable_chunks,
                                   parse_document)
from .corpus.loader import load_corpus
from .generation.answerer import Answerer
from .generation.llm import LLMClient
from .postprocess import build_qa_item
from .retrieval.bm25_index import BM25Index
from .retrieval.dense_index import DenseIndex
from .retrieval.reranker import Reranker
from .retrieval.retriever import Retriever
from .schema import Article, Chunk, QAItem


def build_articles(cfg: Config, limit: int | None = None) -> list[Article]:
    """Parse the whole corpus into articles.

    ``limit`` (optional) caps the number of documents — useful for quick smoke
    tests on a laptop. Leave as ``None`` for the full corpus (competition runs).
    """
    docs = load_corpus(cfg.resolve(cfg.paths.corpus_dir))
    if limit:
        docs = docs[:limit]

    articles: list[Article] = []
    for d in docs:
        articles.extend(parse_document(d))
    return articles


def build_and_save_index(cfg: Config, limit: int | None = None) -> tuple[int, int]:
    """Parse corpus → chunk → build BM25 + dense indexes → persist.

    ``limit`` caps documents for a quick smoke test; ``None`` = full corpus.
    Returns ``(num_articles, num_chunks)``.
    """
    print("[build] (1/5) loading + parsing corpus …", flush=True)
    docs = load_corpus(cfg.resolve(cfg.paths.corpus_dir))
    if limit:
        docs = docs[:limit]
    articles, chunks, report = build_indexable_chunks(
        docs, split_long=cfg.chunk.split_long, max_chars=cfg.chunk.max_chars)
    print(report.summary(), flush=True)

    print("[build] (2/5) building BM25 index (pyvi tokenize — có thể lâu) …", flush=True)
    bm25 = BM25Index().build(chunks)

    print(f"[build] (3/5) building dense index (backend={cfg.models.dense_backend}) "
          f"— bước này lâu nhất; có progress bar nếu dùng 'st' …", flush=True)
    dense = DenseIndex(
        backend=cfg.models.dense_backend,
        model_name=cfg.models.dense_model,
        device=cfg.models.device,
        batch_size=cfg.models.dense_batch_size,
        max_seq_length=cfg.models.dense_max_seq_length,
        fp16=cfg.models.dense_fp16,
    ).build(chunks)

    print("[build] (4/5) saving indexes …", flush=True)
    index_dir = cfg.resolve(cfg.paths.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(index_dir / "bm25.pkl")
    dense.save(index_dir / "dense.pkl")
    print("[build] (5/5) done.", flush=True)
    return len(articles), len(chunks)


# --- loading + inference ---------------------------------------------------
def load_retriever(cfg: Config) -> Retriever:
    index_dir = cfg.resolve(cfg.paths.index_dir)
    print(f"[load] doc bm25.pkl + dense.pkl tu {index_dir} (chi LOAD, KHONG build) …",
          flush=True)
    bm25 = BM25Index.load(index_dir / "bm25.pkl")
    # QUAN TRONG: truyen device/fp16 -> query-encode chay tren GPU.
    # Neu khong, dense.pkl load mac dinh device=None (CPU) + fp32 -> cham nhieu.
    dense = DenseIndex.load(
        index_dir / "dense.pkl",
        device=cfg.models.device,
        batch_size=cfg.models.dense_batch_size,
        max_seq_length=cfg.models.dense_max_seq_length,
        fp16=cfg.models.dense_fp16,
    )
    print(f"[load] dense backend={dense.backend} device={cfg.models.device!r} "
          f"fp16={cfg.models.dense_fp16} | reranker={cfg.models.reranker_model}",
          flush=True)
    reranker = Reranker(
        backend=cfg.models.reranker_backend,
        model_name=cfg.models.reranker_model,
        device=cfg.models.device,
    )
    r = cfg.retrieval

    hyde_generator = None
    if getattr(r, "use_hyde", False):
        from .generation.llm import LLMClient
        print("⚠️  [HyDE] use_hyde=TRUE -> moi query se goi LLM sinh van ban gia dinh "
              "TRUOC khi retrieve. Cham hon NHIEU va can Ollama/LLM san sang. "
              "Neu chi chay extractive IR -> nen TAT (use_hyde: false).", flush=True)
        llm = LLMClient(
            backend=cfg.models.llm_backend,
            model=cfg.models.llm_model,
            base_url=cfg.models.llm_base_url,
            temperature=cfg.models.llm_temperature,
            max_tokens=cfg.models.llm_max_tokens,
        )
        hyde_generator = llm.generate_hyde

    return Retriever(
        bm25, dense, reranker,
        bm25_top_n=r.bm25_top_n, dense_top_n=r.dense_top_n,
        rrf_weights=tuple(r.rrf_weights), rrf_k=r.rrf_k,
        rerank_top_n=r.rerank_top_n, use_reranker=r.use_reranker,
        min_k=r.min_k, max_k=r.max_k, rel_threshold=r.rel_threshold,
        rel_score_transform=getattr(r, "rel_score_transform", "none"),
        hyde_generator=hyde_generator
    )


def make_answerer(cfg: Config) -> Answerer:
    llm = LLMClient(
        backend=cfg.models.llm_backend,
        model=cfg.models.llm_model,
        base_url=cfg.models.llm_base_url,
        temperature=cfg.models.llm_temperature,
        max_tokens=cfg.models.llm_max_tokens,
    )
    return Answerer(llm)


class Pipeline:
    """Holds loaded retriever + answerer for batch inference."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.retriever = load_retriever(cfg)
        self.answerer = make_answerer(cfg)

    def answer_one(self, qid: int, question: str) -> QAItem:
        articles = self.retriever.retrieve(question)
        answer = self.answerer.answer(question, articles)
        return build_qa_item(qid, question, answer, articles)
