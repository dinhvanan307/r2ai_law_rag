"""Configuration loader (YAML + sensible defaults)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RetrievalCfg:
    bm25_top_n: int = 100
    dense_top_n: int = 100
    rrf_weights: tuple[float, float] = (1.0, 1.0)
    rrf_k: int = 60
    rerank_top_n: int = 60
    use_reranker: bool = True
    min_k: int = 1
    max_k: int = 8
    rel_threshold: float = 0.5
    # Score transform applied BEFORE the relative threshold in _select.
    #   "none"    -> threshold on raw scores (default; identical to old behaviour)
    #   "sigmoid" -> map cross-encoder logits through 1/(1+e^-x) first, so the
    #                relative ratio score/top is stable when reranker scores are
    #                logits / can be negative. Pick via scripts/tune_from_cache.py.
    rel_score_transform: str = "none"


@dataclass
class ModelsCfg:
    # "auto" picks the heavy backend if installed, else the lite fallback.
    dense_backend: str = "auto"          # auto | st | tfidf
    dense_model: str = "BAAI/bge-m3"
    dense_max_seq_length: int = 512      # cap tokens/chunk (prevents 128GB attn)
    dense_batch_size: int = 16
    dense_fp16: bool = False             # half precision on mps/cuda → faster
    reranker_backend: str = "auto"       # auto | cross-encoder | lexical
    reranker_model: str = "AITeamVN/Vietnamese_Reranker"
    llm_backend: str = "extractive"      # extractive | ollama | vllm | hf
    llm_model: str = "qwen2.5:7b-instruct"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    device: str | None = None


@dataclass
class ChunkCfg:
    split_long: bool = True
    max_chars: int = 1200


@dataclass
class PathsCfg:
    corpus_dir: str = "data/corpus"
    test_file: str = "data/test/test.json"
    index_dir: str = "data/index"
    results_file: str = "results.json"
    submission_zip: str = "submission.zip"


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    chunk: ChunkCfg = field(default_factory=ChunkCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (ROOT / path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc, data: dict) -> None:
    for k, v in (data or {}).items():
        if hasattr(dc, k):
            cur = getattr(dc, k)
            if k == "rrf_weights" and isinstance(v, list):
                v = tuple(v)
            setattr(dc, k, v)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        path = ROOT / "config.yaml"
    path = Path(path)
    if not path.exists():
        return cfg
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _merge(cfg.paths, data.get("paths", {}))
    _merge(cfg.chunk, data.get("chunk", {}))
    _merge(cfg.retrieval, data.get("retrieval", {}))
    _merge(cfg.models, data.get("models", {}))
    return cfg
