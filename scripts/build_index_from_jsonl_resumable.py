"""Resumable index build directly from legal_articles.jsonl — survives disconnects/crashes.

Encodes dense embeddings in shards, saving each shard to disk as it finishes.
If the run crashes or is disconnected, re-running this script will resume from
where it left off. Once all shards are ready, it assembles them into `dense.pkl`.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from src.config import load_config
from src.schema import Article
from src.corpus.legal_chunker import build_chunks
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex


def load_articles_from_jsonl(jsonl_path: Path) -> list[Article]:
    articles = []
    print(f"[resumable_jsonl] Loading articles from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                doc_num = item.get("doc_num")
                doc_ref = item.get("doc_ref")
                art_num = item.get("article_number")
                text = item.get("text", "")

                if not doc_num or not doc_ref or not art_num:
                    continue

                parts = doc_ref.split("|")
                if len(parts) < 2:
                    continue
                doc_name = parts[1]
                
                doc_type = doc_name.split(" ")[0]
                doc_title = doc_name.replace(doc_type, "", 1).replace(doc_num, "", 1).strip()

                articles.append(Article(
                    law_id=doc_num,
                    doc_type=doc_type,
                    doc_title=doc_title,
                    article_no=art_num,
                    text=text
                ))
            except Exception:
                pass
    return articles


def encode_shards(model, texts, shard_dir: Path, shard_size: int,
                  batch_size: int) -> int:
    shard_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    n_shards = (n + shard_size - 1) // shard_size
    for s in range(n_shards):
        out = shard_dir / f"emb_{s:05d}.npy"
        if out.exists():
            continue
        lo, hi = s * shard_size, min((s + 1) * shard_size, n)
        emb = model.encode(texts[lo:hi], batch_size=batch_size,
                           show_progress_bar=True, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, emb)
        tmp.rename(out)
        print(f"[resumable_jsonl] shard {s+1}/{n_shards} saved ({hi-lo} vecs)", flush=True)
    return n_shards


def assemble(shard_dir: Path, n_shards: int, chunks, model_name: str,
              out_path: Path) -> None:
    mats = [np.load(shard_dir / f"emb_{s:05d}.npy") for s in range(n_shards)]
    emb = np.concatenate(mats, axis=0).astype(np.float32)
    assert emb.shape[0] == len(chunks), \
        f"emb rows {emb.shape[0]} != chunks {len(chunks)}"
    payload = {"backend": "st", "model_name": model_name,
               "chunks": chunks, "emb": emb, "sparse": False}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[resumable_jsonl] assembled dense.pkl: {emb.shape} → {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resumable index build from legal_articles.jsonl.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard-size", type=int, default=5000,
                    help="Chunks per checkpoint shard (smaller = safer, more I/O)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    index_dir = cfg.resolve(cfg.paths.index_dir)
    shard_dir = index_dir / "_emb_shards"
    
    articles_file = Path("data/legal_articles.jsonl")
    if not articles_file.exists():
        raise SystemExit(f"Error: {articles_file} does not exist! Please run from r2ai_pipeline directory.")

    print("[resumable_jsonl] Loading and chunking articles from jsonl...", flush=True)
    articles = load_articles_from_jsonl(articles_file)
    chunks = build_chunks(articles, split_long=cfg.chunk.split_long, max_chars=cfg.chunk.max_chars)
    print(f"[resumable_jsonl] Generated {len(chunks):,} chunks", flush=True)

    # --- BM25 (always build brand new) ---
    bm25_path = index_dir / "bm25.pkl"
    print("[resumable_jsonl] building BM25 …", flush=True)
    BM25Index().build(chunks).save(bm25_path)
    print("[resumable_jsonl] BM25 saved", flush=True)

    # --- Dense (sharded, resumable) ---
    if cfg.models.dense_backend == "tfidf":
        raise SystemExit("Resumable build is for backend=st. Set dense_backend: st.")

    di = DenseIndex(backend="st", model_name=cfg.models.dense_model,
                    device=cfg.models.device, batch_size=cfg.models.dense_batch_size,
                    max_seq_length=cfg.models.dense_max_seq_length,
                    fp16=cfg.models.dense_fp16)
    model = di._load_st_model()
    texts = [c.text for c in chunks]

    print(f"[resumable_jsonl] encoding in shards of {args.shard_size} → {shard_dir}",
          flush=True)
    n_shards = encode_shards(model, texts, shard_dir, args.shard_size,
                             cfg.models.dense_batch_size)

    print("[resumable_jsonl] all shards present → assembling dense.pkl", flush=True)
    assemble(shard_dir, n_shards, chunks, cfg.models.dense_model,
             index_dir / "dense.pkl")
    print("[resumable_jsonl] DONE. (Có thể xóa thư mục _emb_shards để tiết kiệm dung lượng.)",
          flush=True)


if __name__ == "__main__":
    main()
