"""Multi-GPU index build for Kaggle (2×T4) directly from legal_articles.jsonl.

Optimized to run on Kaggle's dual GPU environment using multi-process pooling,
avoiding raw text chunking bugs by reading clean pre-parsed articles.
"""
from __future__ import annotations

import argparse
import pickle
import time
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from src.config import load_config
from src.schema import Article
from src.corpus.legal_chunker import build_chunks, split_stored_article
from src.retrieval.bm25_index import BM25Index


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def load_articles_from_jsonl(jsonl_path: Path) -> list[Article]:
    articles = []
    print(f"[kaggle_jsonl] Loading articles from {jsonl_path}...")
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

                # Recover heading -> clean article_title, strip "**" from body.
                # article_uid is law_id|doc_name|article_no => matching ID is
                # untouched; only the embedded/indexed TEXT improves.
                art_title, body = split_stored_article(text)

                articles.append(Article(
                    law_id=doc_num,
                    doc_type=doc_type,
                    doc_title=doc_title,
                    article_no=art_num,
                    article_title=art_title,
                    text=body
                ))
            except Exception:
                pass
    return articles


def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle multi-GPU index build from legal_articles.jsonl.")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Per-process encode batch (T4 16GB fp16 handles 128 for "
                         "1024-dim @ seq512; lower to 64 if OOM).")
    ap.add_argument("--devices", default="",
                    help="Comma list, e.g. 'cuda:0,cuda:1'. Empty = all visible GPUs.")
    ap.add_argument("--no-fp16", action="store_true",
                    help="Disable fp16 (debug only; slower, more VRAM).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    index_dir = cfg.resolve(cfg.paths.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    articles_file = Path("data/legal_articles.jsonl")
    if not articles_file.exists():
        raise SystemExit(f"Error: {articles_file} does not exist! Please run from the r2ai_pipeline root folder.")

    t0 = time.time()
    articles = load_articles_from_jsonl(articles_file)
    chunks = build_chunks(articles, split_long=cfg.chunk.split_long, max_chars=cfg.chunk.max_chars)
    print(f"[kaggle_jsonl] Loaded {len(articles):,} articles -> {len(chunks):,} chunks ({time.time()-t0:.0f}s)", flush=True)
    texts = [c.text for c in chunks]

    # --- BM25 (always build brand new) ---
    bm25_path = index_dir / "bm25.pkl"
    t0 = time.time()
    print("[kaggle_jsonl] building BM25 ...", flush=True)
    BM25Index().build(chunks).save(bm25_path)
    print(f"[kaggle_jsonl] BM25 saved ({time.time()-t0:.0f}s)", flush=True)

    # --- Dense (multi-GPU) ---
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from sentence_transformers import SentenceTransformer

    n_gpu = torch.cuda.device_count()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        devices = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu else ["cpu"]
    print(f"[kaggle_jsonl] visible GPUs={n_gpu}; encode devices={devices}", flush=True)

    model = SentenceTransformer(cfg.models.dense_model)
    model.max_seq_length = cfg.models.dense_max_seq_length
    use_fp16 = (not args.no_fp16) and any(d.startswith("cuda") for d in devices)
    if use_fp16:
        try:
            model = model.half()           # T4 OK (Turing fp16); NEVER bf16 here
            print("[kaggle_jsonl] fp16 enabled", flush=True)
        except Exception as e:
            print(f"[kaggle_jsonl] fp16 not applied ({e}) -> fp32", flush=True)

    t0 = time.time()
    if len([d for d in devices if d.startswith("cuda")]) > 1:
        print(f"[kaggle_jsonl] multi-process encode across {devices} ...", flush=True)
        pool = model.start_multi_process_pool(target_devices=devices)
        try:
            emb = model.encode_multi_process(
                texts, pool, batch_size=args.batch_size)
        finally:
            model.stop_multi_process_pool(pool)
    else:
        single = devices[0]
        print(f"[kaggle_jsonl] single-device encode on {single} ...", flush=True)
        model = model.to(single)
        emb = model.encode(texts, batch_size=args.batch_size,
                           show_progress_bar=True)

    emb = np.asarray(emb, dtype=np.float32)
    emb = _l2_normalize(emb)               # match DenseIndex.build(normalize=True)
    assert emb.shape[0] == len(chunks), \
        f"emb rows {emb.shape[0]} != chunks {len(chunks)}"
    print(f"[kaggle_jsonl] encoded {emb.shape} in {time.time()-t0:.0f}s", flush=True)

    # --- save in DenseIndex payload format ---
    payload = {"backend": "st", "model_name": cfg.models.dense_model,
               "chunks": chunks, "emb": emb, "sparse": False}
    dense_path = index_dir / "dense.pkl"
    with open(dense_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[kaggle_jsonl] dense.pkl saved: {emb.shape} -> {dense_path}", flush=True)
    print("[kaggle_jsonl] DONE. Download data/index/{bm25.pkl,dense.pkl} back to Mac.", flush=True)


if __name__ == "__main__":
    main()
