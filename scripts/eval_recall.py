"""Recall@k diagnostic — decide whether to train the RERANKER or the EMBEDDING first.

Measures, on a dev set, how often the gold article(s) appear in the top-k
*candidate pool* of the current retriever (BM25 + dense → RRF).

Interpretation:
* **Recall@100 high (>~0.9)** but final F2 low  → the candidate pool already
  contains the gold; the bottleneck is **ranking** → train the **RERANKER** first.
* **Recall@100 low**                            → the gold is missing from the
  pool; the bottleneck is **retrieval** → improve the **EMBEDDING** (and RRF/BM25).

By default reranking is DISABLED here so we measure the raw retrieval ceiling
(fast, and rerank only reorders — it can't add missing articles). Use
``--with-rerank`` to also see post-rerank recall.

Usage:
    python scripts/eval_recall.py --test data/test/dev_test.json \
        --gold data/test/dev_gold.json --ks 5,10,20,50,100
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import load_config
from src.pipeline import load_retriever

DIEU_RE = re.compile(r"Điều\s+(\d+)")


def art_key(uid: str) -> str | None:
    parts = uid.split("|")
    if len(parts) < 2:
        return None
    m = DIEU_RE.search(parts[-1])
    return f"{parts[0].strip()}::Điều {m.group(1)}" if m else None


def load_map(path: str, field: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("questions") or []
    return {r["id"]: (r.get(field) or []) for r in data}


def main() -> None:
    ap = argparse.ArgumentParser(description="Recall@k retrieval diagnostic.")
    ap.add_argument("--test", default="data/test/dev_test.json")
    ap.add_argument("--gold", default="data/test/dev_gold.json")
    ap.add_argument("--ks", default="5,10,20,50,100")
    ap.add_argument("--with-rerank", action="store_true",
                    help="Measure recall AFTER reranking (default: raw pool)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.ks.split(",")})
    maxk = max(ks)

    cfg = load_config(args.config)
    retr = load_retriever(cfg)
    if not args.with_rerank:
        retr.use_reranker = False

    questions = load_map(args.test, "question")
    gold = load_map(args.gold, "relevant_articles")

    recalls = {k: [] for k in ks}
    scored_q = 0
    for qid, q in questions.items():
        g = {k for e in gold.get(qid, []) if (k := art_key(e))}
        if not g:
            continue
        scored_q += 1
        cands = retr.candidates(q, top_n=maxk)
        pred = [art_key(a.article_uid) for a in cands]
        for k in ks:
            hit = len(set(pred[:k]) & g)
            recalls[k].append(hit / len(g))

    print(f"[recall] dev queries scored: {scored_q}  "
          f"(rerank: {'ON' if args.with_rerank else 'OFF — raw pool'})")
    print(f"\n{'K':>6} | {'Recall@K (macro)':>16}")
    print("-" * 27)
    n = scored_q or 1
    for k in ks:
        r = sum(recalls[k]) / n
        print(f"{k:>6} | {r:>16.4f}")

    ceiling = sum(recalls[maxk]) / n
    print("\n[recall] Diễn giải:")
    if ceiling >= 0.9:
        print(f"  Recall@{maxk} = {ceiling:.3f} (CAO) → pool ứng viên đã chứa gold."
              f"\n  ⇒ Nút thắt là XẾP HẠNG → train RERANKER trước.")
    else:
        print(f"  Recall@{maxk} = {ceiling:.3f} (THẤP) → gold lọt khỏi pool."
              f"\n  ⇒ Nút thắt là TRUY HỒI → cải thiện EMBEDDING (+ RRF/BM25) trước.")


if __name__ == "__main__":
    main()
