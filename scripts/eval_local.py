"""Local IR scorer replicating the competition's macro-F2 metric.

Given a predictions file (results.json) and a gold file with the same ids, it
computes macro Precision / Recall / **F2** at the article (Điều) level, matching
the competition's normalisation: an article identity is ``law_id|name|Điều X``
reduced to the normalised ``Điều X`` token within its document.

Gold file format:
    [{"id": 1, "relevant_articles": ["04/2017/QH14|Luật ...|Điều 4", ...]}, ...]

Usage:
    python scripts/eval_local.py --pred results.json --gold data/test/gold.json
"""
import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

DIEU_RE = re.compile(r"Điều\s+(\d+)")


def norm_article(entry: str) -> str | None:
    """Reduce ``law_id|name|Điều X`` to a comparable key ``law_id::Điều X``.

    We key on (document code, article number) which is robust to small
    differences in the human-readable ``name`` field.
    """
    parts = entry.split("|")
    if len(parts) < 3:
        return None
    law_id = parts[0].strip()
    m = DIEU_RE.search(parts[-1])
    if not m:
        return None
    return f"{law_id}::Điều {m.group(1)}"


def articles_from_answer(answer: str, gold_law_ids: set[str]) -> set[str]:
    """The grader also extracts 'Điều X' from the answer text. We approximate by
    pairing each Điều mention with any gold law id present in the answer (best
    effort; the official grader has the answer key)."""
    nums = {f"Điều {n}" for n in DIEU_RE.findall(answer or "")}
    keys = set()
    for lid in gold_law_ids:
        if lid in (answer or ""):
            for d in nums:
                keys.add(f"{lid}::{d}")
    return keys


def load_map(path: Path, field: str) -> dict[int, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("questions") or []
    return {row["id"]: (row.get(field) or []) for row in data}


def f2(p: float, r: float) -> float:
    if p == 0 and r == 0:
        return 0.0
    return (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Local macro-F2 IR scorer.")
    ap.add_argument("--pred", default="results.json")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--use-answer", action="store_true",
                    help="Also count Điều mentions found in the answer text")
    args = ap.parse_args()

    pred_articles = load_map(Path(args.pred), "relevant_articles")
    pred_answers = load_map(Path(args.pred), "answer") if args.use_answer else {}
    gold_articles = load_map(Path(args.gold), "relevant_articles")

    Ps, Rs, F2s = [], [], []
    for qid, gold in gold_articles.items():
        gold_keys = {k for e in gold if (k := norm_article(e))}
        gold_law_ids = {e.split("|")[0].strip() for e in gold}

        pred = pred_articles.get(qid, [])
        pred_keys = {k for e in pred if (k := norm_article(e))}
        if args.use_answer:
            ans = pred_answers.get(qid, "")
            pred_keys |= articles_from_answer(
                ans if isinstance(ans, str) else "", gold_law_ids)

        if not pred_keys:
            p = r = 0.0
        else:
            tp = len(pred_keys & gold_keys)
            p = tp / len(pred_keys)
            r = tp / len(gold_keys) if gold_keys else 0.0
        Ps.append(p)
        Rs.append(r)
        F2s.append(f2(p, r))

    n = len(F2s) or 1
    print(f"[eval] questions scored: {len(F2s)}")
    print(f"[eval] macro Precision: {sum(Ps)/n:.4f}")
    print(f"[eval] macro Recall:    {sum(Rs)/n:.4f}")
    print(f"[eval] macro F2:        {sum(F2s)/n:.4f}")


if __name__ == "__main__":
    main()
