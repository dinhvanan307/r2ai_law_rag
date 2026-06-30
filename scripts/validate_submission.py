"""Validate results.json against the competition format BEFORE submitting.

Checks:
* file is named results.json (warns otherwise);
* top-level is a non-empty list;
* every item has id (int), question (str), answer (str),
  relevant_docs (list[str]), relevant_articles (list[str]);
* relevant_docs entries match ``<code>|<name>`` (2 parts);
* relevant_articles entries match ``<code>|<name>|<điều>`` (3 parts) and the
  3rd part contains "Điều";
* (optional) all ids from a test file are present, none extra/duplicate.

Usage:
    python scripts/validate_submission.py --results results.json [--test data/test/test.json]
"""
import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

DIEU_RE = re.compile(r"Điều\s+\d+")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate results.json.")
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--test", default=None, help="Optional test file to cross-check ids")
    args = ap.parse_args()

    path = Path(args.results)
    errors: list[str] = []
    warnings: list[str] = []

    if path.name != "results.json":
        warnings.append(f"File name is '{path.name}', must be 'results.json' to be graded.")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        errors.append("Top-level must be a non-empty JSON list.")
        _report(errors, warnings)
        return

    ids: list = []
    for i, item in enumerate(data):
        loc = f"item[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{loc}: not an object")
            continue
        for key, typ in [("id", int), ("question", str), ("answer", str)]:
            if key not in item:
                errors.append(f"{loc}: missing '{key}'")
            elif not isinstance(item[key], typ):
                errors.append(f"{loc}: '{key}' must be {typ.__name__}")
        ids.append(item.get("id"))

        for key in ("relevant_docs", "relevant_articles"):
            if not isinstance(item.get(key), list):
                errors.append(f"{loc}: '{key}' must be a list")

        for d in item.get("relevant_docs", []) or []:
            if not isinstance(d, str) or len(d.split("|")) != 2:
                errors.append(f"{loc}: relevant_docs entry malformed (need '<code>|<name>'): {d!r}")
        for a in item.get("relevant_articles", []) or []:
            if not isinstance(a, str) or len(a.split("|")) != 3:
                errors.append(f"{loc}: relevant_articles entry malformed (need '<code>|<name>|<điều>'): {a!r}")
            elif not DIEU_RE.search(a.split("|")[-1]):
                warnings.append(f"{loc}: relevant_articles 3rd part has no 'Điều N': {a!r}")

        if not (item.get("relevant_articles")):
            warnings.append(f"{loc} (id={item.get('id')}): empty relevant_articles → 0 recall for this Q")
        if not DIEU_RE.search(item.get("answer", "")):
            warnings.append(f"{loc} (id={item.get('id')}): answer has no 'Điều N' (grader extracts from answer too)")

    # Duplicate ids
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        errors.append(f"Duplicate ids: {sorted(dupes)}")

    # Cross-check against test ids
    if args.test:
        test = json.loads(Path(args.test).read_text(encoding="utf-8"))
        if isinstance(test, dict):
            test = test.get("data") or test.get("questions") or []
        test_ids = {row["id"] for row in test}
        missing = test_ids - set(ids)
        extra = set(ids) - test_ids
        if missing:
            errors.append(f"Missing {len(missing)} ids from test set: {sorted(missing)[:10]}...")
        if extra:
            warnings.append(f"{len(extra)} extra ids not in test set: {sorted(extra)[:10]}...")

    _report(errors, warnings, n=len(data))


def _report(errors: list[str], warnings: list[str], n: int = 0) -> None:
    for w in warnings:
        print(f"  WARN: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print(f"\n[validate] FAILED with {len(errors)} error(s), {len(warnings)} warning(s).")
        raise SystemExit(1)
    print(f"\n[validate] PASSED — {n} items, {len(warnings)} warning(s).")


if __name__ == "__main__":
    main()
