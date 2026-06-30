"""Load a raw legal corpus from disk into ``LegalDoc`` objects.

Supported layouts (auto-detected per file in ``data/corpus/**``):

1. JSON object  -> one document::

       {"law_id": "04/2017/QH14", "doc_type": "Luật",
        "title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
        "text": "Điều 1. ...", "source_url": "..."}

2. JSON list of such objects -> many documents.

3. Plain ``.txt`` -> one document; metadata parsed from the filename
   ``<law_id>__<doc_type>__<title>.txt`` (use ``-`` for ``/`` in law_id),
   or from a leading metadata header if present.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import LegalDoc
from .doc_name import infer_doc_type_from_law_id


def _doc_from_obj(obj: dict) -> LegalDoc:
    law_id = str(obj.get("law_id") or obj.get("id") or "").strip()
    doc_type = str(obj.get("doc_type") or "").strip() or infer_doc_type_from_law_id(law_id)
    title = str(obj.get("title") or obj.get("name") or obj.get("trich_yeu") or "").strip()
    text = str(obj.get("text") or obj.get("raw_text") or obj.get("content") or "")
    return LegalDoc(
        law_id=law_id,
        doc_type=doc_type,
        title=title,
        raw_text=text,
        source_url=str(obj.get("source_url") or obj.get("url") or ""),
        meta={k: v for k, v in obj.items()
              if k not in {"law_id", "id", "doc_type", "title", "name",
                           "trich_yeu", "text", "raw_text", "content",
                           "source_url", "url"}},
    )


def _doc_from_txt(path: Path) -> LegalDoc:
    text = path.read_text(encoding="utf-8")
    stem = path.stem
    law_id = doc_type = title = ""
    if "__" in stem:
        parts = stem.split("__")
        law_id = parts[0].replace("-", "/").strip()
        if len(parts) > 1:
            doc_type = parts[1].strip()
        if len(parts) > 2:
            title = parts[2].strip()
    if not doc_type:
        doc_type = infer_doc_type_from_law_id(law_id)
    return LegalDoc(law_id=law_id, doc_type=doc_type, title=title, raw_text=text,
                    source_url=str(path))


def _docs_from_jsonl(path: Path) -> list[LegalDoc]:
    """Stream a JSONL file (one document object per line) — low memory."""
    docs: list[LegalDoc] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                docs.append(_doc_from_obj(obj))
    return docs


def _docs_from_parquet(path: Path) -> list[LegalDoc]:
    """Read a Parquet corpus (needs pyarrow). Columns map via ``_doc_from_obj``."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Reading .parquet needs pyarrow: pip install pyarrow") from e
    table = pq.read_table(path)
    return [_doc_from_obj(row) for row in table.to_pylist()]


def _load_file(path: Path) -> list[LegalDoc]:
    """Load one corpus file; supports .json / .jsonl / .parquet / .txt / .md."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _docs_from_jsonl(path)
    if suffix == ".parquet":
        return _docs_from_parquet(path)
    if suffix == ".json":
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            print(f"[loader] skip empty JSON: {path}")
            return []
        data = json.loads(raw)
        if isinstance(data, list):
            return [_doc_from_obj(o) for o in data if isinstance(o, dict)]
        if isinstance(data, dict):
            return [_doc_from_obj(data)]
        return []
    if suffix in {".txt", ".md"}:
        return [_doc_from_txt(path)]
    return []


def load_corpus(corpus_path: str | Path) -> list[LegalDoc]:
    """Load a legal corpus from a directory OR a single consolidated file.

    * **Directory** → recursively loads every ``.json/.jsonl/.parquet/.txt/.md``.
    * **Single file** (``corpus.jsonl`` / ``corpus.parquet`` / ``.json``) →
      loads just that file (preferred for large corpora — see consolidate_corpus.py).
    """
    root = Path(corpus_path)
    docs: list[LegalDoc] = []
    if not root.exists():
        return docs

    if root.is_file():
        docs = _load_file(root)
    else:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                docs.extend(_load_file(path))

    # Drop empty docs.
    return [d for d in docs if d.raw_text.strip() and d.law_id]
