"""Write LegalDoc objects to the corpus directory as pipeline-ready JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ...schema import LegalDoc


def _safe_name(law_id: str) -> str:
    return re.sub(r"[^0-9A-Za-zĐđ]+", "_", law_id).strip("_")


def save_doc(doc: LegalDoc, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_safe_name(doc.law_id)}.json"
    payload = {
        "law_id": doc.law_id,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "source_url": doc.source_url,
        "text": doc.raw_text,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
