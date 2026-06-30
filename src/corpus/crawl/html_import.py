"""Import documents from locally-saved HTML files (any source).

Use this for sources that require login (e.g. luatvietnam.vn): the user opens
the page in their own authenticated browser, saves it (Ctrl+S → "Webpage,
HTML Only"), and drops the file in a folder. No automation of login or paywall
is performed — this only parses files the user already has access to.

Pairing metadata: a sidecar ``manifest.json`` maps each HTML filename to its
metadata, or the filename convention ``<law_id>__<doc_type>__<title>.html``
is used (replace ``/`` with ``-`` in law_id).
"""
from __future__ import annotations

import json
from pathlib import Path

from ...schema import LegalDoc
from ..doc_name import infer_doc_type_from_law_id
from .normalize import html_to_text


def _meta_from_filename(path: Path) -> dict:
    stem = path.stem
    law_id = doc_type = title = ""
    if "__" in stem:
        parts = stem.split("__")
        law_id = parts[0].replace("-", "/").strip()
        doc_type = parts[1].strip() if len(parts) > 1 else ""
        title = parts[2].strip() if len(parts) > 2 else ""
    if not doc_type:
        doc_type = infer_doc_type_from_law_id(law_id)
    return {"law_id": law_id, "doc_type": doc_type, "title": title}


def import_html_dir(
    html_dir: str | Path,
    manifest_path: str | Path | None = None,
    content_selector: str | None = None,
) -> list[LegalDoc]:
    """Parse every .html/.htm in a directory into LegalDoc objects."""
    root = Path(html_dir)
    manifest: dict[str, dict] = {}
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    docs: list[LegalDoc] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".html", ".htm"}:
            continue
        meta = manifest.get(path.name) or _meta_from_filename(path)
        if not meta.get("law_id"):
            print(f"[import] skip {path.name}: no law_id (add to manifest.json "
                  f"or rename '<law_id>__<doc_type>__<title>.html')")
            continue
        html = path.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html, content_selector=content_selector)
        docs.append(LegalDoc(
            law_id=str(meta["law_id"]).strip(),
            doc_type=str(meta.get("doc_type") or infer_doc_type_from_law_id(meta["law_id"])).strip(),
            title=str(meta.get("title", "")).strip(),
            raw_text=text,
            source_url=meta.get("source_url", str(path)),
        ))
    return docs
