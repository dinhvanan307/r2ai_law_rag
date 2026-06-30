"""Smoke tests: chunker correctness + end-to-end format guarantees.

Run:  python -m pytest tests/ -q     (or)     python tests/test_smoke.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus.doc_name import format_doc_name
from src.corpus.legal_chunker import parse_document, normalize_article_no
from src.schema import LegalDoc


def test_doc_name_formula():
    name = format_doc_name("luật", "04/2017/QH14", "Luật Hỗ trợ doanh nghiệp nhỏ và vừa")
    assert name == "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa"


def test_normalize_article_no():
    assert normalize_article_no("Điều 4.") == "Điều 4"
    assert normalize_article_no(12) == "Điều 12"


def test_chunker_splits_articles():
    doc = LegalDoc(
        law_id="00/2020/QH14", doc_type="Luật", title="Luật Thử",
        raw_text="Chương I\nĐiều 1. A\n1. noi dung.\nĐiều 2. B\n1. noi dung khac.",
    )
    arts = parse_document(doc)
    assert len(arts) == 2
    assert arts[0].article_no == "Điều 1"
    assert arts[0].chuong.startswith("Chương")
    assert arts[1].article_uid.endswith("|Điều 2")


def test_article_uid_format():
    doc = LegalDoc(law_id="04/2017/QH14", doc_type="Luật",
                   title="Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                   raw_text="Điều 4. Tiêu chí\n1. abc.")
    a = parse_document(doc)[0]
    assert a.article_uid == ("04/2017/QH14|Luật 04/2017/QH14 "
                             "Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4")
    assert a.doc_uid == "04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print("All smoke tests passed.")
