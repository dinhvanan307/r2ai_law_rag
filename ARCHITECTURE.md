# Kiến trúc hệ thống — R2AI 2026 (Vietnamese Legal IR & QA)

RAG **batch offline 2 pha**: index corpus pháp luật một lần, rồi suy luận từng câu
hỏi trong test set → xuất `results.json`. Thiết kế **IR-first** (cuộc thi chấm
**F2 macro**, recall-weighted) + QA grounded có dẫn nguồn.

---

## Sơ đồ tổng thể

```
╔═ PHA 1 — INGEST + INDEX (offline, 1 lần) ══════════════════════════════╗
  HF datasets: UTS_VLC (306) + doanhieung/vbpl (21,337) ≈ 21.6k văn bản
      │  loader.py  (đọc json/txt, bỏ file rỗng)
      ▼
  Legal-aware Chunker  (corpus/legal_chunker.py)
      • tách theo ĐIỀU (đơn vị BTC chấm điểm), giữ ngữ cảnh Chương/Mục
      • parent-child: Điều dài → tách Khoản (max 1200 ký tự)
      • nuốt header markdown **Điều** / # / >
      • uid chuẩn = law_id|doc_name|Điều X  (khớp đáp án BTC)
      ▼
  ┌─────────────────────────┬──────────────────────────────────────┐
  │ BM25 (lexical)          │ Dense (semantic)                     │
  │ retrieval/bm25_index.py │ retrieval/dense_index.py             │
  │ tokenize tiếng Việt     │ st: AITeamVN/Vietnamese_Embedding    │
  │ (pyvi) — bắt số hiệu/Điều│ fallback: TF-IDF **sparse** (scale) │
  └─────────────────────────┴──────────────────────────────────────┘
            └──► data/index/{bm25.pkl, dense.pkl}
╚════════════════════════════════════════════════════════════════════════╝

╔═ PHA 2 — INFERENCE (mỗi câu hỏi) ══════════════════════════════════════╗
  question  (test.json)
      │  Retriever._rank()  (retrieval/retriever.py)
      ▼
  BM25 top-100 ┐
               ├─► RRF fusion (weights [bm25=0.3, dense=0.7], k=30)
  Dense top-100┘        │  gộp chunk → ĐIỀU (giữ điểm cao nhất)
                        ▼
  Cross-encoder rerank top-50  (AITeamVN/Vietnamese_Reranker)
                        ▼
  Chọn lọc theo F2  _select(): min_k=1, max_k=3, rel_threshold=0.3
                        ▼  K điều liên quan
  Answer EXTRACTIVE  (generation/answerer.py — KHÔNG gọi LLM cho bản nộp)
      • trích thẳng "Điều X của <Loại+Mã VB>" từ context (llm_backend=extractive)
      • backend LLM (Ollama/vLLM/hf) vẫn cắm được nhưng ngoài luồng nộp
                        ▼
  Postprocess (postprocess.py): build relevant_docs / relevant_articles
                        ▼
  results.json ─► validate_submission.py ─► make_submission.py ─► submission.zip
╚════════════════════════════════════════════════════════════════════════╝
```

---

## Bản đồ thành phần → file → cấu hình hiện tại

| Tầng | File | Cấu hình (config.yaml) |
|---|---|---|
| Tải corpus | `src/corpus/loader.py` | `paths.corpus_dir = data/corpus` |
| Định danh VB | `src/corpus/doc_name.py` | tên VB = Loại + Mã + Trích yếu |
| Chunking | `src/corpus/legal_chunker.py` | `chunk.split_long=true`, `max_chars=1200` |
| Lexical | `src/retrieval/bm25_index.py` | `bm25_top_n=100` |
| Dense | `src/retrieval/dense_index.py` | `dense_backend=st`, `dense_model=AITeamVN/Vietnamese_Embedding`, `dense_top_n=100` |
| Fusion | `src/retrieval/fusion.py` | `rrf_weights=[0.3,0.7]`, `rrf_k=30` |
| Rerank | `src/retrieval/reranker.py` | `reranker_backend=cross-encoder`, `AITeamVN/Vietnamese_Reranker`, `rerank_top_n=50` |
| Chọn lọc F2 | `src/retrieval/retriever.py` | `min_k=1`, `max_k=3`, `rel_threshold=0.3` |
| Sinh answer | `src/generation/answerer.py` | `llm_backend=extractive` (bản nộp — không gọi LLM) |
| Hậu xử lý | `src/postprocess.py` | uid → relevant_docs/articles |
| Điều phối | `src/pipeline.py` | `build_articles` (có `limit`), `Pipeline.answer_one` |

## Pipeline scripts
| Mục đích | Script |
|---|---|
| Build index | `scripts/build_index_from_jsonl.py` (bản resume: `_resumable`; Kaggle đa GPU: `_kaggle`) |
| Suy luận → results.json | `scripts/run_inference.py` (hỗ trợ `--shard`) |
| Kiểm tra định dạng | `scripts/validate_submission.py` |
| Đóng gói nộp | `scripts/make_submission.py` |
| Chấm offline (F2) | `scripts/eval_local.py`, `scripts/eval_recall.py` |
| Dò tham số F2 | `scripts/tune_f2.py` (grid-search `rel_threshold`/`max_k`/`rrf_k`) |

> Hệ thống dùng mô hình **off-the-shelf, không fine-tune**. Việc thu thập/crawl corpus
> và thử nghiệm huấn luyện nằm ngoài repo nộp; corpus được cung cấp sẵn dưới dạng
> `data/legal_articles.jsonl` để build lại chỉ mục.

## Lưu trữ corpus
Corpus được gộp về **1 file** `data/legal_articles.jsonl` (load nhanh, tránh bloat
git) thay vì hàng nghìn file rời. `loader.load_corpus()` nhận **cả file lẫn thư mục**;
trỏ `paths.corpus_dir` tới file gộp rồi build index. Dedup
theo `law_id` (giữ bản text dài nhất). *(Không ảnh hưởng định dạng nộp `results.json`.)*

## Nguyên tắc thiết kế
- **Đơn vị = Điều**: mọi truy hồi quy về Điều vì BTC so khớp `law_id|tên VB|Điều X`.
- **Hybrid > đơn lẻ**: BM25 (số hiệu/khớp từ) + Dense (ngữ nghĩa) → RRF (bền với chênh lệch thang điểm).
- **IR-first**: tối ưu retrieval (chấm tự động liên tục) trước; QA promote hàng tuần.
- **Pluggable + lite fallback**: chạy được không GPU (TF-IDF sparse + extractive) để CI/smoke test; bật model thật cho thi đấu.
- **Grounded**: answer bắt buộc trích Điều có trong context → ăn điểm "căn cứ" (grader trích Điều từ `answer`) + chống bịa.

## Tham chiếu tài liệu
`README.md` (tổng quan & quy trình tái tạo) · `SUBMISSION/README.md` (hướng dẫn tái hiện nộp BTC) ·
`SUBMISSION/DATA.md` (dữ liệu & nguồn) · `SUBMISSION/MODELS.md` (mô hình & checkpoint) ·
`SUBMISSION/BAO_CAO_PHUONG_PHAP.md` (báo cáo phương pháp).
