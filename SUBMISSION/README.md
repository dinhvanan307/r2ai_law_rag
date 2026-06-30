# R2AI 2026 — Vietnamese Legal IR & QA — Hướng dẫn tái hiện (Reproduce)

Hệ thống RAG truy hồi điều luật (IR, chấm **F2 macro**) + trích dẫn căn cứ pháp lý cho cuộc thi
**BUILD AI LEGAL ASSISTANT (R2AI 2026)**. Pipeline batch offline 2 pha: **build index** một lần →
**inference** 2000 câu test → xuất `results.json` → đóng gói `submission.zip`.

- **Kết quả công khai (public leaderboard):** Articles **F2 = 0.4366** (Precision 0.362, Recall 0.553), Docs F2 = 0.525.
- **Mô hình:** dùng nguyên bản (off-the-shelf), **KHÔNG fine-tune** → "checkpoint" = chỉ số HuggingFace + index đã build.
- **Phần cứng tham chiếu:** Kaggle Notebook **2× NVIDIA T4 (16GB)**, build + inference đều fp16.

> Tài liệu liên quan trong thư mục này: [DATA.md](DATA.md) (nguồn & định dạng dữ liệu),
> [MODELS.md](MODELS.md) (mô hình & checkpoint), [BAO_CAO_PHUONG_PHAP.md](BAO_CAO_PHUONG_PHAP.md) (báo cáo phương pháp).

---

## 0. Đối chiếu yêu cầu nộp bài (BTC) → thành phần đã chuẩn bị

Theo *Quy trình và Hướng dẫn nộp bài* (mục 3.2 – 3.3), gói kiểm thử riêng gồm 5 thành phần:

| Yêu cầu BTC | Thành phần trong gói | Vị trí |
|---|---|---|
| File kết quả dự đoán (đúng định dạng) | `results.json` → `submission.zip` (phẳng, root-level) | sinh ở Bước 3–4 |
| Tài liệu mô tả dữ liệu (nguồn, cấu trúc, link tải) | [DATA.md](DATA.md) | thư mục này |
| Checkpoint / mô hình (kiến trúc, phiên bản, link, vị trí đặt) | [MODELS.md](MODELS.md) | thư mục này |
| Mã nguồn (config, script, requirements, cấu trúc thư mục) | `r2ai_code_bundle.zip` + [§7](#7-cấu-trúc-mã-nguồn) | thư mục này |
| Tài liệu hướng dẫn chạy lại (README đầy đủ) | tệp này | thư mục này |
| Báo cáo phương pháp (bổ sung) | [BAO_CAO_PHUONG_PHAP.md](BAO_CAO_PHUONG_PHAP.md) | thư mục này |

**Mốc thời gian (GMT+7):** Kiểm thử công khai kết thúc **30/06/2026 23:59**; Kiểm thử riêng
**01/07/2026 00:00 → 03/07/2026 00:00**. Mỗi người được chọn **tối đa 5 file kết quả** để chấm riêng.
Hệ thống nộp: <https://leaderboard.aiguru.com.vn/competitions/13/> → *My Submissions*.

**Tuân thủ mô hình/dữ liệu (mục 2.4):** mọi mô hình có **trọng số công khai, < 14B, phát hành trước
01/03/2026**, không dùng mô hình đóng (GPT-4o/Gemini). Chi tiết & nguồn dữ liệu: [MODELS.md](MODELS.md), [DATA.md](DATA.md).

---

## 0. Yêu cầu môi trường

| Thành phần | Phiên bản tham chiếu |
|---|---|
| OS | Ubuntu 22.04 (Kaggle) / macOS (dev) |
| Python | 3.10 – 3.11 |
| GPU | NVIDIA T4 16GB × 2 (build & inference); chạy được trên 1 GPU |
| CUDA | 12.x (torch cu128) |
| RAM | ≥ 16 GB |
| Đĩa trống | ≥ 8 GB (index ~2.9 GB + corpus ~0.3 GB + model cache ~3 GB) |

Cài đặt phụ thuộc:

```bash
cd r2ai_pipeline
python -m pip install -r requirements.txt          # core + tokenize tiếng Việt
python -m pip install -r requirements-gpu.txt       # torch / sentence-transformers / transformers (máy GPU)
```

Danh sách phụ thuộc đầy đủ: xem [`requirements.txt`](../requirements.txt) (core) và
[`requirements-gpu.txt`](../requirements-gpu.txt) (GPU). Không cần Ollama/vLLM cho bản nộp này
(answer sinh theo phương pháp **extractive**, không gọi LLM).

---

## 1. Lấy dữ liệu & checkpoint (qua đường link)

Tải gói artifact và đặt đúng vị trí (chi tiết nguồn trong [DATA.md](DATA.md), [MODELS.md](MODELS.md)):

| Artifact | Nội dung | Đặt vào | Link |
|---|---|---|---|
| `legal_articles.clean.jsonl` | Corpus 148,106 Điều (đã làm sạch + vá) | `data/` | _<điền link Google Drive/OneDrive>_ |
| `index/` (`bm25.pkl`, `dense.pkl`, `MANIFEST.json`) | Index đã build (≈2.9 GB) — **dùng để bỏ qua bước build** | `data/index/` | _<điền link>_ |
| `R2AIStage1DATA.json` | Bộ test 2000 câu của BTC | `data/test/` | _(BTC cung cấp)_ |

Mô hình HuggingFace tự tải khi chạy lần đầu (không cần upload thủ công):

```bash
# (tuỳ chọn) đặt token để tránh rate-limit
export HF_TOKEN=<hf_token_cua_ban>
```

- Embedding: `AITeamVN/Vietnamese_Embedding` (1024-dim)
- Reranker: `AITeamVN/Vietnamese_Reranker` (cross-encoder)

> Có **2 cách tái hiện**: (A) tải `index/` đã build rồi nhảy thẳng tới **Bước 3 (inference)**;
> hoặc (B) build lại từ `legal_articles.clean.jsonl` ở **Bước 2**. Kết quả nộp 0.4366 dùng đúng
> index ở bảng trên (md5 trong `data/index/MANIFEST.json`).

---

## 2. Build index (bỏ qua nếu đã tải `index/`)

Build tạo `data/index/bm25.pkl` (lexical) + `data/index/dense.pkl` (embedding) từ corpus.

**Trên Kaggle (2× T4 — khuyến nghị, ~30–40 phút):** dùng notebook
[`notebooks/build_index_kaggle_FINAL.ipynb`](../notebooks/build_index_kaggle_FINAL.ipynb).
Notebook tự: copy `src/` → đổi `legal_articles.clean.jsonl` thành file nguồn → build fp16 đa GPU → zip `index/`.

**Trên máy GPU (CLI tương đương):**

```bash
cd r2ai_pipeline
# script đọc cố định data/legal_articles.jsonl -> tạo bản sao từ clean.jsonl:
cp data/legal_articles.clean.jsonl data/legal_articles.jsonl
python scripts/build_index_from_jsonl.py        # -> data/index/{bm25.pkl, dense.pkl}
```

Số liệu tham chiếu: 148,106 Điều → **333,420 chunk** (tách Khoản cho Điều dài, `max_chars=1200`).
Kiểm tra md5 khớp `data/index/MANIFEST.json`.

---

## 3. Inference 2000 câu → `results.json`

Cấu hình nộp nằm ở [`config.yaml`](../config.yaml) (xem [Bảng tham số](#5-tham-số-pipeline-bản-nộp)).

**Trên Kaggle (2× T4, ~8–15 phút):** dùng [`notebooks/kaggle_inference.ipynb`](../notebooks/kaggle_inference.ipynb)
— tự dò index trong `/kaggle/input`, sinh `config.kaggle.yaml` (device=cuda, fp16), chạy song song
2 GPU (chia shard), merge theo `id`, rồi validate + đóng gói.

**Trên máy GPU (CLI tương đương, 1 GPU):**

```bash
cd r2ai_pipeline
python scripts/run_inference.py \
    --config config.yaml \
    --test data/test/R2AIStage1DATA.json \
    --out results.json
```

Chạy song song 2 GPU rồi gộp (tuỳ chọn, nhanh gấp đôi):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py --config config.yaml \
    --test data/test/R2AIStage1DATA.json --out part1.json --shard 1/2 &
CUDA_VISIBLE_DEVICES=1 python scripts/run_inference.py --config config.yaml \
    --test data/test/R2AIStage1DATA.json --out part2.json --shard 2/2 &
wait
# gộp 2 shard theo id (đã merge sẵn trong notebook; hoặc dùng đoạn Python ở mục 6)
```

---

## 4. Kiểm tra định dạng & đóng gói nộp

```bash
python scripts/validate_submission.py --results results.json --test data/test/R2AIStage1DATA.json
python scripts/make_submission.py     --results results.json --out submission.zip
```

`make_submission.py` tạo zip **phẳng** chỉ chứa `results.json` ngay tại thư mục gốc của file nén
(`submission.zip └── results.json`, **không** đặt trong thư mục con — đúng mục 2.2 BTC).
`validate_submission.py` kiểm tra schema từng record:

```jsonc
{
  "id": 1,                                  // int — khớp id câu hỏi trong test
  "question": "…",                          // str — GIỮ NGUYÊN, không sửa
  "answer": "…",                            // str — tiếng Việt, nhắc trực tiếp "Điều X"
  "relevant_docs":     ["<mã VB>|<tên VB>"],            // 2 phần
  "relevant_articles": ["<mã VB>|<tên VB>|Điều N"]      // 3 phần
}
```

**Quy định từng trường (mục 2.3 BTC)** — `<tên VB>` theo công thức **Loại văn bản + Mã văn bản + Trích yếu**:

```
relevant_docs[i]     = 04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa
relevant_articles[i] = 04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4
```

> ⚠️ **Lưu ý định dạng tên văn bản.** Bản nộp public 0.4366 xuất `<tên VB>` ở dạng rút gọn
> `Loại + Mã` (vd `Luật 04/2017/QH14`) do index dùng lúc đó có trường trích yếu rỗng — và vẫn
> đạt 0.4366, cho thấy **grader public normalize lỏng phần tên** (khớp chủ yếu trên *mã VB* + *Điều*).
> Hàm `src/corpus/doc_name.py::format_doc_name()` đã hỗ trợ sinh **đủ trích yếu**, và corpus
> `legal_articles.clean.jsonl` đã chứa trích yếu đầy đủ. **Khuyến nghị cho kiểm thử riêng:** build
> index từ `legal_articles.clean.jsonl` rồi chạy lại inference để `results.json` mang **tên đầy đủ
> đúng spec** — không rủi ro nếu grader normalize, an toàn hơn nếu grader siết chặt trên tập riêng.

---

## 5. Tham số pipeline (bản nộp)

| Tầng | File | Tham số bản nộp 0.4366 |
|---|---|---|
| Chunking | `src/corpus/legal_chunker.py` | `split_long=true`, `max_chars=1200` |
| BM25 (lexical) | `src/retrieval/bm25_index.py` | tokenize `pyvi`, `bm25_top_n=100` |
| Dense (semantic) | `src/retrieval/dense_index.py` | `AITeamVN/Vietnamese_Embedding`, 1024-dim, fp16, `dense_top_n=100` |
| Fusion | `src/retrieval/fusion.py` | RRF `rrf_weights=[0.3, 0.7]`, `rrf_k=30` |
| Reranker | `src/retrieval/reranker.py` | `AITeamVN/Vietnamese_Reranker` (cross-encoder), `rerank_top_n=60` |
| Chọn lọc F2 | `src/retrieval/retriever.py` | `min_k=1`, `max_k=8`, `rel_threshold=0.3`, `rel_score_transform=none` |
| Sinh answer | `src/generation/answerer.py` | `llm_backend=extractive` (không gọi LLM) |
| HyDE | — | `use_hyde=false` |

Chiến lược chọn lọc: giữ các Điều có `score ≥ rel_threshold × top_score`, tối đa `max_k`, tối thiểu
`min_k`. Cấu hình **lỏng** (max_k=8, threshold=0.3) ưu tiên **Recall** vì F2 trọng số Recall gấp đôi
Precision.

---

## 6. Phụ lục — gộp 2 shard theo id

```python
import json
a = {r["id"]: r for r in json.load(open("part1.json"))}
a.update({r["id"]: r for r in json.load(open("part2.json"))})
out = [a[k] for k in sorted(a)]
json.dump(out, open("results.json", "w"), ensure_ascii=False, indent=2)
print(len(out), "records")
```

---

## 7. Cấu trúc mã nguồn

```
r2ai_pipeline/
├── src/
│   ├── corpus/      legal_chunker · doc_name · loader        (ingest + chunk theo Điều)
│   ├── retrieval/   bm25_index · dense_index · fusion · reranker · retriever
│   ├── generation/  answerer · prompt · llm                  (extractive cho bản nộp)
│   ├── pipeline.py  postprocess.py  config.py  schema.py
├── scripts/
│   ├── build_index_from_jsonl.py          (build index từ jsonl)
│   ├── run_inference.py                    (sinh results.json, hỗ trợ --shard)
│   ├── validate_submission.py  make_submission.py
│   └── eval_local.py  eval_recall.py  tune_f2.py   (chấm offline & tinh chỉnh F2)
├── notebooks/
│   ├── build_index_kaggle_FINAL.ipynb      (build trên Kaggle 2×T4)
│   └── kaggle_inference.ipynb              (inference 2000 câu, 2 GPU)
├── data/
│   ├── legal_articles.clean.jsonl          (corpus 148,106 Điều)
│   ├── index/ {bm25.pkl, dense.pkl, MANIFEST.json}
│   └── test/  R2AIStage1DATA.json          (2000 câu)
├── config.yaml                             (cấu hình bản nộp)
├── requirements.txt  requirements-gpu.txt
└── SUBMISSION/  README.md · DATA.md · MODELS.md · BAO_CAO_PHUONG_PHAP.md
```

---

## 8. Checklist tái hiện nhanh

```bash
# 0. cài đặt
pip install -r requirements.txt -r requirements-gpu.txt
# 1. (cách A) tải index/ đã build vào data/index/   — HOẶC build lại (Bước 2)
# 2. inference
python scripts/run_inference.py --config config.yaml \
    --test data/test/R2AIStage1DATA.json --out results.json
# 3. validate + đóng gói
python scripts/validate_submission.py --results results.json --test data/test/R2AIStage1DATA.json
python scripts/make_submission.py --results results.json --out submission.zip
```
