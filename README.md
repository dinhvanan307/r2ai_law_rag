# R2AI 2026 — Vietnamese Legal IR & QA Pipeline

Hệ thống **RAG batch offline** truy hồi điều luật (Information Retrieval, chấm **F2 macro** — recall-weighted)
và trích dẫn căn cứ pháp lý cho cuộc thi **BUILD AI LEGAL ASSISTANT (R2AI 2026)** —
[leaderboard.aiguru.com.vn/competitions/13](https://leaderboard.aiguru.com.vn/competitions/13/).

Kiến trúc **2 pha**: *build index* corpus pháp luật một lần → *inference* lần lượt 2000 câu test →
xuất `results.json` → đóng gói `submission.zip`.

| | |
|---|---|
| **Kết quả public leaderboard** | Articles **F2 = 0.4366** (P 0.362 · R 0.553) · Docs F2 = 0.525 |
| **Mô hình** | Off-the-shelf, **KHÔNG fine-tune** — "checkpoint" = chỉ số HuggingFace + index đã build |
| **Phần cứng tham chiếu** | Kaggle Notebook **2× NVIDIA T4 16GB**, build + inference fp16 |
| **Sinh câu trả lời** | **Extractive** (không gọi LLM) — trích thẳng Điều liên quan từ context |
| **Tuân thủ thể lệ** | Mọi model có trọng số công khai, **< 14B**, phát hành **trước 01/03/2026** |

> 📖 **[SUBMISSION/README.md](SUBMISSION/README.md)** — hướng dẫn tái hiện đầy đủ (bản nộp BTC) ·
> **[ARCHITECTURE.md](ARCHITECTURE.md)** — kiến trúc & bản đồ thành phần ·
> **[SUBMISSION/BAO_CAO_PHUONG_PHAP.md](SUBMISSION/BAO_CAO_PHUONG_PHAP.md)** — báo cáo phương pháp & phân tích recall.

---

## TL;DR — chạy nhanh nhất

```bash
cd r2ai_pipeline
pip install -r requirements.txt -r requirements-gpu.txt        # cần GPU NVIDIA + CUDA 12.x
# Tải data/index/ đã build sẵn (link trong SUBMISSION/MODELS.md) -> data/index/
# Đặt bộ test BTC -> data/test/R2AIStage1DATA.json
python scripts/run_inference.py --config config.yaml \
    --test data/test/R2AIStage1DATA.json --out results.json    # ~8–15' trên 1 T4
python scripts/validate_submission.py --results results.json --test data/test/R2AIStage1DATA.json
python scripts/make_submission.py --results results.json --out submission.zip
```

Chi tiết từng bước, cách build lại index, và xử lý sự cố: xem các mục bên dưới.

---

## 1. Bài toán & cách chấm

Đầu vào là **2000 câu hỏi** pháp luật (lĩnh vực doanh nghiệp). Với mỗi câu, hệ thống phải trả về:

- `relevant_docs` — danh sách **văn bản** liên quan (định danh `mã VB | tên VB`).
- `relevant_articles` — danh sách **Điều** liên quan (định danh `mã VB | tên VB | Điều N`).
- `answer` — câu trả lời tiếng Việt có trích dẫn căn cứ.

Grader so khớp **định danh** (không chấm văn xuôi `answer` ở metric chính) và tính **F2 macro**:

```
F2 = (1 + 2²) · P · R / (2² · P + R)        # β = 2 → Recall nặng gấp 4 lần Precision
```

→ **Hệ quả thiết kế:** ưu tiên **Recall**. Thà trả thừa vài Điều (giảm P nhẹ) còn hơn bỏ sót (giảm R mạnh).
Đây là lý do mọi tham số chọn lọc đặt **lỏng** (xem [§7](#7-tham-số-pipeline--cách-tinh-chỉnh-f2)).
Vì metric chấm trên **định danh Điều/VB** chứ không chấm văn xuôi, hệ thống dùng **answer extractive**
(trích thẳng định danh, không gọi LLM) — thêm LLM chỉ tăng rủi ro hallucinate, chi phí GPU, độ trễ mà
không ăn thêm điểm.

**Đơn vị nguyên tử = Điều.** BTC so khớp ở cấp `Điều`, nên toàn bộ pipeline (chunk, retrieve, rerank,
select) đều quy về **Điều** thay vì đoạn văn tuỳ ý.

---

## 2. Kiến trúc

```
PHA 1 — INDEX (offline, 1 lần, ~30–40' trên 2×T4)
  corpus 148,106 Điều  (data/legal_articles.clean.jsonl)
    │  legal-aware chunker (src/corpus/legal_chunker.py)
    │    • tách theo ĐIỀU, giữ ngữ cảnh Chương/Mục
    │    • Điều dài > 1200 ký tự → tách tiếp theo Khoản (parent-child)
    │    • định danh chuẩn:  mã VB | tên VB | Điều N
    ▼  333,420 chunk
  ┌─────────────────────────────┬──────────────────────────────────────┐
  │ BM25 (lexical)              │ Dense (semantic)                     │
  │ tokenize pyvi               │ AITeamVN/Vietnamese_Embedding        │
  │ → bắt số hiệu VB / "Điều X" │ 1024-dim, fp16 → bắt paraphrase      │
  └─────────────────────────────┴──────────────────────────────────────┘
    ▼  data/index/{bm25.pkl, dense.pkl, MANIFEST.json}

PHA 2 — INFERENCE (mỗi câu hỏi)
  question
    │  BM25 top-100  +  Dense top-100
    ▼
  RRF fusion (weights [bm25=0.3, dense=0.7], k=30)   ← bền với chênh lệch thang điểm
    │  gộp chunk → Điều (giữ điểm cao nhất)
    ▼
  Cross-encoder rerank top-60 (AITeamVN/Vietnamese_Reranker)   ← đòn bẩy Precision
    │
    ▼
  Chọn lọc F2  (src/retrieval/retriever.py)
    │   giữ Điều có score ≥ rel_threshold × top_score; min_k=1, max_k=8
    ▼  K Điều liên quan
  Answer EXTRACTIVE (src/generation/answerer.py — không gọi LLM)
    │  trích "Điều X của <Loại + Mã VB>" từ context
    ▼
  Postprocess → relevant_docs / relevant_articles
    ▼
  results.json → validate_submission → make_submission → submission.zip
```

**Vì sao hybrid?** BM25 bắt khớp chính xác *số hiệu văn bản* và cụm "Điều X" (lexical); Dense bắt *diễn
đạt khác chữ cùng nghĩa* (semantic). **RRF** hợp nhất hai bảng xếp hạng theo *thứ hạng* (không theo điểm
thô) nên bền vững khi hai retriever có thang điểm lệch nhau. **Reranker** cross-encoder đọc đồng thời
(câu hỏi, Điều) để chấm lại tinh, nâng Precision của top.

Bản đồ chi tiết `tầng → file → tham số`: xem [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 3. Stack mô hình

| Vai trò | Model | Tham số | License | Vai trò trong pipeline |
|---|---|---|---|---|
| Embedding (dense) | [`AITeamVN/Vietnamese_Embedding`](https://huggingface.co/AITeamVN/Vietnamese_Embedding) | ~568M | Apache-2.0 | Vector 1024-dim cho Điều + câu hỏi; nền BGE-M3 fine-tune tiếng Việt/legal |
| Reranker | [`AITeamVN/Vietnamese_Reranker`](https://huggingface.co/AITeamVN/Vietnamese_Reranker) | ~568M | Apache-2.0 | Cross-encoder chấm lại top-60 ứng viên |
| Generator | — (không dùng) | — | — | Answer **extractive**, `llm_backend=extractive` |

Mô hình HuggingFace **tự tải** khi chạy lần đầu (qua `sentence-transformers`) — không cần upload thủ công.
Đặt `HF_TOKEN` để tránh rate-limit. Lý do chọn 2 model AITeamVN: đã fine-tune sẵn cho **tiếng Việt + pháp
lý**, mạnh ngay out-of-the-box, nhẹ (568M, chạy gọn trên T4 16GB), thoả mọi ràng buộc thể lệ.

---

## 4. Yêu cầu môi trường

| Thành phần | Phiên bản tham chiếu |
|---|---|
| OS | Ubuntu 22.04 (Kaggle) / macOS (dev) |
| Python | 3.10 – 3.11 |
| GPU | NVIDIA T4 16GB (chạy được 1 GPU; bản nộp dùng 2× để chia shard) |
| CUDA | 12.x (torch cu128) |
| RAM | ≥ 16 GB |
| Đĩa trống | ≥ 8 GB (index ~2.9 GB + corpus ~0.3 GB + model cache ~3 GB) |

```bash
cd r2ai_pipeline
python -m pip install -r requirements.txt        # core: pyvi, numpy, scikit-learn, pyyaml…
python -m pip install -r requirements-gpu.txt    # torch / sentence-transformers / transformers
export HF_TOKEN=<hf_token_cua_ban>               # (tuỳ chọn) tránh rate-limit khi tải model
```

> Không cần Ollama/vLLM/Qwen cho bản nộp này — answer sinh theo phương pháp **extractive**.
> Trên Mac dev có thể chạy `device: mps`; máy GPU NVIDIA dùng `device: cuda`; không GPU dùng `cpu` (chậm).

---

## 5. Dữ liệu

### 5.1 Định dạng corpus

`data/legal_articles.clean.jsonl` — mỗi dòng là **một Điều luật** (JSON Lines, UTF-8):

| Trường | Mô tả |
|---|---|
| `id` | Khoá tăng dần, duy nhất |
| `doc_num` | Mã hiệu văn bản — vd `03/2022/TT-BXD` |
| `doc_ref` | `mã VB \| tên đầy đủ VB` |
| `article_number` | Số hiệu Điều — vd `Điều 1` |
| `article_ref` | `mã VB \| tên VB \| Điều N` — **khoá so khớp đáp án BTC** |
| `text` | Toàn văn Điều (giữ tiêu đề `**Điều N. …**`, các Khoản/Điểm) |
| `text_length` | Độ dài ký tự của `text` |

Nguồn: 2 dataset HuggingFace (`undertheseanlp/UTS_VLC` + `doanhieung/vbpl`, gốc vbpl.vn) + 2 văn bản vá
thủ công. Chi tiết nguồn/license: [SUBMISSION/DATA.md](SUBMISSION/DATA.md).

### 5.2 Hai văn bản đã vá (quan trọng cho Recall)

- **Bộ luật Lao động `45/2019/QH14`** — thay 220 Điều bản tiếng Anh → tiếng Việt (loại rò rỉ "Article N").
- **Luật Đấu thầu `22/2023/QH15`** — làm mới 86 Điều + bổ sung 9 Điều thiếu `[15,26,32,33,70,73,83,84,85]`
  (Điều 71 khuyết thật trong bản gốc — ghi nhận, không bịa).

### 5.3 Cần đặt những file gì

| File | Nội dung | Vị trí | Lấy ở đâu |
|---|---|---|---|
| `legal_articles.clean.jsonl` | Corpus 148,106 Điều (~307 MB) | `data/` | link trong [DATA.md](SUBMISSION/DATA.md) |
| `index/` (`bm25.pkl`, `dense.pkl`, `MANIFEST.json`) | Index đã build (~2.9 GB) | `data/index/` | link trong [MODELS.md](SUBMISSION/MODELS.md) |
| `R2AIStage1DATA.json` | Bộ test 2000 câu | `data/test/` | BTC cung cấp |

> ⚠️ Các artifact nặng (`data/index/*.pkl`, `legal_articles.clean.jsonl`) **bị `.gitignore`** để repo không
> vượt giới hạn GitHub 100MB — tải qua link, không nằm trong repo.

---

## 6. Quy trình tái tạo (reproduce)

Có **2 nhánh**. Nhánh **A** nhanh (tải index sẵn, bỏ qua build); nhánh **B** đầy đủ (build lại từ corpus).
Bản nộp 0.4366 dùng đúng index có md5 ghi trong `data/index/MANIFEST.json`.

### Nhánh A — dùng index đã build (khuyến nghị để chấm lại)

```
Tải data/index/ → Bước 6.2 (inference) → 6.3 (validate + đóng gói)
```

### Nhánh B — build lại từ corpus

```
Đặt corpus → 6.1 (build index) → 6.2 (inference) → 6.3 (validate + đóng gói)
```

### 6.1 Build index *(bỏ qua nếu đi nhánh A)*

Tạo `data/index/bm25.pkl` + `dense.pkl` từ corpus.

**Cách 1 — Kaggle 2×T4 (khuyến nghị, ~30–40').** Mở notebook
[`notebooks/build_index_kaggle_FINAL.ipynb`](notebooks/build_index_kaggle_FINAL.ipynb), attach corpus làm
Kaggle Dataset, Run All. Notebook tự copy `src/` → đổi tên file nguồn → build fp16 đa GPU → zip `index/`.

**Cách 2 — CLI máy GPU (1 GPU):**

```bash
# script đọc cố định data/legal_articles.jsonl → tạo bản sao từ clean.jsonl:
cp data/legal_articles.clean.jsonl data/legal_articles.jsonl
python scripts/build_index_from_jsonl.py --config config.yaml
#   → data/index/{bm25.pkl, dense.pkl, MANIFEST.json}
#   log mong đợi: "Loaded 148106 articles" → "Generated 333420 chunks" → build BM25 → build dense
```

Máy yếu/ngắt giữa chừng: dùng bản resume `scripts/build_index_from_jsonl_resumable.py --shard-size 5000`
(ghi từng shard, chạy lại sẽ tiếp tục). Sau khi build, đối chiếu md5 với `data/index/MANIFEST.json`.

### 6.2 Inference 2000 câu → `results.json`

**Cách 1 — Kaggle 2×T4 (~8–15').** Notebook [`notebooks/kaggle_inference.ipynb`](notebooks/kaggle_inference.ipynb):
tự dò index trong `/kaggle/input`, sinh `config.kaggle.yaml` (device=cuda, fp16), chạy song song 2 GPU
(chia shard) rồi merge theo `id`, validate + đóng gói.

**Cách 2 — CLI 1 GPU:**

```bash
python scripts/run_inference.py \
    --config config.yaml \
    --test data/test/R2AIStage1DATA.json \
    --out results.json
```

**Cách 3 — CLI 2 GPU (nhanh gấp đôi, chia shard round-robin rồi gộp):**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py --config config.yaml \
    --test data/test/R2AIStage1DATA.json --out part1.json --shard 1/2 &
CUDA_VISIBLE_DEVICES=1 python scripts/run_inference.py --config config.yaml \
    --test data/test/R2AIStage1DATA.json --out part2.json --shard 2/2 &
wait
# gộp 2 shard theo id:
python - <<'PY'
import json
a = {r["id"]: r for r in json.load(open("part1.json"))}
a.update({r["id"]: r for r in json.load(open("part2.json"))})
out = [a[k] for k in sorted(a)]
json.dump(out, open("results.json","w"), ensure_ascii=False, indent=2)
print(len(out), "records")
PY
```

> Thử nhanh trên ít câu: thêm `--limit 20` để chạy 20 câu đầu (smoke test trước khi chạy full).

### 6.3 Kiểm tra định dạng & đóng gói

```bash
python scripts/validate_submission.py --results results.json --test data/test/R2AIStage1DATA.json
python scripts/make_submission.py     --results results.json --out submission.zip
```

`make_submission.py` tạo zip **phẳng** (`submission.zip └── results.json`, không thư mục con — đúng thể lệ).
`validate_submission.py` kiểm tra schema từng record và đối chiếu `id` khớp test set.

---

## 7. Tham số pipeline & cách tinh chỉnh F2

Tất cả nằm trong [`config.yaml`](config.yaml). Bảng tham số bản nộp 0.4366:

| Tầng | File | Tham số | Ý nghĩa |
|---|---|---|---|
| Chunking | `legal_chunker.py` | `split_long=true`, `max_chars=1200` | Điều > 1200 ký tự tách tiếp theo Khoản (≈460 token < 512, không truncate) |
| BM25 | `bm25_index.py` | tokenize `pyvi`, `bm25_top_n=100` | lấy 100 ứng viên lexical |
| Dense | `dense_index.py` | `AITeamVN/Vietnamese_Embedding`, 1024-dim, fp16, `dense_top_n=100` | 100 ứng viên semantic |
| Fusion | `fusion.py` | RRF `weights=[0.3, 0.7]`, `k=30` | dense nặng hơn lexical 0.7/0.3 |
| Reranker | `reranker.py` | `AITeamVN/Vietnamese_Reranker`, `rerank_top_n=60` | chấm lại 60 ứng viên đầu |
| Chọn lọc F2 | `retriever.py` | `min_k=1`, `max_k=8`, `rel_threshold=0.3` | giữ Điều có `score ≥ 0.3 × top_score`, 1–8 Điều |
| Answer | `answerer.py` | `llm_backend=extractive` | trích định danh, không gọi LLM |

**Chiến lược chọn lọc:** sau rerank, giữ các Điều có `score ≥ rel_threshold × top_score`, tối đa `max_k`,
tối thiểu `min_k`. Cấu hình **lỏng** (max_k=8, threshold=0.3) ưu tiên **Recall** — phù hợp F2 (β=2).

**Tinh chỉnh:** dùng `scripts/tune_f2.py --test-file <dev_test.json> --gold-file <dev_gold.json>` để
grid-search `rel_threshold`/`max_k`/`rrf_k` trên một dev set có gold. ⚠️ *Lưu ý:* dev set tự sinh **không
hẳn đại diện** cho phân phối test thật — đã quan sát trường hợp sweep tinh chỉnh **cải thiện dev nhưng
giảm public LB**. Khi tune, luôn đối chiếu lại public leaderboard, đừng tin tuyệt đối dev.

---

## 8. Đánh giá & kết quả

| Metric (public leaderboard) | Giá trị |
|---|---|
| Articles **F2 macro** | **0.4366** (Precision 0.362 · Recall 0.553) |
| Docs F2 macro | 0.525 |

**Chấm offline** (cần file gold định danh):

```bash
python scripts/eval_local.py  --pred results.json --gold data/test/gold.json              # F2 macro
python scripts/eval_local.py  --pred results.json --gold data/test/gold.json --use-answer # tính cả Điều nhắc trong answer
python scripts/eval_recall.py --test data/test/dev_test.json --gold data/test/dev_gold.json \
    --ks 5,10,20,50,100 --with-rerank                                                      # recall@k theo độ sâu
```

`eval_recall.py` đo **trần recall** ở từng độ sâu K — công cụ chẩn đoán nút thắt: phân biệt phần gold
"đã trong pool nhưng rớt do chọn lọc" (sửa được miễn phí bằng tinh chỉnh selection) với phần "nằm sâu/miss
cấu trúc" (cần nới window hoặc đổi embedding). Phân tích recall-ceiling chi tiết: xem
[SUBMISSION/BAO_CAO_PHUONG_PHAP.md](SUBMISSION/BAO_CAO_PHUONG_PHAP.md).

---

## 9. Tham chiếu script

| Script | Mục đích | Flag chính |
|---|---|---|
| `build_index_from_jsonl.py` | Build BM25 + dense từ `data/legal_articles.jsonl` | `--config`, `--limit` |
| `build_index_from_jsonl_resumable.py` | Như trên, ghi từng shard (resume khi ngắt) | `--config`, `--shard-size` (mặc định 5000) |
| `build_index_kaggle_from_jsonl.py` | Bản dùng trong notebook Kaggle (đa GPU) | (gọi từ notebook) |
| `run_inference.py` | Sinh `results.json` từ test set | `--config`, `--test`, `--out`, `--limit`, `--shard i/n` |
| `validate_submission.py` | Kiểm tra schema + đối chiếu id | `--results`, `--test` |
| `make_submission.py` | Đóng gói `results.json` thành zip phẳng | `--results`, `--out` |
| `eval_local.py` | Chấm F2 macro offline | `--pred`, `--gold` (bắt buộc), `--use-answer` |
| `eval_recall.py` | Recall@k diagnostic theo độ sâu | `--test`, `--gold`, `--ks`, `--with-rerank`, `--config` |
| `tune_f2.py` | Grid-search tham số chọn lọc F2 | `--test-file`, `--gold-file` |

Hai notebook Kaggle: `build_index_kaggle_FINAL.ipynb` (build index 2×T4) và `kaggle_inference.ipynb`
(inference 2000 câu, 2 GPU + merge + đóng gói).

---

## 10. Định dạng `results.json`

```jsonc
{
  "id": 1,                                       // int — khớp id câu hỏi trong test
  "question": "…",                               // str — GIỮ NGUYÊN, không sửa
  "answer": "…",                                 // str — tiếng Việt, nhắc trực tiếp "Điều X"
  "relevant_docs":     ["<mã VB>|<tên VB>"],            // 2 phần
  "relevant_articles": ["<mã VB>|<tên VB>|Điều N"]      // 3 phần
}
```

`<tên VB>` theo công thức **Loại văn bản + Mã văn bản + Trích yếu**, ví dụ:
`04/2017/QH14|Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4`.

---

## 11. Xử lý sự cố

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `CUDA out of memory` khi build/inference | Giảm `dense_batch_size` trong `config.yaml` (64 → 16 → 8); đảm bảo fp16 bật |
| Tải model bị `429 / rate limit` | Đặt `export HF_TOKEN=…`; hoặc `snapshot_download(...)` trước rồi chạy offline |
| `build_index_from_jsonl.py` báo không thấy file | Script đọc cố định `data/legal_articles.jsonl` → `cp` từ `clean.jsonl` (xem 6.1) |
| Không có GPU | Đặt `device: cpu` trong `config.yaml` (chạy được nhưng rất chậm cho 2000 câu) |
| Mac Apple Silicon | `device: mps`; nếu lỗi buffer attention, giảm `dense_max_seq_length`/`batch_size` |
| md5 index không khớp `MANIFEST.json` | Bản corpus/đoạn chunk khác bản nộp → build lại từ đúng `legal_articles.clean.jsonl` |
| 2 shard không gộp đúng | Dùng đoạn merge theo `id` ở [§6.2 Cách 3](#62-inference-2000-câu--resultsjson) |

---

## 12. Cấu trúc thư mục

```
r2ai_pipeline/
├── src/
│   ├── corpus/       legal_chunker · doc_name · loader · crawl/   (ingest + chunk theo Điều)
│   ├── retrieval/    bm25_index · dense_index · fusion · reranker · retriever · text_utils
│   ├── generation/   answerer · prompt · llm                      (extractive cho bản nộp)
│   ├── pipeline.py · postprocess.py · config.py · schema.py
├── scripts/          build_index_from_jsonl[_resumable|_kaggle] · run_inference ·
│                     validate_submission · make_submission · eval_local · eval_recall · tune_f2
├── notebooks/        build_index_kaggle_FINAL.ipynb · kaggle_inference.ipynb   (Kaggle 2×T4)
├── data/
│   ├── legal_articles.clean.jsonl    (corpus 148,106 Điều — gitignored, tải qua link)
│   ├── index/ {bm25.pkl, dense.pkl, MANIFEST.json}   (gitignored)
│   └── seeds/ · test/R2AIStage1DATA.json · gold.json · test.json
├── config.yaml · config.first_submit.yaml
├── requirements.txt · requirements-gpu.txt
├── run_local.sh · first_submit_mac.sh
└── SUBMISSION/       README.md · DATA.md · MODELS.md · BAO_CAO_PHUONG_PHAP.md · docx/
```

---

## 13. Tài liệu liên quan

| Tài liệu | Nội dung |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Sơ đồ kiến trúc & bản đồ `tầng → file → tham số` |
| [SUBMISSION/README.md](SUBMISSION/README.md) | Hướng dẫn tái hiện đầy đủ (đối chiếu yêu cầu BTC, môi trường, link artifact) |
| [SUBMISSION/DATA.md](SUBMISSION/DATA.md) | Nguồn, cấu trúc, link tải dữ liệu |
| [SUBMISSION/MODELS.md](SUBMISSION/MODELS.md) | Kiến trúc, phiên bản, link & vị trí đặt model/checkpoint |
| [SUBMISSION/BAO_CAO_PHUONG_PHAP.md](SUBMISSION/BAO_CAO_PHUONG_PHAP.md) | Báo cáo phương pháp & phân tích recall-ceiling |
| [SUBMISSION/docx/](SUBMISSION/docx/) | Bản `.docx` của 4 tài liệu nộp BTC |
```
