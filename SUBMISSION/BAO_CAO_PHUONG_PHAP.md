# Báo cáo phương pháp — R2AI 2026 (Vietnamese Legal IR & QA)

**Đội/Tác giả:** Đinh Xuân Công · **Cuộc thi:** BUILD AI LEGAL ASSISTANT (R2AI 2026)
**Kết quả public:** Articles **F2 = 0.4366** (Precision 0.362, Recall 0.553) · Docs F2 = 0.525
**Tóm tắt 1 dòng:** RAG truy hồi điều luật 3 tầng (BM25 + dense → RRF → cross-encoder rerank → chọn lọc tối ưu F2), answer **extractive**, mô hình **off-the-shelf không fine-tune**.

---

## 1. Bài toán & chỉ số

Cho 2000 câu hỏi pháp lý tiếng Việt, hệ thống phải trả về tập **Điều luật** căn cứ (và tập **Văn bản** chứa các Điều đó). BTC chấm **F2 macro** ở 2 cấp:

- `relevant_articles`: khoá so khớp `mã VB | tên VB | Điều N` (3 phần).
- `relevant_docs`: khoá `mã VB | tên VB` (2 phần).

F2 đặt trọng số **Recall gấp đôi Precision** (β=2). Đây là kim chỉ nam cho mọi quyết định thiết kế: **ưu tiên không bỏ sót** hơn là sạch nhiễu. Toàn bộ tinh chỉnh selection (max_k, threshold) đều nghiêng về Recall.

## 2. Dữ liệu (tóm tắt — chi tiết ở DATA.md)

Corpus = văn bản quy phạm pháp luật VN gốc vbpl.vn, gộp từ 2 dataset HuggingFace công khai (`undertheseanlp/UTS_VLC` 306 VB lõi sạch + `doanhieung/vbpl` 21,337 VB, import `--national-only`) cộng **vá thủ công 2 văn bản trọng yếu** từ bản RTF: Bộ luật Lao động `45/2019/QH14` (thay 220 Điều bản tiếng Anh → tiếng Việt) và Luật Đấu thầu `22/2023/QH15` (làm mới 86 Điều + bổ sung 9 Điều thiếu). Sau dedup + chunk theo Điều: **148,106 Điều** (`legal_articles.clean.jsonl`), 0 trùng id, 0 rò rỉ "Article N" tiếng Anh.

> Bài học dữ liệu: bản crawl gốc có 2 lỗi ngầm kéo điểm public — (a) Bộ luật Lao động là **bản tiếng Anh**, làm answer chứa "Article N" và lệch khỏi khoá tiếng Việt; (b) Luật Đấu thầu **truncate 27/96 Điều**. Vá 2 lỗi này nâng public **0.4239 → 0.4366** mà không hề đổi mô hình — chứng minh chất lượng corpus là đòn bẩy lớn nhất trước khi tối ưu retrieval.

## 3. Kiến trúc pipeline

Pipeline batch offline 2 pha: **build index một lần** → **inference 2000 câu**. Năm tầng:

### 3.1 Chunk theo Điều (`src/corpus/legal_chunker.py`)
Tách corpus theo đơn vị **Điều** (đúng đơn vị BTC chấm), giữ ngữ cảnh `Chương/Mục`, định danh chuẩn `mã VB | tên VB | Điều N`. Điều dài được tách tiếp theo **Khoản** (`split_long=true`, `max_chars=1200`) để vector không bị loãng ngữ nghĩa. 148,106 Điều → **333,420 chunk**.

### 3.2 Truy hồi kép — lexical + semantic
- **BM25** (`src/retrieval/bm25_index.py`): tokenize tiếng Việt bằng **pyvi** (word-segmentation quan trọng với tiếng Việt — "căn cứ" ≠ "căn" + "cứ"), lấy `bm25_top_n=100`. Bắt **trùng thuật ngữ pháp lý chính xác** (số hiệu VB, cụm định danh).
- **Dense** (`src/retrieval/dense_index.py`): `AITeamVN/Vietnamese_Embedding` (fine-tune tiếng Việt của BGE-M3, 1024-dim), encode fp16 trên T4, lấy `dense_top_n=100`. Bắt **diễn đạt khác chữ nhưng cùng nghĩa** (paraphrase).

Hai ranker bù trừ nhau: chẩn đoán cho thấy BM25-only ceiling (75.8%) ≈ dense-only (73.5%) nhưng **UNION = 76.9%** → mỗi ranker bắt được phần gold ranker kia bỏ sót.

### 3.3 Fusion — Reciprocal Rank Fusion (`src/retrieval/fusion.py`)
Gộp 2 bảng xếp hạng bằng **RRF** (`rrf_weights=[0.3, 0.7]` cho [bm25, dense], `rrf_k=30`). RRF cộng `weight / (rrf_k + rank)` theo từng ranker — bền với chênh lệch thang điểm giữa BM25 (dương, không chặn) và cosine dense ([-1,1]), không cần chuẩn hoá score. RRF bị chặn trên bởi UNION của input → là tầng hợp nhất, không phải tầng "phá trần".

### 3.4 Rerank — cross-encoder (`src/retrieval/reranker.py`)
Chấm lại `rerank_top_n=60` ứng viên đầu bằng `AITeamVN/Vietnamese_Reranker` (cross-encoder, đọc đồng thời cặp câu hỏi–Điều → chính xác hơn bi-encoder dense). Trên index đã vá, reranker cho lift **+2.7% recall@8** so với thứ tự RRF thuần → giữ trong pipeline.

### 3.5 Chọn lọc tối ưu F2 (`src/retrieval/retriever.py`)
Từ danh sách đã rerank, giữ các Điều có `score ≥ rel_threshold × top_score`, tối đa `max_k=8`, tối thiểu `min_k=1`. Cấu hình **lỏng** (`max_k=8`, `rel_threshold=0.3`, `rel_score_transform=none`) **ưu tiên Recall** đúng tinh thần F2. Đây là tầng "núm vặn" rẻ nhất (offline, không tốn GPU) và là nơi tinh chỉnh chính.

### 3.6 Sinh answer — extractive (`src/generation/answerer.py`)
Answer **trích thẳng** định danh Điều từ context (`llm_backend=extractive`), **không gọi LLM**. Lý do: chỉ số chấm trên `relevant_articles`/`relevant_docs` (định danh), không chấm văn xuôi answer → thêm LLM sinh chỉ tăng rủi ro hallucinate, chi phí GPU và độ trễ mà không ăn điểm. Quyết định loại bỏ generator là một lựa chọn **result-oriented** rõ ràng.

## 4. Phần cứng & quy trình chạy

Kaggle Notebook **2× NVIDIA T4 (16GB)**, build + inference đều fp16. Build index ~30–40 phút (một lần). Inference 2000 câu chia **2 shard song song** (mỗi GPU một nửa, merge theo `id`) ~8–15 phút. Toàn bộ lệnh tái hiện ở [README.md](README.md).

## 5. Kết quả & phân tích headroom

| Mốc (dev set 130 câu có gold, article-level) | Recall |
|---|---:|
| Điểm nộp RERANKED@max_k=8 | 68.1% |
| Trần cửa sổ RRF@60 | 71.9% |
| Trần 2-ranker UNION@500 | **76.9%** |
| BM25-only@500 / dense-only@500 | 75.8% / 73.5% |

Đọc bảng: khoảng cách 68.1% → 71.9% là gold **đã nằm trong top-60** nhưng rớt dưới max_k — lấy được bằng tinh chỉnh **selection** (miễn phí). Khoảng 71.9% → 76.9% là gold nằm sâu trong pool — cần nới window (tốn GPU). Phần **23% còn lại** tới 100% là **miss cấu trúc** (paraphrase/multi-hop mà cả lexical lẫn semantic đều không bắt) → chỉ phá được bằng **query-expansion (HyDE)** hoặc **đổi/fine-tune embedding** — để dành cho vòng sau.

> Lưu ý phương pháp luận: dev set 130 câu **không đại diện** cho public (dev≈0.57 vs public≈0.42). Vì test set của BTC chỉ có `id+question` (không nhãn vàng), mọi tinh chỉnh "khớp dev" có rủi ro overfit. Do đó các quyết định nộp đều ưu tiên **cấu hình lỏng, bền** thay vì siết theo dev.

## 6. Quyết định thiết kế then chốt (vì sao)

| Quyết định | Lý do |
|---|---|
| Off-the-shelf, **không fine-tune** | Không có tập huấn luyện gắn nhãn đại diện; fine-tune trên 130 dev pair = bẫy overfit. |
| **Extractive**, bỏ LLM generator | Chấm trên định danh, không chấm văn xuôi → LLM chỉ thêm rủi ro/chi phí. |
| Selection **lỏng** (max_k=8, thr=0.3) | F2 trọng số Recall ×2; siết chặt (max_k=10, thr=0.9) đã **regress** public 0.4366→0.4178. |
| Giữ reranker | Trên index đã vá cho **+2.7%@8** (dương rõ). |
| Ưu tiên **vá dữ liệu** trước tối ưu model | Vá 2 VB nâng public +0.0127, lớn hơn mọi đòn retrieval thử nghiệm. |

## 7. Tái lập

Toàn bộ pipeline tái hiện được end-to-end: tải `index/` đã build (kèm md5 trong `MANIFEST.json`) rồi chạy inference, **hoặc** build lại từ `legal_articles.clean.jsonl`. Mô hình HuggingFace tự tải khi chạy lần đầu. Lệnh chi tiết, tham số đầy đủ và checklist nhanh: xem [README.md](README.md). Nguồn & định dạng dữ liệu: [DATA.md](DATA.md). Mô hình & checkpoint: [MODELS.md](MODELS.md).
