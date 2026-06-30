# Tài liệu mô hình & checkpoint — R2AI 2026

## 1. Mô hình sử dụng trong bản nộp (0.4366)

Tất cả mô hình **dùng nguyên bản (off-the-shelf), KHÔNG fine-tune**. Đều thoả điều kiện BTC:
mã nguồn mở, < 14B tham số, ra mắt trước 2026-03-01.

| Vai trò | Model | HuggingFace | Tham số | License | Vai trò trong pipeline |
|---|---|---|---|---|---|
| Embedding (dense) | `AITeamVN/Vietnamese_Embedding` | https://huggingface.co/AITeamVN/Vietnamese_Embedding | ~568M | Apache-2.0 | Sinh vector 1024-dim cho Điều + câu hỏi (semantic retrieval) |
| Reranker | `AITeamVN/Vietnamese_Reranker` | https://huggingface.co/AITeamVN/Vietnamese_Reranker | ~568M | Apache-2.0 | Cross-encoder chấm lại top-60 ứng viên |
| Generator | — (không dùng) | — | — | — | Answer sinh theo **extractive** (trích Điều từ context), không gọi LLM |

> `AITeamVN/Vietnamese_Embedding` là bản fine-tune trên tiếng Việt của kiến trúc **BGE-M3**.
> Bản nộp **không** dùng LLM sinh (`llm_backend=extractive`) nên không cần Ollama/vLLM/Qwen.

## 2. Checkpoint

Vì **không fine-tune**, không có checkpoint trọng số tự huấn luyện. "Checkpoint" để tái hiện gồm:

1. **Snapshot mô hình HuggingFace** — tự tải khi chạy lần đầu. Để cố định phiên bản, ghim revision:

   ```python
   # mô hình tải tự động qua sentence-transformers; ghim commit nếu cần reproduce tuyệt đối
   from huggingface_hub import snapshot_download
   snapshot_download("AITeamVN/Vietnamese_Embedding")   # thêm revision="<commit_sha>" nếu muốn pin
   snapshot_download("AITeamVN/Vietnamese_Reranker")
   ```

2. **Index đã build** = artifact suy luận trực tiếp (tránh build lại ~40 phút GPU):

   | File | Kích thước | md5 (`data/index/MANIFEST.json`) |
   |---|---|---|
   | `bm25.pkl` | ~937 MB | `eb495e488a13d9aade1671e87b8ea903` |
   | `dense.pkl` | ~1.96 GB | `9ea16a779709f903e04ad54da1fc4860` |

## 3. Đường link tải checkpoint/index

| Artifact | Link |
|---|---|
| `index/` (bm25.pkl + dense.pkl + MANIFEST.json) | _<điền link Google Drive/OneDrive>_ |
| (tuỳ chọn) cache mô hình HF đã tải | _<điền link hoặc để hệ thống tự tải>_ |

> Đặt quyền **Anyone with the link → Viewer**. Sau khi tải, đặt vào `data/index/` và đối chiếu md5.

## 4. Cấu hình runtime của mô hình (bản nộp)

```yaml
models:
  dense_backend: st
  dense_model: AITeamVN/Vietnamese_Embedding
  dense_max_seq_length: 512
  dense_batch_size: 64
  dense_fp16: true                       # T4 fp16
  reranker_backend: cross-encoder
  reranker_model: AITeamVN/Vietnamese_Reranker
  llm_backend: extractive                # KHÔNG gọi LLM
  device: cuda
```

## 5. Tuân thủ quy định cuộc thi

| Tiêu chí BTC | Trạng thái |
|---|---|
| Mã nguồn mở | ✅ Apache-2.0 |
| < 14B tham số | ✅ ~568M mỗi model |
| Ra mắt trước 2026-03-01 | ✅ 2024 |
| Không dùng model đóng (GPT-4o/Gemini…) trong pipeline nộp | ✅ |
