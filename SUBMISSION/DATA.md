# Tài liệu mô tả dữ liệu — R2AI 2026

## 1. Nguồn dữ liệu (corpus pháp luật)

Toàn bộ corpus là **văn bản quy phạm pháp luật Việt Nam**, gốc từ Cổng thông tin
vbpl.vn, lấy qua 2 dataset công khai trên HuggingFace cộng 2 văn bản vá thủ công.

| Nguồn | Link | Số VB dùng | Loại văn bản chính | Ngày tải | License / ghi chú |
|---|---|---:|---|---|---|
| undertheseanlp/UTS_VLC | https://huggingface.co/datasets/undertheseanlp/UTS_VLC | 306 | Hiến pháp, Bộ luật, Luật | 2026-06-18 | MIT; gốc vbpl.vn; corpus lõi sạch |
| doanhieung/vbpl | https://huggingface.co/datasets/doanhieung/vbpl | 21,337 | Quyết định, Thông tư, Nghị định, Nghị quyết, Chỉ thị… | 2026-06-18 | License `other`; gốc vbpl.vn; import `--national-only`, bỏ VB hết hiệu lực rõ ràng |
| Vá thủ công | vbpl.vn / thuvienphapluat (RTF) | 2 | Bộ luật Lao động `45/2019/QH14`; Luật Đấu thầu `22/2023/QH15` | 2026-06-29 | bổ sung bản tiếng Việt + đủ Điều (xem §4) |

> **Lưu ý pháp lý:** văn bản quy phạm pháp luật của Nhà nước **không thuộc đối tượng bảo hộ quyền
> tác giả** (Luật SHTT Việt Nam). Vẫn khai báo nguồn để minh bạch và tái lập.

**Tổng hợp sau khi gộp + dedup + chunk theo Điều:** `legal_articles.clean.jsonl` = **148,106 Điều**.

## 2. Đường link truy cập dữ liệu

| Artifact | Kích thước | Link |
|---|---|---|
| `legal_articles.clean.jsonl` (corpus đã xử lý) | ~307 MB | _<điền link Google Drive/OneDrive>_ |
| `index/` đã build (`bm25.pkl`, `dense.pkl`, `MANIFEST.json`) | ~2.9 GB | _<điền link>_ |

> Nén `.zip` hoặc `.tar.gz` trước khi upload. Đặt quyền **Anyone with the link → Viewer**.

## 3. Cấu trúc & định dạng dữ liệu

`legal_articles.clean.jsonl` — mỗi dòng là **một Điều luật** (JSON Lines, UTF-8):

| Trường | Kiểu | Mô tả | Ví dụ |
|---|---|---|---|
| `id` | str | Khoá tăng dần, duy nhất | `"0"` |
| `doc_num` | str | Mã hiệu văn bản | `"03/2022/TT-BXD"` |
| `doc_ref` | str | `mã VB \| tên đầy đủ VB` | `"03/2022/TT-BXD\|Thông tư 03/2022/TT-BXD …"` |
| `article_number` | str | Số hiệu Điều | `"Điều 1"` |
| `article_ref` | str | `mã VB \| tên VB \| Điều N` — **khoá so khớp đáp án BTC** | `"03/2022/TT-BXD\|…\|Điều 1"` |
| `text` | str | Toàn văn Điều (giữ tiêu đề `**Điều N. …**`, các Khoản/Điểm) | `"**Điều 1. Vị trí chức năng**\n\n1. …"` |
| `text_length` | str | Độ dài ký tự của `text` | `"1927"` |

Ví dụ một dòng:

```json
{"id":"0","doc_num":"03/2022/TT-BXD","doc_ref":"03/2022/TT-BXD|Thông tư 03/2022/TT-BXD Thông tư Hướng dẫn …","article_number":"Điều 1","article_ref":"03/2022/TT-BXD|Thông tư 03/2022/TT-BXD …|Điều 1","text":"**Điều 1. Vị trí chức năng**\n\n1. Sở Xây dựng là cơ quan …","text_length":"1927"}
```

**Bộ test** `data/test/R2AIStage1DATA.json` (BTC cung cấp) — danh sách 2000 câu:

```json
[{"id": 1, "question": "…"}, …]
```

## 4. Quy trình xử lý dữ liệu (tái hiện được)

1. **Ingest** 2 dataset HuggingFace (chỉ văn bản cấp quốc gia) → chuẩn hoá về
   `{law_id, doc_type, title, text, source_url}`.
2. **Dedup** theo `law_id`, giữ bản `text` dài nhất; bỏ file rỗng / VB hết hiệu lực rõ ràng.
3. **Chunk theo Điều** (`src/corpus/legal_chunker.py`): tách theo `Điều`, giữ ngữ cảnh `Chương/Mục`,
   định danh chuẩn `mã VB | tên VB | Điều N` để khớp đúng đơn vị BTC chấm điểm.
4. **Vá 2 văn bản trọng yếu** từ bản RTF tải tay:
   - **Bộ luật Lao động 45/2019/QH14**: thay 220 Điều bản tiếng Anh → tiếng Việt (loại bỏ rò rỉ
     "Article N" trong answer).
   - **Luật Đấu thầu 22/2023/QH15**: làm mới 86 Điều + bổ sung 9 Điều thiếu `[15,26,32,33,70,73,83,84,85]`.
     (Điều 71 vắng trong bản crawl — ghi nhận là khuyết thật, không bịa.)
   - Kiểm chứng nội dung bằng **content-anchor** (đối chiếu tiêu đề Điều mỏ neo), không chỉ đếm số Điều.
5. **Xuất** `legal_articles.clean.jsonl` (148,106 dòng, 0 trùng `id`, 0 vi phạm schema, 0 rò rỉ tiếng Anh).

## 5. Kiểm định chất lượng

| Kiểm tra | Kết quả |
|---|---|
| Số Điều | 148,106 |
| Trùng `id` | 0 |
| Vi phạm schema | 0 |
| Rò rỉ "Article N" tiếng Anh trong corpus | 0 (đã vá Bộ luật Lao động) |
| md5 index khớp `MANIFEST.json` | `bm25.pkl=eb495e48…`, `dense.pkl=9ea16a77…` |
