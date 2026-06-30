#!/usr/bin/env bash
# R2AI 2026 — LẦN NỘP ĐẦU TIÊN, chạy pipeline HOÀN CHỈNH của dự án TRÊN MAC.
# Tiền đề: index đã build sẵn (data/index/dense.pkl + bm25.pkl) — đúng như hiện trạng.
# Pipeline: BM25 + Dense(AITeamVN/Vietnamese_Embedding) -> RRF -> Vietnamese_Reranker
#           -> chọn top-K theo F2 -> answer extractive (tự trích "Điều X") -> zip.
#
# Lần đầu chạy sẽ tự tải embedding + reranker từ HuggingFace (~vài trăm MB, 1 lần).
# KHÔNG cần Ollama (dùng llm_backend=extractive). Muốn QA chất hơn: xem cuối file.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-./.venv/bin/python}"   # đổi nếu venv ở chỗ khác: PY=python3 ./first_submit_mac.sh
CFG=config.first_submit.yaml

echo ">>> [0/3] (tùy chọn) đo trần Recall trên dev trước khi tiêu hạn mức nộp"
echo ">>>     $PY scripts/eval_recall.py --test data/test/dev_test.json --gold data/test/dev_gold.json --ks 5,10,20,50,100 --with-rerank"
echo ">>>     (bỏ qua bước này nếu muốn nộp luôn)"
echo

echo ">>> [1/3] INFERENCE 2000 câu -> results.json  (lâu nhất: encode query + rerank)"
$PY scripts/run_inference.py --config "$CFG" --test data/test/R2AIStage1DATA.json --out results.json

echo ">>> [2/3] VALIDATE schema (bắt buộc PASS trước khi nộp)"
$PY scripts/validate_submission.py --results results.json --test data/test/R2AIStage1DATA.json

echo ">>> [3/3] ĐÓNG GÓI zip phẳng -> submission.zip"
$PY scripts/make_submission.py --results results.json --out submission.zip

echo
echo ">>> XONG. Nộp file submission.zip lên leaderboard.aiguru.com.vn"
echo ">>> Nâng cấp QA (sau khi IR ổn): bật Ollama rồi đổi trong $CFG:"
echo ">>>   llm_backend: ollama   (đã có llm_model: qwen3:8b — <14B, hợp lệ)"
echo ">>>   chạy:  ollama serve  &&  ollama pull qwen3:8b   rồi chạy lại script này."
