#!/usr/bin/env bash
# R2AI 2026 — chạy pipeline A→G TRÊN MAC (thay cho notebook Colab).
# Dùng: ./run_local.sh <stage>
#   setup   : cài deps vào .venv hiện tại + bật config 'st' + device mps
#   A       : build st index (resumable) — phần nặng nhất
#   B       : eval recall (trần truy hồi)
#   E       : tune F2 (chốt max_k / rel_threshold / rrf_k)
#   G       : inference + validate + đóng gói submission
#   all     : setup→A→B→E→G  (bỏ qua dev)
#
# CHẠY TỪNG STAGE để kiểm soát; KHÔNG nên chạy 'all' lần đầu.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"   # override: PY=./.venv/bin/python ./run_local.sh A

set_cfg () {  # set_cfg <key_regex> <replacement>
  $PY - "$1" "$2" <<'EOF'
import re, sys, pathlib
pat, rep = sys.argv[1], sys.argv[2]
p = pathlib.Path('config.yaml'); c = p.read_text()
c = re.sub(pat, rep, c); p.write_text(c)
print(f"  config: {rep}")
EOF
}

stage="${1:-}"; [ -z "$stage" ] && { sed -n '2,20p' "$0"; exit 1; }

case "$stage" in

setup)
  echo ">>> [setup] cài deps + bật config st/mps"
  $PY -m pip install -q sentence-transformers rank_bm25 pyvi scikit-learn pyyaml peft accelerate
  set_cfg 'dense_backend:\s*\w+'    'dense_backend: st'
  set_cfg 'dense_model:\s*\S+'      'dense_model: AITeamVN/Vietnamese_Embedding'
  set_cfg 'dense_fp16:\s*\w+'       'dense_fp16: true'
  set_cfg 'dense_batch_size:\s*\d+' 'dense_batch_size: 16'
  set_cfg 'device:\s*\w+'           'device: mps'
  echo ">>> xong. Nếu Mac báo thiếu RAM ở stage A: hạ dense_batch_size về 8."
  ;;

A)
  echo ">>> [A] build st index (resumable) từ legal_articles.jsonl — đứt thì chạy lại stage A, nó skip shard đã có"
  echo ">>> MẸO: nếu đã có 77 shard trên Drive, tải folder data/index/_emb_shards/ về ĐÚNG vị trí này TRƯỚC khi chạy để chỉ phải encode 5 shard cuối."
  $PY scripts/build_index_from_jsonl_resumable.py --shard-size 5000
  ;;

B)
  echo ">>> [B] eval recall (trần truy hồi, rerank OFF)"
  $PY scripts/eval_recall.py --test data/test/dev_test.json --gold data/test/dev_gold.json --ks 5,10,20,50,100
  ;;

E)
  echo ">>> [E] tune F2"
  $PY scripts/tune_f2.py --test-file data/test/dev_test.json --gold-file data/test/dev_gold.json
  ;;

G)
  echo ">>> [G] inference + validate + đóng gói (dùng Ollama qwen3:8b nếu có; nếu chưa: set llm_backend extractive)"
  $PY scripts/run_inference.py --test data/test/test.json --out results.json
  $PY scripts/validate_submission.py --results results.json --test data/test/test.json
  $PY scripts/make_submission.py --results results.json --out submission.zip
  echo ">>> submission.zip đã sẵn sàng."
  ;;

all)
  "$0" setup && "$0" A && "$0" B && "$0" E && "$0" G
  ;;

*) echo "stage không hợp lệ: $stage"; sed -n '2,20p' "$0"; exit 1 ;;
esac
