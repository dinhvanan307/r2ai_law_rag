"""Run the pipeline over a test file and write results.json.

Test file format (list or {"data": [...]}):
    [{"id": 1, "question": "..."}, ...]

Usage:
    python scripts/run_inference.py --test data/test/test.json --out results.json
"""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import load_config
from src.pipeline import Pipeline


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("questions") or []
    out = []
    for row in data:
        out.append({"id": row["id"], "question": row.get("question", "")})
    return out


def _device_report():
    try:
        import torch
        avail = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if avail else "—"
        return avail, name
    except Exception as e:  # torch chua cai / loi import
        return False, f"(torch import failed: {e})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run legal RAG inference.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--test", default=None, help="Override test file path")
    ap.add_argument("--out", default=None, help="Override results.json path")
    ap.add_argument("--limit", type=int, default=0, help="Only first N questions")
    ap.add_argument("--shard", default="1/1",
                    help="'i/n' — chia 2000 cau cho n worker, worker nay lam phan i "
                         "(round-robin). VD 2 T4: worker0=1/2, worker1=2/2. Merge sau.")
    args = ap.parse_args()

    # Parse shard "i/n" (1-based). Round-robin => moi worker tai deu loai cau.
    try:
        _si, _sn = (int(x) for x in args.shard.split("/"))
        assert 1 <= _si <= _sn
    except Exception:
        raise SystemExit(f"--shard phai dang 'i/n' (1<=i<=n), nhan duoc {args.shard!r}")

    def log(*a):
        print(*a, flush=True)   # flush=True -> Kaggle/pipe hien log LIVE, khong bi buffer

    cfg = load_config(args.config)
    test_path = Path(args.test) if args.test else cfg.resolve(cfg.paths.test_file)
    out_path = Path(args.out) if args.out else cfg.resolve(cfg.paths.results_file)

    questions = load_questions(test_path)
    if args.limit:
        questions = questions[: args.limit]
    n_total = len(questions)
    if _sn > 1:
        questions = questions[_si - 1::_sn]   # round-robin slice cho worker nay

    want_device = (cfg.models.device or "").lower()
    cuda_ok, gpu_name = _device_report()
    # --- BANNER: moi thong tin can de chan misconfig hien ngay 8 dong dau ---
    log("=" * 64)
    log(f"[run_inference] config      = {args.config}")
    log(f"[run_inference] test file   = {test_path}  (N_total={n_total}, "
        f"shard {args.shard} -> {len(questions)} cau)")
    log(f"[run_inference] llm_backend = {cfg.models.llm_backend}")
    log(f"[run_inference] use_hyde    = {getattr(cfg.retrieval, 'use_hyde', False)}")
    log(f"[run_inference] device(cfg) = {cfg.models.device!r} | dense_fp16={cfg.models.dense_fp16}")
    log(f"[run_inference] cuda avail  = {cuda_ok} | GPU = {gpu_name}")
    log(f"[run_inference] reranker    = {cfg.models.reranker_model} (use={cfg.retrieval.use_reranker})")
    log(f"[run_inference] rerank_top_n= {cfg.retrieval.rerank_top_n} | max_k={cfg.retrieval.max_k}")
    log("=" * 64)

    # FAIL-FAST: yeu cau cuda nhung khong co GPU -> dung NGAY, dung de chay 9h tren CPU
    if want_device == "cuda" and not cuda_ok:
        raise SystemExit(
            "❌ device=cuda nhung torch.cuda.is_available()=False.\n"
            "   => Accelerator chua bat (hoac bi reset sau khi import notebook).\n"
            "   FIX: Settings -> Accelerator = GPU T4 x2 -> Restart & Run All.\n"
            "   (Khong fail-fast thi reranker bo ve CPU -> chay ~3-10h.)")

    log("[run_inference] dang nap pipeline (index .pkl + tai model)... "
        "neu TREO o day = tai model/HF cham hoac doc .pkl, KHONG phai infer.")
    t_load = time.time()
    pipe = Pipeline(cfg)
    log(f"[run_inference] pipeline READY sau {time.time()-t_load:.1f}s -> bat dau infer")

    results = []
    t0 = time.time()
    step = 20 if len(questions) > 100 else 1
    for i, q in enumerate(questions, 1):
        item = pipe.answer_one(q["id"], q["question"])
        results.append(item.to_submission())
        if i == 1 or i % step == 0 or i == len(questions):
            el = time.time() - t0
            rate = el / i
            eta = rate * (len(questions) - i)
            log(f"[shard {args.shard}]   {i}/{len(questions)}  | {el:.0f}s da chay "
                f"| {rate:.2f}s/cau | ETA ~{eta/60:.1f} phut")

    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[run_inference] DONE: ghi {len(results)} items -> {out_path} "
        f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
