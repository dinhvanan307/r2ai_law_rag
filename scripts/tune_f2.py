import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import itertools

# Thêm đường dẫn src vào sys.path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.pipeline import load_retriever
from scripts.eval_local import norm_article, f2

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_tuning(dev_test_path, dev_gold_path):
    print("1. Loading Dev Set...")
    dev_test = load_json(dev_test_path)
    dev_gold = load_json(dev_gold_path)
    
    # Map qid -> gold keys
    gold_map = {}
    for row in dev_gold:
        qid = row["id"]
        articles = row.get("relevant_articles", [])
        gold_keys = {k for a in articles if (k := norm_article(a))}
        gold_map[qid] = gold_keys
    
    print("2. Loading Base Config & Retriever (Index)...")
    cfg = load_config()
    # Force use_reranker on if we are tuning its parameters
    cfg.retrieval.use_reranker = True
    
    retriever = load_retriever(cfg)
    
    # Định nghĩa Grid Search
    thresholds = [0.3, 0.5, 0.7, 0.8]
    max_ks = [3, 5, 8, 12]
    rrf_ks = [30, 60, 100]
    
    combinations = list(itertools.product(thresholds, max_ks, rrf_ks))
    print(f"3. Starting Grid Search over {len(combinations)} configurations...")
    
    results = []
    
    # We can pre-retrieve top 100 BM25 and Dense, but our retriever does it internally.
    # To keep things simple, we will just change retriever.config attributes and re-run.
    # A faster way is to cache the raw BM25 and Dense scores per query, but for a small dev set, 
    # running full retrieve() is fine.
    
    # Since BM25 and Dense search can take time, let's cache them per query if we can.
    # Actually, retriever.retrieve() is quite fast. Let's just run it.
    
    # For a real dev set (100+ queries), we would cache bm25 and dense results, 
    # and only loop the fusion + reranking. Here we just loop everything for simplicity.
    
    # To save time, we will retrieve ONCE per query with max top_n, and then slice it.
    # Wait, the RRF and Reranker depend on those config values.
    
    best_f2 = -1.0
    best_config = None
    
    for t_idx, (th, mk, rk) in enumerate(tqdm(combinations, desc="Tuning")):
        # Update parameters
        retriever.rel_threshold = th
        retriever.max_k = mk
        retriever.rrf_k = rk
        
        Ps, Rs, F2s = [], [], []
        
        for row in dev_test:
            qid = row["id"]
            question = row["question"]
            
            # Bỏ qua nếu query này không có gold
            if qid not in gold_map:
                continue
                
            gold_keys = gold_map[qid]
            
            # Retrieve
            try:
                retrieved_articles = retriever.retrieve(question)
                pred_keys = {k for a in retrieved_articles if (k := norm_article(f"{a.law_id}|doc|{a.article_no}"))}
                
                if not pred_keys:
                    p = r = 0.0
                else:
                    tp = len(pred_keys & gold_keys)
                    p = tp / len(pred_keys)
                    r = tp / len(gold_keys) if gold_keys else 0.0
                
            except Exception as e:
                # If index is not built, it might fail
                print(f"Error retrieving for Q{qid}: {e}")
                p = r = 0.0
                
            Ps.append(p)
            Rs.append(r)
            F2s.append(f2(p, r))
            
        n = len(F2s) or 1
        macro_p = sum(Ps) / n
        macro_r = sum(Rs) / n
        macro_f2 = sum(F2s) / n
        
        results.append({
            "rel_threshold": th,
            "max_k": mk,
            "rrf_k": rk,
            "macro_p": macro_p,
            "macro_r": macro_r,
            "macro_f2": macro_f2
        })
        
        if macro_f2 > best_f2:
            best_f2 = macro_f2
            best_config = results[-1]
            
    print("\n--- TUNING RESULTS ---")
    results.sort(key=lambda x: x["macro_f2"], reverse=True)
    
    print("\nTop 5 Configurations:")
    print(f"{'Threshold':<10} | {'Max_K':<6} | {'RRF_K':<6} | {'Precision':<10} | {'Recall':<10} | {'F2 Score':<10}")
    print("-" * 70)
    for res in results[:5]:
        print(f"{res['rel_threshold']:<10} | {res['max_k']:<6} | {res['rrf_k']:<6} | {res['macro_p']:<10.4f} | {res['macro_r']:<10.4f} | {res['macro_f2']:<10.4f}")
        
    print(f"\n✅ BEST CONFIGURATION (F2: {best_config['macro_f2']:.4f}):")
    print(f"   rel_threshold : {best_config['rel_threshold']}")
    print(f"   max_k         : {best_config['max_k']}")
    print(f"   rrf_k         : {best_config['rrf_k']}")
    
    print("\nPlease update config.yaml with these values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Retrieval Hyperparameters")
    parser.add_argument("--test-file", default="data/test/dev_test.json")
    parser.add_argument("--gold-file", default="data/test/dev_gold.json")
    args = parser.parse_args()
    
    run_tuning(args.test_file, args.gold_file)
