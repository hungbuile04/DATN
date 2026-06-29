# src/eval/eval_reranker_blind.py

"""
Ablation Study: Reranker khi KHÔNG CÓ ticker metadata.

Giả lập kịch bản: tài liệu không chứa metadata mã cổ phiếu.
→ Tắt ticker detection để đánh giá reranker trên TOÀN BỘ câu hỏi.

So sánh 2 chiến lược:
  A) dense_blind:    Dense Search thuần (không ticker boost, không rerank)
  B) dense_reranker: Dense Search + Reranker (không ticker boost, CÓ rerank)

Mục đích: Chứng minh reranker có giá trị khi thiếu metadata.

Usage:
    python -m src.eval.eval_reranker_blind --questions src/eval/questions.json
    python -m src.eval.eval_reranker_blind --questions src/eval/questions_hard.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG
from src.retrieval.retriever import DualRetriever
from src.eval.eval_retrieval import (
    _safe_retrieve,
    _score_strategy,
    _aggregate_and_print,
    hit_rate_at_k,
    precision_at_k,
    mrr,
)


def eval_reranker_blind(retriever: DualRetriever, questions: list[dict],
                        text_top_k: int = 6, image_top_k: int = 4) -> dict:
    """
    So sánh Dense vs Dense+Reranker khi TẮT ticker detection.
    """
    print("\n" + "="*70)
    print("THÍ NGHIỆM: Reranker Blind (tắt ticker detection)")
    print("  → Giả lập kịch bản tài liệu KHÔNG có metadata mã cổ phiếu")
    print("="*70)

    # ── Monkey-patch: tắt ticker detection ──
    original_detect = retriever._detect_tickers
    retriever._detect_tickers = lambda q: []  # Luôn trả về rỗng

    # Cũng tắt infer_tickers để image pipeline không dùng ticker
    original_infer = retriever._infer_tickers_from_chunks
    retriever._infer_tickers_from_chunks = lambda chunks: []

    strategies = {
        "dense_blind":    {"desc": "Dense Search thuần (không ticker, không rerank)"},
        "dense_reranker": {"desc": "Dense Search + Reranker (không ticker)"},
    }
    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}
    total_k = text_top_k + image_top_k

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q["id"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])

        if not doc:
            continue

        print(f"\n  [{i}/{len(questions)}] {qid}: {question[:55]}...")

        # ── A) Dense Blind (không ticker, không rerank) ──
        retriever.use_reranker = False
        res_dense = _safe_retrieve(retriever, question, text_top_k, image_top_k)
        if res_dense:
            all_dense = res_dense.text_chunks + res_dense.image_chunks
            _score_strategy(metrics["dense_blind"], all_dense, doc, requires, total_k)

        # ── B) Dense + Reranker (không ticker, CÓ rerank) ──
        retriever.use_reranker = True
        res_rerank = _safe_retrieve(retriever, question, text_top_k, image_top_k)
        if res_rerank:
            all_rerank = res_rerank.text_chunks + res_rerank.image_chunks
            _score_strategy(metrics["dense_reranker"], all_rerank, doc, requires, total_k)

        # Log
        hit_d = hit_rate_at_k(all_dense if res_dense else [], doc, requires, total_k)
        hit_r = hit_rate_at_k(all_rerank if res_rerank else [], doc, requires, total_k)
        tag = "⚡" if hit_r > hit_d else ("⚠" if hit_r < hit_d else "➖")
        print(f"    {tag} hit@k: dense={hit_d} reranker={hit_r}")

        time.sleep(0.5)

    # ── Khôi phục ticker detection ──
    retriever._detect_tickers = original_detect
    retriever._infer_tickers_from_chunks = original_infer

    # ── Bonus: so sánh với baseline CÓ ticker ──
    print(f"\n\n  📊 BONUS: So sánh Dense+Ticker (baseline thật) vs Dense+Reranker (blind)")
    retriever.use_reranker = False
    metrics_baseline = {"p@k": [], "hit@k": [], "mrr": []}
    for q in questions:
        doc = q.get("doc", "")
        requires = q.get("requires", [])
        if not doc:
            continue
        res = _safe_retrieve(retriever, q["question"], text_top_k, image_top_k)
        if res:
            all_c = res.text_chunks + res.image_chunks
            _score_strategy(metrics_baseline, all_c, doc, requires, total_k)

    n = len(metrics_baseline["p@k"])
    if n > 0:
        print(f"  Dense+Ticker (baseline):  P@k={sum(metrics_baseline['p@k'])/n:.4f}  "
              f"Hit@k={sum(metrics_baseline['hit@k'])/n:.4f}  "
              f"MRR={sum(metrics_baseline['mrr'])/n:.4f}  N={n}")

    return _aggregate_and_print(strategies, metrics, "Reranker Blind (không ticker)")


def run(questions_path: str, output_path: str, text_top_k: int = 6, image_top_k: int = 4):
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not google_key or not openrouter_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY hoặc OPENROUTER_API_KEY trong .env")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
        openrouter_key=openrouter_key,
        use_reranker=True,
    )

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "reranker_blind_ablation",
        "description": "Tắt ticker detection → so sánh Dense vs Dense+Reranker trên toàn bộ câu hỏi",
        "config": {"text_top_k": text_top_k, "image_top_k": image_top_k},
    }

    all_results["blind_ablation"] = eval_reranker_blind(
        retriever, questions, text_top_k, image_top_k
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n✓ Results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reranker Blind Ablation (tắt ticker)")
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/eval_reranker_blind.json")
    parser.add_argument("--text_top_k", type=int, default=6)
    parser.add_argument("--image_top_k", type=int, default=4)
    args = parser.parse_args()
    run(args.questions, args.output, args.text_top_k, args.image_top_k)
