# src/eval/eval_reranker.py

"""
Đánh giá tính năng Adaptive Reranker.

Ablation study — 2 chiến lược:
  A) dense_only:      Dense Search + Ticker Boost (không rerank)
  B) adaptive_rerank: Dense Search + Ticker Boost + Adaptive Reranker
                      (Reranker CHỈ kích hoạt khi KHÔNG detect được ticker)

Logic:
  - Có ticker → ticker boost đủ chính xác → bỏ qua reranker
  - Không ticker → dense search dễ nhiễu → reranker giúp lọc

So sánh: Precision@k, Hit Rate@k, MRR  (tổng thể + chia theo nhóm có/không ticker)

Usage:
    python -m src.eval.eval_reranker
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
)


def eval_reranker_ablation(retriever: DualRetriever, questions: list[dict],
                           text_top_k: int = 6, image_top_k: int = 4) -> dict:
    """
    Ablation Reranker:
      A) dense_only       — use_reranker = False
      B) adaptive_rerank  — use_reranker = True (chỉ rerank khi KHÔNG có ticker)
    """
    print("\n" + "="*70)
    print("THÍ NGHIỆM: Ablation Adaptive Reranker")
    print("="*70)

    strategies = {
        "dense_only":      {"desc": "Dense + Ticker Boost (không rerank)"},
        "adaptive_rerank": {"desc": "Dense + Ticker Boost + Adaptive Reranker"},
    }
    # Metrics tổng thể
    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}
    # Metrics chia nhóm: có ticker vs không ticker
    metrics_with_ticker = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}
    metrics_no_ticker = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}

    total_k = text_top_k + image_top_k
    n_with_ticker = 0
    n_no_ticker = 0

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q["id"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])

        if not doc:
            continue

        print(f"\n  [{i}/{len(questions)}] {qid}: {question[:50]}...")

        # Detect ticker trước để biết nhóm
        detected_tickers = retriever._detect_tickers(question)
        has_ticker = len(detected_tickers) > 0
        group_metrics = metrics_with_ticker if has_ticker else metrics_no_ticker
        if has_ticker:
            n_with_ticker += 1
        else:
            n_no_ticker += 1

        # ── A) Dense Only ──
        retriever.use_reranker = False
        res_dense = _safe_retrieve(retriever, question, text_top_k, image_top_k)
        if res_dense:
            all_dense = res_dense.text_chunks + res_dense.image_chunks
            _score_strategy(metrics["dense_only"], all_dense, doc, requires, total_k)
            _score_strategy(group_metrics["dense_only"], all_dense, doc, requires, total_k)

        # ── B) Adaptive Reranker ──
        retriever.use_reranker = True
        res_reranker = _safe_retrieve(retriever, question, text_top_k, image_top_k)
        if res_reranker:
            all_rerank = res_reranker.text_chunks + res_reranker.image_chunks
            _score_strategy(metrics["adaptive_rerank"], all_rerank, doc, requires, total_k)
            _score_strategy(group_metrics["adaptive_rerank"], all_rerank, doc, requires, total_k)

        # Log
        hit_dense = hit_rate_at_k(all_dense if res_dense else [], doc, requires, total_k)
        hit_rerank = hit_rate_at_k(all_rerank if res_reranker else [], doc, requires, total_k)
        ticker_tag = f"[{','.join(detected_tickers)}]" if has_ticker else "[no-ticker]"
        print(f"    {ticker_tag} hit@k: dense={hit_dense} adaptive={hit_rerank}")

        time.sleep(0.5)

    # ── Kết quả tổng thể ──
    summary = _aggregate_and_print(strategies, metrics, "Adaptive Reranker (Tổng thể)")

    # ── Kết quả nhóm CÓ ticker ──
    print(f"\n\n  📊 NHÓM CÓ TICKER ({n_with_ticker} câu):")
    _aggregate_and_print(strategies, metrics_with_ticker, "Có Ticker")

    # ── Kết quả nhóm KHÔNG CÓ ticker ──
    print(f"\n\n  📊 NHÓM KHÔNG CÓ TICKER ({n_no_ticker} câu):")
    _aggregate_and_print(strategies, metrics_no_ticker, "Không Ticker")

    summary["n_with_ticker"] = n_with_ticker
    summary["n_no_ticker"] = n_no_ticker

    return summary


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
        "experiment": "adaptive_reranker_ablation",
        "config": {"text_top_k": text_top_k, "image_top_k": image_top_k},
    }

    all_results["reranker_ablation"] = eval_reranker_ablation(
        retriever, questions, text_top_k, image_top_k
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n✓ Results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá Adaptive Reranker")
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/eval_reranker.json")
    parser.add_argument("--text_top_k", type=int, default=6)
    parser.add_argument("--image_top_k", type=int, default=4)
    args = parser.parse_args()
    run(args.questions, args.output, args.text_top_k, args.image_top_k)
