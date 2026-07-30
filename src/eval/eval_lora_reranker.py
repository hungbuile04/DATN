# src/eval/eval_lora_reranker.py

"""
So sánh head-to-head 3 chiến lược reranking ở cấp Retrieval:

    A) dense_only:      Dense Search + Ticker Boost (không rerank)
    B) gemini_rerank:   Dense + Ticker Boost + Gemini Flash Reranker (API)
    C) lora_rerank:     Dense + Ticker Boost + LoRA Cross-Encoder (local)

Metrics: Precision@k, Hit Rate@k, MRR, Latency (ms/query)

Usage:
    python -m src.eval.eval_lora_reranker
    python -m src.eval.eval_lora_reranker --questions src/eval/questions_hard.json
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


def eval_3way_comparison(
    retriever_dense: DualRetriever,
    retriever_gemini: DualRetriever | None,
    retriever_lora: DualRetriever,
    questions: list[dict],
    text_top_k: int = 6,
    image_top_k: int = 4,
) -> dict:
    """
    So sánh 3 chiến lược reranking trên eval set.

    Args:
        retriever_dense:  DualRetriever với use_reranker=False
        retriever_gemini: DualRetriever với Gemini reranker (None nếu không có API key)
        retriever_lora:   DualRetriever với LoRA reranker
        questions:        list eval questions
        text_top_k:       top-k text
        image_top_k:      top-k image

    Returns:
        dict summary metrics
    """
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM: So sánh Reranker — Dense vs Gemini vs LoRA")
    print("=" * 70)

    strategies = {
        "dense_only": {"desc": "Dense + Ticker Boost (không rerank)"},
        "lora_rerank": {"desc": "Dense + Ticker Boost + LoRA Cross-Encoder (local)"},
    }
    if retriever_gemini is not None:
        strategies["gemini_rerank"] = {"desc": "Dense + Ticker Boost + Gemini Flash (API)"}

    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}
    latencies = {s: [] for s in strategies}

    total_k = text_top_k + image_top_k

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q["id"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])

        if not doc:
            continue

        print(f"\n  [{i}/{len(questions)}] {qid}: {question[:60]}...")

        # ── A) Dense Only ──
        t0 = time.perf_counter()
        retriever_dense.use_reranker = False
        res_dense = _safe_retrieve(retriever_dense, question, text_top_k, image_top_k)
        latencies["dense_only"].append((time.perf_counter() - t0) * 1000)
        if res_dense:
            all_dense = res_dense.text_chunks + res_dense.image_chunks
            _score_strategy(metrics["dense_only"], all_dense, doc, requires, total_k)

        # ── B) Gemini Reranker ──
        if retriever_gemini is not None:
            t0 = time.perf_counter()
            retriever_gemini.use_reranker = True
            res_gemini = _safe_retrieve(retriever_gemini, question, text_top_k, image_top_k)
            latencies["gemini_rerank"].append((time.perf_counter() - t0) * 1000)
            if res_gemini:
                all_gemini = res_gemini.text_chunks + res_gemini.image_chunks
                _score_strategy(metrics["gemini_rerank"], all_gemini, doc, requires, total_k)

        # ── C) LoRA Reranker ──
        t0 = time.perf_counter()
        retriever_lora.use_reranker = True
        res_lora = _safe_retrieve(retriever_lora, question, text_top_k, image_top_k)
        latencies["lora_rerank"].append((time.perf_counter() - t0) * 1000)
        if res_lora:
            all_lora = res_lora.text_chunks + res_lora.image_chunks
            _score_strategy(metrics["lora_rerank"], all_lora, doc, requires, total_k)

        # Log comparison
        hit_d = hit_rate_at_k(all_dense if res_dense else [], doc, requires, total_k)
        hit_l = hit_rate_at_k(all_lora if res_lora else [], doc, requires, total_k)
        log = f"    hit@k: dense={hit_d} lora={hit_l}"
        if retriever_gemini and res_gemini:
            hit_g = hit_rate_at_k(all_gemini, doc, requires, total_k)
            log += f" gemini={hit_g}"
        print(log)

        time.sleep(0.2)  # Nhẹ nhàng

    # ── Kết quả tổng thể ──
    summary = _aggregate_and_print(strategies, metrics, "Dense vs Gemini vs LoRA Reranker")

    # ── Latency summary ──
    print(f"\n\n  ⏱  LATENCY SUMMARY (ms/query):")
    print(f"  {'Strategy':<20} {'Mean':>8} {'Median':>8} {'P95':>8}")
    print(f"  {'─' * 48}")
    for strat in strategies:
        lats = sorted(latencies[strat])
        if lats:
            mean_lat = sum(lats) / len(lats)
            median_lat = lats[len(lats) // 2]
            p95_lat = lats[int(len(lats) * 0.95)]
            print(f"  {strat:<20} {mean_lat:>8.1f} {median_lat:>8.1f} {p95_lat:>8.1f}")
            summary[f"{strat}_latency_mean_ms"] = round(mean_lat, 1)

    return summary


def run(questions_path: str, output_path: str,
        text_top_k: int = 6, image_top_k: int = 4,
        skip_gemini: bool = False):
    """Chạy eval so sánh 3 chiến lược reranking."""

    google_key = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    chroma_path = str(CFG["paths"]["vector_db"])

    # ── Retriever A: Dense only ──
    retriever_dense = DualRetriever(
        chroma_path=chroma_path,
        api_key=google_key,
        cfg=CFG,
        openrouter_key=openrouter_key,
        use_reranker=False,
    )

    # ── Retriever B: Gemini reranker (optional) ──
    retriever_gemini = None
    if not skip_gemini and openrouter_key:
        retriever_gemini = DualRetriever(
            chroma_path=chroma_path,
            api_key=google_key,
            cfg=CFG,
            openrouter_key=openrouter_key,
            use_reranker=True,
            reranker_type="gemini",
        )

    # ── Retriever C: LoRA reranker ──
    retriever_lora = DualRetriever(
        chroma_path=chroma_path,
        api_key=google_key,
        cfg=CFG,
        use_reranker=True,
        reranker_type="lora",
    )

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "3way_reranker_comparison",
        "config": {
            "text_top_k": text_top_k,
            "image_top_k": image_top_k,
            "n_questions": len(questions),
        },
    }

    all_results["comparison"] = eval_3way_comparison(
        retriever_dense, retriever_gemini, retriever_lora,
        questions, text_top_k, image_top_k,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n✓ Results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="So sánh Dense vs Gemini vs LoRA Reranker"
    )
    parser.add_argument("--questions", default="src/eval/questions_hard.json")
    parser.add_argument("--output", default="results/eval_lora_vs_gemini.json")
    parser.add_argument("--text_top_k", type=int, default=6)
    parser.add_argument("--image_top_k", type=int, default=4)
    parser.add_argument("--skip_gemini", action="store_true",
                        help="Bỏ qua Gemini reranker (nếu không có API key)")
    args = parser.parse_args()
    run(args.questions, args.output, args.text_top_k, args.image_top_k, args.skip_gemini)
