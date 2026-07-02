# src/eval/eval_retrieval.py

"""
Đánh giá kiến trúc Retrieval bằng retrieval-level metrics.

3 thí nghiệm:
  1. Single vs Dual Collection
     — Gộp chung (text+table+image) vs Tách (text|table riêng, image riêng)
     → So sánh Precision@k, Hit Rate@k, MRR

  2. BM25 vs Dense vs Hybrid (RRF)
     — Đóng góp từng thành phần search

  3. Ticker-aware vs Không ticker
     — Ticker boost/filter có giúp lấy đúng chunk hơn?

Dùng questions.json với các trường:
    - requires: ["text"], ["table"], ["image"], ["text","image"], ...
    - doc: tên tài liệu mong muốn (dùng để check relevant)
    - source_page: trang mong muốn (dùng để check relevant)

Usage:
    python src/eval/eval_retrieval.py
    python src/eval/eval_retrieval.py --questions src/eval/questions.json
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
from src.retrieval.retriever import (
    DualRetriever, RetrievedChunk,
    _ticker_boost,
)


# ══════════════════════════════════════════════
# Retrieval Metrics
# ══════════════════════════════════════════════

def precision_at_k(retrieved: list[RetrievedChunk], relevant_doc: str,
                   relevant_types: list[str], k: int) -> float:
    """Tỷ lệ chunk relevant trong top-k."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if _is_relevant(c, relevant_doc, relevant_types))
    return hits / len(top_k)


def recall_at_k(retrieved: list[RetrievedChunk], relevant_doc: str,
                relevant_types: list[str], k: int,
                total_relevant: int = 1) -> float:
    """Tỷ lệ relevant chunks đã lấy được trong top-k."""
    top_k = retrieved[:k]
    hits = sum(1 for c in top_k if _is_relevant(c, relevant_doc, relevant_types))
    return hits / total_relevant if total_relevant > 0 else 0.0


def hit_rate_at_k(retrieved: list[RetrievedChunk], relevant_doc: str,
                  relevant_types: list[str], k: int) -> int:
    """Có ít nhất 1 chunk relevant trong top-k? 0 hoặc 1."""
    top_k = retrieved[:k]
    return 1 if any(_is_relevant(c, relevant_doc, relevant_types) for c in top_k) else 0


def mrr(retrieved: list[RetrievedChunk], relevant_doc: str,
        relevant_types: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank của chunk relevant đầu tiên."""
    for rank, c in enumerate(retrieved, 1):
        if _is_relevant(c, relevant_doc, relevant_types):
            return 1.0 / rank
    return 0.0


def _is_relevant(chunk: RetrievedChunk, relevant_doc,
                 relevant_types: list[str]) -> bool:
    """Chunk có relevant không? Dựa trên doc name + chunk type.
    relevant_doc có thể là str hoặc list[str] (cho câu hỏi so sánh nhiều mã).
    """
    if relevant_doc:
        if isinstance(relevant_doc, list):
            # Chunk relevant nếu khớp BẤT KỲ doc nào trong list
            if not any(d in chunk.doc for d in relevant_doc):
                return False
        else:
            if relevant_doc not in chunk.doc:
                return False
    if relevant_types and chunk.chunk_type not in relevant_types:
        return False
    return True


# ══════════════════════════════════════════════
# Thí nghiệm 1: Single vs Dual Collection
# ══════════════════════════════════════════════

def eval_single_vs_dual(retriever: DualRetriever, questions: list[dict],
                        text_top_k: int = 5, image_top_k: int = 3) -> dict:
    """
    So sánh 3 chiến lược:
      A) dual      — text pipeline (top_k=5) + image pipeline (top_k=3) riêng
      B) text_only — chỉ text pipeline (top_k=8), bỏ image
      C) merged    — gộp text+image thành 1 list, lấy top-8
    """
    print("\n" + "="*70)
    print("THÍ NGHIỆM 1: Single vs Dual Collection")
    print("="*70)

    total_k = text_top_k + image_top_k
    strategies = {
        "dual":      {"desc": "Text(5) + Image(3) riêng biệt"},
        "text_only": {"desc": "Chỉ Text pipeline (top-8)"},
        "merged":    {"desc": "Gộp Text+Image → top-8"},
    }
    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q["id"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])

        print(f"\n  [{i}/{len(questions)}] {qid}: {question[:50]}...")

        # Retrieve full (cả text + image)
        result = _safe_retrieve(retriever, question, text_top_k, image_top_k)
        if result is None:
            continue

        # Strategy A: Dual (text riêng + image riêng)
        dual_all = result.text_chunks + result.image_chunks
        _score_strategy(metrics["dual"], dual_all, doc, requires, total_k)

        # Strategy B: Text only (gấp đôi top-k, bỏ image)
        result_text = _safe_retrieve(retriever, question, total_k, 0)
        if result_text:
            _score_strategy(metrics["text_only"], result_text.text_chunks, doc, requires, total_k)

        # Strategy C: Merged (gộp text + image, sort lại theo score)
        merged_list = sorted(dual_all, key=lambda c: c.score, reverse=True)[:total_k]
        _score_strategy(metrics["merged"], merged_list, doc, requires, total_k)

        # Log
        dual_hit = hit_rate_at_k(dual_all, doc, requires, total_k)
        text_hit = hit_rate_at_k(result_text.text_chunks if result_text else [], doc, requires, total_k)
        merged_hit = hit_rate_at_k(merged_list, doc, requires, total_k)
        print(f"    hit@k: dual={dual_hit} text_only={text_hit} merged={merged_hit}")

    return _aggregate_and_print(strategies, metrics, "Single vs Dual")


# ══════════════════════════════════════════════
# Thí nghiệm 2: BM25 vs Dense vs Hybrid
# ══════════════════════════════════════════════

def eval_search_components(retriever: DualRetriever, questions: list[dict],
                           top_k: int = 5) -> dict:
    """
    Đánh giá Dense search (phương pháp hiện tại) trên text pipeline.
    So sánh có/không ticker boost.
    """
    print("\n" + "="*70)
    print("THÍ NGHIỆM 2: Dense Search Quality")
    print("="*70)

    strategies = {
        "dense_with_boost":  {"desc": "Dense + ticker boost (hiện tại)"},
        "dense_no_boost":    {"desc": "Dense không boost"},
    }
    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}

    candidate_k = top_k * 4

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q["id"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])
        text_requires = [r for r in requires if r in ("text", "table")]
        if not text_requires:
            text_requires = ["text", "table"]

        print(f"\n  [{i}/{len(questions)}] {qid}: {question[:50]}...")

        try:
            query_vec = retriever._embed_query(question)
        except Exception as e:
            print(f"    ⚠ Embed error: {e}")
            time.sleep(10)
            continue

        tickers = retriever._detect_tickers(question)

        # A) Dense + ticker boost (current pipeline)
        dense_results = retriever._dense_search(
            query_vec, retriever.text_col, top_n=candidate_k
        )
        if tickers:
            boosted = _ticker_boost(dense_results, tickers, boost=1.5)
        else:
            boosted = dense_results
        boost_chunks = _ids_to_text_chunks(retriever, boosted[:top_k])
        _score_strategy(metrics["dense_with_boost"], boost_chunks, doc, text_requires, top_k)

        # B) Dense without boost
        no_boost_chunks = _ids_to_text_chunks(retriever, dense_results[:top_k])
        _score_strategy(metrics["dense_no_boost"], no_boost_chunks, doc, text_requires, top_k)

        boost_hit = hit_rate_at_k(boost_chunks, doc, text_requires, top_k)
        no_boost_hit = hit_rate_at_k(no_boost_chunks, doc, text_requires, top_k)
        print(f"    hit@k: with_boost={boost_hit} no_boost={no_boost_hit}")

        time.sleep(0.5)

    return _aggregate_and_print(strategies, metrics, "Dense Search Quality")


# ══════════════════════════════════════════════
# Thí nghiệm 3: Ticker-aware vs Không ticker
# ══════════════════════════════════════════════

def eval_ticker_aware(retriever: DualRetriever, questions: list[dict],
                      text_top_k: int = 5, image_top_k: int = 3) -> dict:
    """
    So sánh có/không ticker detection:
      A) with_ticker    — ticker boost (text) + hard filter (image)
      B) without_ticker — không detect ticker, search bình thường
    """
    print("\n" + "="*70)
    print("THÍ NGHIỆM 3: Ticker-aware vs Không Ticker")
    print("="*70)

    # Chỉ dùng câu hỏi có ticker rõ ràng
    ticker_qs = [q for q in questions if q.get("doc", "")]
    print(f"  Dùng {len(ticker_qs)} câu hỏi có doc xác định")

    strategies = {
        "with_ticker":    {"desc": "Có ticker boost + filter"},
        "without_ticker": {"desc": "Không ticker"},
    }
    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}

    total_k = text_top_k + image_top_k

    for i, q in enumerate(ticker_qs, 1):
        question = q["question"]
        qid = q["id"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])

        print(f"\n  [{i}/{len(ticker_qs)}] {qid}: {question[:50]}...")

        # A) With ticker (normal flow)
        result_with = _safe_retrieve(retriever, question, text_top_k, image_top_k)
        if result_with is None:
            continue
        all_with = result_with.text_chunks + result_with.image_chunks
        _score_strategy(metrics["with_ticker"], all_with, doc, requires, total_k)

        # B) Without ticker — Dense search without boost/filter
        try:
            query_vec = retriever._embed_query(question)
        except Exception:
            time.sleep(10)
            continue

        # Text pipeline without ticker boost
        candidate_k = text_top_k * 4
        dense_text = retriever._dense_search(
            query_vec, retriever.text_col, top_n=candidate_k
        )
        # Không boost ticker
        text_chunks_no_ticker = _ids_to_text_chunks(retriever, dense_text[:text_top_k])

        # Image pipeline without ticker filter
        dense_img = retriever._dense_search(
            query_vec, retriever.image_col, top_n=image_top_k * 4
        )
        image_chunks_no_ticker = _ids_to_image_chunks(retriever, dense_img[:image_top_k])

        all_without = text_chunks_no_ticker + image_chunks_no_ticker
        _score_strategy(metrics["without_ticker"], all_without, doc, requires, total_k)

        hit_with = hit_rate_at_k(all_with, doc, requires, total_k)
        hit_without = hit_rate_at_k(all_without, doc, requires, total_k)
        print(f"    hit@k: with_ticker={hit_with} without_ticker={hit_without}")

        time.sleep(0.5)

    return _aggregate_and_print(strategies, metrics, "Ticker-aware vs Không")


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _safe_retrieve(retriever, question, text_top_k, image_top_k):
    """Retrieve with retry on timeout."""
    for attempt in range(3):
        try:
            return retriever.retrieve(question, text_top_k=text_top_k, image_top_k=image_top_k)
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"    ⚠ Retrieve error (attempt {attempt+1}): {e}")
            print(f"    Waiting {wait}s...")
            time.sleep(wait)
    print(f"    ✗ Skipping question after 3 retries")
    return None


def _ids_to_text_chunks(retriever: DualRetriever,
                        scored: list[tuple[str, float]]) -> list[RetrievedChunk]:
    """Convert (chunk_id, score) list → RetrievedChunk list từ text collection."""
    chunks = []
    for chunk_id, score in scored:
        c = retriever._build_text_chunk(chunk_id, score)
        if c:
            chunks.append(c)
    return chunks


def _ids_to_image_chunks(retriever: DualRetriever,
                         scored: list[tuple[str, float]]) -> list[RetrievedChunk]:
    """Convert (chunk_id, score) list → RetrievedChunk list từ image collection."""
    chunks = []
    for chunk_id, score in scored:
        c = retriever._build_image_chunk(chunk_id, score)
        if c:
            chunks.append(c)
    return chunks


def _score_strategy(metrics: dict, chunks: list[RetrievedChunk],
                    doc: str, requires: list[str], k: int):
    """Tính và append metrics cho 1 strategy, 1 câu hỏi."""
    metrics["p@k"].append(precision_at_k(chunks, doc, requires, k))
    metrics["hit@k"].append(hit_rate_at_k(chunks, doc, requires, k))
    metrics["mrr"].append(mrr(chunks, doc, requires))


def _aggregate_and_print(strategies: dict, metrics: dict, title: str) -> dict:
    """Tính trung bình và in bảng kết quả."""
    print(f"\n\n{'─'*70}")
    print(f"KẾT QUẢ: {title}")
    print(f"{'─'*70}")
    print(f"  {'Strategy':<20} {'P@k':>8} {'Hit@k':>8} {'MRR':>8}  {'N':>4}  Description")
    print(f"  {'─'*65}")

    summary = {}
    for strategy in strategies:
        m = metrics[strategy]
        n = len(m["p@k"])
        if n == 0:
            continue
        avg_p = sum(m["p@k"]) / n
        avg_h = sum(m["hit@k"]) / n
        avg_m = sum(m["mrr"]) / n

        desc = strategies[strategy]["desc"]
        print(f"  {strategy:<20} {avg_p:>8.4f} {avg_h:>8.4f} {avg_m:>8.4f}  {n:>4}  {desc}")

        summary[strategy] = {
            "precision_at_k": round(avg_p, 4),
            "hit_rate_at_k": round(avg_h, 4),
            "mrr": round(avg_m, 4),
            "n": n,
        }

    # Per-category breakdown
    return summary


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def run(questions_path: str, output_path: str,
        text_top_k: int = 5, image_top_k: int = 3):

    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    print("Initializing DualRetriever...")
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {"text_top_k": text_top_k, "image_top_k": image_top_k},
    }

    # Thí nghiệm 1
    all_results["single_vs_dual"] = eval_single_vs_dual(
        retriever, questions, text_top_k, image_top_k
    )

    # Thí nghiệm 2
    all_results["search_components"] = eval_search_components(
        retriever, questions, top_k=text_top_k
    )

    # Thí nghiệm 3
    all_results["ticker_aware"] = eval_ticker_aware(
        retriever, questions, text_top_k, image_top_k
    )

    # ── Save ──
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n✓ Results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá kiến trúc Retrieval (retrieval-level metrics)"
    )
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/eval_retrieval.json")
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    args = parser.parse_args()
    run(args.questions, args.output, args.text_top_k, args.image_top_k)
