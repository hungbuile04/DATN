# src/eval/eval_retrieval_v2.py

"""
Đánh giá Retrieval v2 — bổ sung so sánh Dense vs BM25 vs Hybrid.

4 thí nghiệm:
  1. Single vs Dual Collection
  2. Dense vs BM25 vs Hybrid (RRF)       ← MỚI
  3. Dense: có/không ticker boost
  4. Ticker-aware vs Không ticker

BM25 triển khai bằng rank_bm25, tokenizer = str.split() (không word-segment
tiếng Việt) — đây là điều kiện thực tế khi không dùng công cụ tách từ chuyên biệt.

Usage:
    python src/eval/eval_retrieval_v2.py
    python src/eval/eval_retrieval_v2.py --questions src/eval/questions.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG
from src.retrieval.retriever import (
    DualRetriever, RetrievedChunk,
    _ticker_boost,
)


# ══════════════════════════════════════════════
# BM25 Engine (lightweight, dùng rank_bm25)
# ══════════════════════════════════════════════

# Từ dừng tiếng Việt phổ biến — loại bỏ để BM25 tập trung vào từ mang nghĩa
_VIETNAMESE_STOPWORDS = {
    # Giới từ, liên từ, trợ từ
    "và", "của", "là", "trong", "với", "các", "cho", "được", "có", "này",
    "từ", "đến", "theo", "về", "trên", "tại", "vào", "ra", "đó", "đã",
    "sẽ", "do", "khi", "để", "cũng", "như", "không", "hay", "hoặc",
    "một", "những", "nhiều", "hơn", "nhất", "rất", "đang", "bởi", "nếu",
    "thì", "mà", "tuy", "nhưng", "vẫn", "còn", "nên", "lại", "qua",
    "sau", "trước", "giữa", "bao", "gì", "nào", "đây", "ấy", "kia",
    # Từ hay gặp trong báo cáo tài chính nhưng không phân biệt
    "dự", "báo", "tổng", "so", "mức",
    # HTML/Markdown artifacts từ chunk
    "document", "page", "topic",
}


class BM25Engine:
    """
    BM25 search trên text collection.

    Tokenizer: regex word split + lọc từ dừng tiếng Việt + lọc số đơn lẻ.
    Không dùng công cụ tách từ (word segmenter) chuyên biệt.
    """

    def __init__(self, retriever: DualRetriever):
        from rank_bm25 import BM25Okapi

        # Lấy toàn bộ documents từ text collection
        data = retriever.text_col.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.docs = data["documents"]

        # Tokenize + lọc stopwords
        self.tokenized = [self._tokenize(doc) for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized)
        print(f"  ✓ BM25 index built: {len(self.ids)} docs")

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenizer cho tiếng Việt (không dùng word segmenter):
        1. Lowercase
        2. Regex split giữ chữ + số + dấu tiếng Việt
        3. Loại từ dừng
        4. Loại số đơn lẻ (1 chữ số) — gây nhiễu nặng trong bảng tài chính
        5. Loại token quá ngắn (1 ký tự)
        """
        text = text.lower()
        tokens = re.findall(r'[\w]+', text, re.UNICODE)
        return [
            t for t in tokens
            if t not in _VIETNAMESE_STOPWORDS
            and len(t) > 1
            and not (t.isdigit() and len(t) <= 2)  # bỏ số 1-2 chữ số
        ]

    def search(self, query: str, top_n: int = 20) -> list[tuple[str, float]]:
        """Trả về [(chunk_id, score)] sorted desc."""
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        # Sort by score desc
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed[:top_n]:
            results.append((self.ids[idx], float(score)))
        return results


# ══════════════════════════════════════════════
# RRF Fusion
# ══════════════════════════════════════════════

def rrf_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion — gộp nhiều ranked list.

    RRF(d) = Σ 1/(k + rank_i(d))
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked, 1):
            rrf_scores[chunk_id] += 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ══════════════════════════════════════════════
# Retrieval Metrics
# ══════════════════════════════════════════════

def precision_at_k(retrieved: list[RetrievedChunk], relevant_doc: str,
                   relevant_types: list[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if _is_relevant(c, relevant_doc, relevant_types))
    return hits / len(top_k)


def hit_rate_at_k(retrieved: list[RetrievedChunk], relevant_doc: str,
                  relevant_types: list[str], k: int) -> int:
    top_k = retrieved[:k]
    return 1 if any(_is_relevant(c, relevant_doc, relevant_types) for c in top_k) else 0


def mrr(retrieved: list[RetrievedChunk], relevant_doc: str,
        relevant_types: list[str]) -> float:
    for rank, c in enumerate(retrieved, 1):
        if _is_relevant(c, relevant_doc, relevant_types):
            return 1.0 / rank
    return 0.0


def _is_relevant(chunk: RetrievedChunk, relevant_doc, relevant_types: list[str]) -> bool:
    if relevant_doc:
        if isinstance(relevant_doc, list):
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

def eval_single_vs_dual(retriever, questions, text_top_k=5, image_top_k=3):
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM 1: Single vs Dual Collection")
    print("=" * 70)

    total_k = text_top_k + image_top_k
    strategies = {
        "dual":      {"desc": "Text(5) + Image(3) riêng biệt"},
        "text_only": {"desc": "Chỉ Text pipeline (top-8)"},
        "merged":    {"desc": "Gộp Text+Image → top-8"},
    }
    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}

    for i, q in enumerate(questions, 1):
        question = q["question"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])
        print(f"\r  [{i}/{len(questions)}] {q['id']}...", end="", flush=True)

        result = _safe_retrieve(retriever, question, text_top_k, image_top_k)
        if result is None:
            continue

        # A) Dual
        dual_all = result.text_chunks + result.image_chunks
        _score(metrics["dual"], dual_all, doc, requires, total_k)

        # B) Text only
        result_text = _safe_retrieve(retriever, question, total_k, 0)
        if result_text:
            _score(metrics["text_only"], result_text.text_chunks, doc, requires, total_k)

        # C) Merged (sort by score)
        merged_list = sorted(dual_all, key=lambda c: c.score, reverse=True)[:total_k]
        _score(metrics["merged"], merged_list, doc, requires, total_k)

    print()
    return _summarize(strategies, metrics, "Single vs Dual")


# ══════════════════════════════════════════════
# Thí nghiệm 2: Dense vs BM25 vs Hybrid (MỚI)
# ══════════════════════════════════════════════

def eval_dense_vs_bm25(retriever, bm25_engine, questions, top_k=5):
    """
    So sánh 3 chiến lược trên TEXT pipeline:
      A) Dense only   — Gemini Embedding 2 cosine similarity
      B) BM25 only    — BM25Okapi, whitespace tokenizer
      C) Hybrid (RRF) — Dense + BM25, Reciprocal Rank Fusion (k=60)
    """
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM 2: Dense vs BM25 vs Hybrid (RRF)")
    print("=" * 70)

    strategies = {
        "dense_only": {"desc": "Dense search (Gemini Embedding 2)"},
        "bm25_only":  {"desc": "BM25 (whitespace tokenizer)"},
        "hybrid_rrf": {"desc": "Hybrid: Dense + BM25 (RRF, k=60)"},
    }
    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}

    candidate_k = top_k * 4

    for i, q in enumerate(questions, 1):
        question = q["question"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])
        text_requires = [r for r in requires if r in ("text", "table")]
        if not text_requires:
            text_requires = ["text", "table"]

        print(f"\r  [{i}/{len(questions)}] {q['id']}...", end="", flush=True)

        try:
            query_vec = retriever._embed_query(question)
        except Exception as e:
            print(f"\n    ⚠ Embed error: {e}")
            time.sleep(10)
            continue

        # A) Dense only
        dense_results = retriever._dense_search(
            query_vec, retriever.text_col, top_n=candidate_k
        )
        dense_chunks = _ids_to_text_chunks(retriever, dense_results[:top_k])
        _score(metrics["dense_only"], dense_chunks, doc, text_requires, top_k)

        # B) BM25 only
        bm25_results = bm25_engine.search(question, top_n=candidate_k)
        bm25_chunks = _ids_to_text_chunks(retriever, bm25_results[:top_k])
        _score(metrics["bm25_only"], bm25_chunks, doc, text_requires, top_k)

        # C) Hybrid RRF
        hybrid_results = rrf_fusion(dense_results, bm25_results, k=60)
        hybrid_chunks = _ids_to_text_chunks(retriever, hybrid_results[:top_k])
        _score(metrics["hybrid_rrf"], hybrid_chunks, doc, text_requires, top_k)

        time.sleep(0.3)

    print()
    return _summarize(strategies, metrics, "Dense vs BM25 vs Hybrid")


# ══════════════════════════════════════════════
# Thí nghiệm 3: Dense có/không ticker boost
# ══════════════════════════════════════════════

def eval_dense_boost(retriever, questions, top_k=5):
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM 3: Dense ± Ticker Boost")
    print("=" * 70)

    strategies = {
        "dense_with_boost": {"desc": "Dense + ticker boost ×1.5"},
        "dense_no_boost":   {"desc": "Dense (không boost)"},
    }
    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}

    candidate_k = top_k * 4

    for i, q in enumerate(questions, 1):
        question = q["question"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])
        text_requires = [r for r in requires if r in ("text", "table")]
        if not text_requires:
            text_requires = ["text", "table"]

        print(f"\r  [{i}/{len(questions)}] {q['id']}...", end="", flush=True)

        try:
            query_vec = retriever._embed_query(question)
        except Exception as e:
            print(f"\n    ⚠ Embed error: {e}")
            time.sleep(10)
            continue

        tickers = retriever._detect_tickers(question)

        dense_results = retriever._dense_search(
            query_vec, retriever.text_col, top_n=candidate_k
        )

        # A) With boost
        if tickers:
            boosted = _ticker_boost(dense_results, tickers, boost=1.5)
        else:
            boosted = dense_results
        boost_chunks = _ids_to_text_chunks(retriever, boosted[:top_k])
        _score(metrics["dense_with_boost"], boost_chunks, doc, text_requires, top_k)

        # B) Without boost
        no_boost_chunks = _ids_to_text_chunks(retriever, dense_results[:top_k])
        _score(metrics["dense_no_boost"], no_boost_chunks, doc, text_requires, top_k)

        time.sleep(0.3)

    print()
    return _summarize(strategies, metrics, "Dense ± Ticker Boost")


# ══════════════════════════════════════════════
# Thí nghiệm 4: Ticker-aware vs Không ticker
# ══════════════════════════════════════════════

def eval_ticker_aware(retriever, questions, text_top_k=5, image_top_k=3):
    print("\n" + "=" * 70)
    print("THÍ NGHIỆM 4: Ticker-aware vs Không Ticker")
    print("=" * 70)

    strategies = {
        "with_ticker":    {"desc": "Có ticker boost + filter"},
        "without_ticker": {"desc": "Không ticker"},
    }
    metrics = {s: {"p@k": [], "hit@k": [], "mrr": []} for s in strategies}

    total_k = text_top_k + image_top_k

    for i, q in enumerate(questions, 1):
        question = q["question"]
        doc = q.get("doc", "")
        requires = q.get("requires", [])
        if not doc:
            continue

        print(f"\r  [{i}/{len(questions)}] {q['id']}...", end="", flush=True)

        # A) With ticker (normal flow)
        result_with = _safe_retrieve(retriever, question, text_top_k, image_top_k)
        if result_with is None:
            continue
        all_with = result_with.text_chunks + result_with.image_chunks
        _score(metrics["with_ticker"], all_with, doc, requires, total_k)

        # B) Without ticker
        try:
            query_vec = retriever._embed_query(question)
        except Exception:
            time.sleep(10)
            continue

        candidate_k = text_top_k * 4
        dense_text = retriever._dense_search(
            query_vec, retriever.text_col, top_n=candidate_k
        )
        text_chunks_no = _ids_to_text_chunks(retriever, dense_text[:text_top_k])

        dense_img = retriever._dense_search(
            query_vec, retriever.image_col, top_n=image_top_k * 4
        )
        img_chunks_no = _ids_to_image_chunks(retriever, dense_img[:image_top_k])

        all_without = text_chunks_no + img_chunks_no
        _score(metrics["without_ticker"], all_without, doc, requires, total_k)

        time.sleep(0.3)

    print()
    return _summarize(strategies, metrics, "Ticker-aware vs Không")


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _safe_retrieve(retriever, question, text_top_k, image_top_k):
    for attempt in range(3):
        try:
            return retriever.retrieve(question, text_top_k=text_top_k, image_top_k=image_top_k)
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"\n    ⚠ Retrieve error (attempt {attempt + 1}): {e}")
            time.sleep(wait)
    return None


def _ids_to_text_chunks(retriever, scored):
    chunks = []
    for chunk_id, score in scored:
        c = retriever._build_text_chunk(chunk_id, score)
        if c:
            chunks.append(c)
    return chunks


def _ids_to_image_chunks(retriever, scored):
    chunks = []
    for chunk_id, score in scored:
        c = retriever._build_image_chunk(chunk_id, score)
        if c:
            chunks.append(c)
    return chunks


def _score(metrics, chunks, doc, requires, k):
    metrics["p@k"].append(precision_at_k(chunks, doc, requires, k))
    metrics["hit@k"].append(hit_rate_at_k(chunks, doc, requires, k))
    metrics["mrr"].append(mrr(chunks, doc, requires))


def _summarize(strategies, metrics, title):
    print(f"\n{'─' * 70}")
    print(f"KẾT QUẢ: {title}")
    print(f"{'─' * 70}")
    print(f"  {'Strategy':<20} {'P@k':>8} {'Hit@k':>8} {'MRR':>8}  {'N':>4}  Description")
    print(f"  {'─' * 68}")

    summary = {}
    for s in strategies:
        m = metrics[s]
        n = len(m["p@k"])
        if n == 0:
            continue
        avg_p = sum(m["p@k"]) / n
        avg_h = sum(m["hit@k"]) / n
        avg_m = sum(m["mrr"]) / n
        desc = strategies[s]["desc"]
        marker = " ★" if avg_m == max(sum(metrics[x]["mrr"]) / len(metrics[x]["mrr"]) for x in strategies if metrics[x]["mrr"]) else ""
        print(f"  {s:<20} {avg_p:>8.4f} {avg_h:>8.4f} {avg_m:>8.4f}  {n:>4}  {desc}{marker}")
        summary[s] = {
            "precision_at_k": round(avg_p, 4),
            "hit_rate_at_k": round(avg_h, 4),
            "mrr": round(avg_m, 4),
            "n": n,
        }
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

    print("Building BM25 index...")
    bm25_engine = BM25Engine(retriever)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {"text_top_k": text_top_k, "image_top_k": image_top_k},
    }

    # Thí nghiệm 1: Single vs Dual
    all_results["single_vs_dual"] = eval_single_vs_dual(
        retriever, questions, text_top_k, image_top_k
    )

    # Thí nghiệm 2: Dense vs BM25 vs Hybrid (MỚI)
    all_results["dense_vs_bm25"] = eval_dense_vs_bm25(
        retriever, bm25_engine, questions, top_k=text_top_k
    )

    # Thí nghiệm 3: Dense ± ticker boost
    all_results["dense_boost"] = eval_dense_boost(
        retriever, questions, top_k=text_top_k
    )

    # Thí nghiệm 4: Ticker-aware
    all_results["ticker_aware"] = eval_ticker_aware(
        retriever, questions, text_top_k, image_top_k
    )

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\n✓ Results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá Retrieval v2 (Dense vs BM25 vs Hybrid)"
    )
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/eval_retrieval_v2.json")
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    args = parser.parse_args()
    run(args.questions, args.output, args.text_top_k, args.image_top_k)
