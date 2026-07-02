# src/eval/eval_ticker_aware.py

"""
Đánh giá vai trò Ticker-Aware Filtering trong Retrieval.

Mục tiêu chứng minh:
  Ticker-aware filtering (soft boost text + hard filter image)
  giúp tăng độ chính xác khi truy vấn đúng công ty,
  mà không cần thêm reranker bên ngoài.

So sánh 2 chiến lược:
  A) with_ticker:    Ticker detection + boost + filter (mặc định)
  B) without_ticker: Tắt hoàn toàn ticker detection → pure Dense search

Metrics: Precision@k, Hit Rate@k, MRR — tính trên toàn bộ câu hỏi
         và tách riêng nhóm "có nhắc ticker" vs "không nhắc ticker".

Usage:
    python src/eval/eval_ticker_aware.py
    python src/eval/eval_ticker_aware.py --questions src/eval/questions.json
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
from src.retrieval.retriever import DualRetriever, RetrievedChunk


# ══════════════════════════════════════════════
# Metrics (tái sử dụng logic từ eval_retrieval.py)
# ══════════════════════════════════════════════

def precision_at_k(chunks: list[RetrievedChunk], doc: str,
                   requires: list[str], k: int) -> float:
    top = chunks[:k]
    if not top:
        return 0.0
    hits = sum(1 for c in top if _is_relevant(c, doc, requires))
    return hits / len(top)


def hit_rate_at_k(chunks: list[RetrievedChunk], doc: str,
                  requires: list[str], k: int) -> int:
    top = chunks[:k]
    return 1 if any(_is_relevant(c, doc, requires) for c in top) else 0


def mrr(chunks: list[RetrievedChunk], doc: str,
        requires: list[str]) -> float:
    for rank, c in enumerate(chunks, 1):
        if _is_relevant(c, doc, requires):
            return 1.0 / rank
    return 0.0


def _is_relevant(chunk: RetrievedChunk, doc: str,
                 requires: list[str]) -> bool:
    if doc and doc not in chunk.doc:
        return False
    if requires and chunk.chunk_type not in requires:
        return False
    return True


# ══════════════════════════════════════════════
# Safe retrieve helper
# ══════════════════════════════════════════════

def _safe_retrieve(retriever, question, text_top_k, image_top_k):
    for attempt in range(3):
        try:
            return retriever.retrieve(
                question,
                text_top_k=text_top_k,
                image_top_k=image_top_k,
            )
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"    ⚠ Retrieve error (attempt {attempt+1}): {e}")
            time.sleep(wait)
    return None


# ══════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════

def run(questions_path: str, output_path: str,
        text_top_k: int = 5, image_top_k: int = 3):

    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # ── Khởi tạo retriever ──
    print("\nInitializing DualRetriever...")
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )

    total_k = text_top_k + image_top_k

    # ── Lưu ref gốc để monkey-patch ──
    _original_detect = retriever._detect_tickers
    _original_infer = retriever._infer_tickers_from_chunks

    # ── Phân nhóm: câu hỏi có nhắc ticker vs không ──
    has_ticker_qs = []
    no_ticker_qs = []
    for q in questions:
        detected = retriever._detect_tickers(q["question"])
        if detected:
            has_ticker_qs.append(q)
        else:
            no_ticker_qs.append(q)

    print(f"\n📊 Phân nhóm câu hỏi:")
    print(f"  has_ticker  (query nhắc mã/tên công ty): {len(has_ticker_qs)} câu")
    print(f"  no_ticker   (query không nhắc công ty):   {len(no_ticker_qs)} câu")

    # ── Chạy đánh giá ──
    detail_rows = []
    results = {}

    for group_name, group_qs in [("has_ticker", has_ticker_qs),
                                  ("no_ticker", no_ticker_qs),
                                  ("all", questions)]:
        print(f"\n{'='*70}")
        print(f"NHÓM: {group_name} ({len(group_qs)} câu)")
        print(f"{'='*70}")

        metrics = {
            "with_ticker":    {"p@k": [], "hit@k": [], "mrr": []},
            "without_ticker": {"p@k": [], "hit@k": [], "mrr": []},
        }

        for i, q in enumerate(group_qs, 1):
            question = q["question"]
            qid = q["id"]
            doc = q.get("doc", "")
            requires = q.get("requires", [])

            print(f"\n  [{i}/{len(group_qs)}] {qid}: {question[:55]}...")

            # ── Strategy A: WITH ticker-aware (mặc định) ──
            # Khôi phục method gốc
            retriever._detect_tickers = _original_detect
            retriever._infer_tickers_from_chunks = _original_infer

            result_with = _safe_retrieve(retriever, question, text_top_k, image_top_k)
            if result_with is None:
                continue

            all_with = result_with.text_chunks + result_with.image_chunks
            p_w = precision_at_k(all_with, doc, requires, total_k)
            h_w = hit_rate_at_k(all_with, doc, requires, total_k)
            m_w = mrr(all_with, doc, requires)

            metrics["with_ticker"]["p@k"].append(p_w)
            metrics["with_ticker"]["hit@k"].append(h_w)
            metrics["with_ticker"]["mrr"].append(m_w)

            # ── Strategy B: WITHOUT ticker-aware ──
            # Monkey-patch: tắt ticker detection + inference
            retriever._detect_tickers = lambda query: []
            retriever._infer_tickers_from_chunks = lambda chunks: []

            result_without = _safe_retrieve(retriever, question, text_top_k, image_top_k)
            if result_without is None:
                continue

            all_without = result_without.text_chunks + result_without.image_chunks
            p_wo = precision_at_k(all_without, doc, requires, total_k)
            h_wo = hit_rate_at_k(all_without, doc, requires, total_k)
            m_wo = mrr(all_without, doc, requires)

            metrics["without_ticker"]["p@k"].append(p_wo)
            metrics["without_ticker"]["hit@k"].append(h_wo)
            metrics["without_ticker"]["mrr"].append(m_wo)

            # Log
            docs_with = [c.doc[:15] for c in all_with[:3]]
            docs_without = [c.doc[:15] for c in all_without[:3]]
            print(f"    with_ticker:    P={p_w:.2f} Hit={h_w} MRR={m_w:.3f}  top_docs={docs_with}")
            print(f"    without_ticker: P={p_wo:.2f} Hit={h_wo} MRR={m_wo:.3f}  top_docs={docs_without}")

            # Lưu chi tiết (chỉ cho group != "all" để tránh trùng)
            if group_name != "all":
                detail_rows.append({
                    "id": qid,
                    "group": group_name,
                    "category": q.get("category", ""),
                    "doc": doc,
                    "with_ticker": {
                        "p@k": p_w, "hit@k": h_w, "mrr": m_w,
                        "top3_docs": [c.doc for c in all_with[:3]],
                    },
                    "without_ticker": {
                        "p@k": p_wo, "hit@k": h_wo, "mrr": m_wo,
                        "top3_docs": [c.doc for c in all_without[:3]],
                    },
                })

            time.sleep(0.3)

        # ── Tổng kết nhóm ──
        results[group_name] = {}
        for strategy in ["with_ticker", "without_ticker"]:
            m = metrics[strategy]
            n = len(m["p@k"])
            if n == 0:
                continue
            results[group_name][strategy] = {
                "precision_at_k": round(sum(m["p@k"]) / n, 4),
                "hit_rate_at_k":  round(sum(m["hit@k"]) / n, 4),
                "mrr":            round(sum(m["mrr"]) / n, 4),
                "n":              n,
            }

    # Khôi phục lại method gốc
    retriever._detect_tickers = _original_detect
    retriever._infer_tickers_from_chunks = _original_infer

    # ── In bảng tổng kết ──
    _print_summary(results)

    # ── Save ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "text_top_k": text_top_k,
            "image_top_k": image_top_k,
            "total_k": total_k,
        },
        "summary": results,
        "detail": detail_rows,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n\n✓ Results saved → {out}")


def _print_summary(results: dict):
    """In bảng so sánh chính."""
    print(f"\n\n{'═'*75}")
    print("TỔNG KẾT: Vai trò Ticker-Aware Filtering trong Retrieval")
    print(f"{'═'*75}")

    group_labels = {
        "has_ticker": "CÂU HỎI CÓ NHẮC TICKER (VNM, Vinamilk...)",
        "no_ticker":  "CÂU HỎI KHÔNG NHẮC TICKER",
        "all":        "TOÀN BỘ CÂU HỎI",
    }

    for group in ["has_ticker", "no_ticker", "all"]:
        group_data = results.get(group, {})
        if not group_data:
            continue

        label = group_labels.get(group, group)
        print(f"\n  ▶ {label}")
        print(f"  {'Strategy':<20} {'P@k':>8} {'Hit@k':>8} {'MRR':>8}  {'N':>4}")
        print(f"  {'─'*55}")

        for strategy in ["with_ticker", "without_ticker"]:
            s = group_data.get(strategy, {})
            if not s:
                continue
            print(f"  {strategy:<20} "
                  f"{s['precision_at_k']:>8.4f} "
                  f"{s['hit_rate_at_k']:>8.4f} "
                  f"{s['mrr']:>8.4f}  "
                  f"{s['n']:>4}")

        # Delta
        wi = group_data.get("with_ticker", {})
        wo = group_data.get("without_ticker", {})
        if wi and wo:
            dp = wi["precision_at_k"] - wo["precision_at_k"]
            dh = wi["hit_rate_at_k"] - wo["hit_rate_at_k"]
            dm = wi["mrr"] - wo["mrr"]
            sign = lambda v: f"+{v:.4f}" if v >= 0 else f"{v:.4f}"
            print(f"  {'Δ (with - without)':<20} "
                  f"{sign(dp):>8} "
                  f"{sign(dh):>8} "
                  f"{sign(dm):>8}")

            if group == "has_ticker":
                if dp > 0 or dh > 0:
                    print(f"\n  ✅ Ticker-aware GIÚP tăng độ chính xác "
                          f"cho câu hỏi có nhắc mã cổ phiếu")
                    print(f"     → Precision tăng {dp:+.2%}, "
                          f"Hit rate tăng {dh:+.2%}")
                else:
                    print(f"\n  ⚠ Ticker-aware không cải thiện rõ rệt "
                          f"cho nhóm has_ticker")

            elif group == "no_ticker":
                if dh >= 0 and dp >= -0.02:
                    print(f"\n  ✅ Ticker-aware (qua majority vote fallback) "
                          f"KHÔNG gây nhiễu cho câu hỏi không nhắc ticker")
                else:
                    print(f"\n  ⚠ Cần kiểm tra: majority vote inference "
                          f"có thể suy luận sai ticker")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá vai trò Ticker-Aware Filtering (with vs without)"
    )
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/eval_ticker_aware.json")
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    args = parser.parse_args()
    run(args.questions, args.output, args.text_top_k, args.image_top_k)
