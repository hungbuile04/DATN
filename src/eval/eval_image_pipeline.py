# src/eval/eval_image_pipeline.py

"""
Đánh giá vai trò Image Pipeline trong kiến trúc Dual Collection.

Mục tiêu chứng minh 2 điều:
  1. Image pipeline GIÚP tìm đúng context cho câu hỏi visual/cross_modal
  2. Image pipeline KHÔNG làm giảm chất lượng retrieval cho câu hỏi text/table

So sánh 2 chiến lược:
  A) with_image:    Dual pipeline — text_col(top_k=5) + image_col(top_k=3)
  B) without_image: Chỉ text_col(top_k=8), bỏ image pipeline hoàn toàn

Chia câu hỏi thành 2 nhóm:
  - needs_image:  requires chứa "image"  (visual, cross_modal)
  - text_only_q:  requires KHÔNG chứa "image" (text, table)

Metrics: Precision@k, Recall@k, Hit Rate@k, MRR

Usage:
    python src/eval/eval_image_pipeline.py
    python src/eval/eval_image_pipeline.py --questions src/eval/questions.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG
from src.retrieval.retriever import DualRetriever, RetrievalResult, RetrievedChunk


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


def recall_at_k(chunks: list[RetrievedChunk], doc: str,
                requires: list[str], k: int,
                total_relevant: int = 1) -> float:
    top = chunks[:k]
    hits = sum(1 for c in top if _is_relevant(c, doc, requires))
    return hits / total_relevant if total_relevant > 0 else 0.0


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
            print(f"    Waiting {wait}s...")
            time.sleep(wait)
    print(f"    ✗ Skipping after 3 retries")
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

    # ── Phân nhóm câu hỏi ──
    needs_image = []
    text_only_q = []
    for q in questions:
        requires = q.get("requires", [])
        if "image" in requires:
            needs_image.append(q)
        else:
            text_only_q.append(q)

    print(f"\n📊 Phân nhóm câu hỏi:")
    print(f"  needs_image  (có 'image' trong requires): {len(needs_image)} câu")
    print(f"  text_only_q  (chỉ text/table):            {len(text_only_q)} câu")

    # ── Khởi tạo retriever ──
    print("\nInitializing DualRetriever...")
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )

    total_k = text_top_k + image_top_k  # = 8

    # ── Lưu kết quả theo nhóm ──
    # Cấu trúc: group → strategy → metric → [values]
    results = {}
    detail_rows = []  # per-question detail

    for group_name, group_qs in [("needs_image", needs_image),
                                  ("text_only_q", text_only_q)]:
        print(f"\n{'='*70}")
        print(f"NHÓM: {group_name} ({len(group_qs)} câu)")
        print(f"{'='*70}")

        metrics = {
            "with_image":    {"p@k": [], "recall@k": [], "hit@k": [], "mrr": []},
            "without_image": {"p@k": [], "recall@k": [], "hit@k": [], "mrr": []},
        }

        for i, q in enumerate(group_qs, 1):
            question = q["question"]
            qid = q["id"]
            doc = q.get("doc", "")
            requires = q.get("requires", [])
            category = q.get("category", "")

            print(f"\n  [{i}/{len(group_qs)}] {qid} ({category}): "
                  f"{question[:55]}...")

            # ── Strategy A: WITH image pipeline (dual) ──
            result_with = _safe_retrieve(retriever, question,
                                         text_top_k, image_top_k)
            if result_with is None:
                continue

            all_with = result_with.text_chunks + result_with.image_chunks

            p_with   = precision_at_k(all_with, doc, requires, total_k)
            r_with   = recall_at_k(all_with, doc, requires, total_k)
            h_with   = hit_rate_at_k(all_with, doc, requires, total_k)
            m_with   = mrr(all_with, doc, requires)

            metrics["with_image"]["p@k"].append(p_with)
            metrics["with_image"]["recall@k"].append(r_with)
            metrics["with_image"]["hit@k"].append(h_with)
            metrics["with_image"]["mrr"].append(m_with)

            # ── Strategy B: WITHOUT image pipeline (text only, top_k=8) ──
            result_without = _safe_retrieve(retriever, question,
                                            total_k, 0)
            if result_without is None:
                continue

            all_without = result_without.text_chunks

            p_without   = precision_at_k(all_without, doc, requires, total_k)
            r_without   = recall_at_k(all_without, doc, requires, total_k)
            h_without   = hit_rate_at_k(all_without, doc, requires, total_k)
            m_without   = mrr(all_without, doc, requires)

            metrics["without_image"]["p@k"].append(p_without)
            metrics["without_image"]["recall@k"].append(r_without)
            metrics["without_image"]["hit@k"].append(h_without)
            metrics["without_image"]["mrr"].append(m_without)

            # Log
            print(f"    with_image:    P={p_with:.2f} R={r_with:.2f} "
                  f"Hit={h_with} MRR={m_with:.3f}  "
                  f"[text={len(result_with.text_chunks)} "
                  f"img={len(result_with.image_chunks)}]")
            print(f"    without_image: P={p_without:.2f} R={r_without:.2f} "
                  f"Hit={h_without} MRR={m_without:.3f}  "
                  f"[text={len(result_without.text_chunks)} img=0]")

            # Chi tiết chunk types
            with_types = [c.chunk_type for c in all_with]
            without_types = [c.chunk_type for c in all_without]
            print(f"    chunk_types: with={with_types}")
            print(f"    chunk_types: without={without_types}")

            detail_rows.append({
                "id": qid,
                "group": group_name,
                "category": category,
                "requires": requires,
                "with_image":    {"p@k": p_with, "recall@k": r_with,
                                  "hit@k": h_with, "mrr": m_with,
                                  "n_text": len(result_with.text_chunks),
                                  "n_img": len(result_with.image_chunks),
                                  "types": with_types},
                "without_image": {"p@k": p_without, "recall@k": r_without,
                                  "hit@k": h_without, "mrr": m_without,
                                  "n_text": len(result_without.text_chunks),
                                  "n_img": 0,
                                  "types": without_types},
            })

            time.sleep(0.3)

        # ── Tổng kết nhóm ──
        results[group_name] = {}
        for strategy in ["with_image", "without_image"]:
            m = metrics[strategy]
            n = len(m["p@k"])
            if n == 0:
                continue
            results[group_name][strategy] = {
                "precision_at_k":  round(sum(m["p@k"]) / n, 4),
                "recall_at_k":     round(sum(m["recall@k"]) / n, 4),
                "hit_rate_at_k":   round(sum(m["hit@k"]) / n, 4),
                "mrr":             round(sum(m["mrr"]) / n, 4),
                "n":               n,
            }

    # ── In bảng tổng kết ──
    _print_summary(results)

    # ── Phân tích thêm theo category nhỏ ──
    _print_category_breakdown(detail_rows)

    # ── Save ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"text_top_k": text_top_k, "image_top_k": image_top_k,
                   "total_k": text_top_k + image_top_k},
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
    print("TỔNG KẾT: Vai trò Image Pipeline trong Retrieval")
    print(f"{'═'*75}")

    for group in ["needs_image", "text_only_q"]:
        group_data = results.get(group, {})
        if not group_data:
            continue

        label = ("CÂU HỎI CẦN IMAGE (visual/cross_modal)"
                 if group == "needs_image"
                 else "CÂU HỎI CHỈ CẦN TEXT/TABLE")

        print(f"\n  ▶ {label}")
        print(f"  {'Strategy':<20} {'P@k':>8} {'R@k':>8} {'Hit@k':>8} "
              f"{'MRR':>8}  {'N':>4}")
        print(f"  {'─'*60}")

        for strategy in ["with_image", "without_image"]:
            s = group_data.get(strategy, {})
            if not s:
                continue
            print(f"  {strategy:<20} "
                  f"{s['precision_at_k']:>8.4f} "
                  f"{s['recall_at_k']:>8.4f} "
                  f"{s['hit_rate_at_k']:>8.4f} "
                  f"{s['mrr']:>8.4f}  "
                  f"{s['n']:>4}")

        # Tính delta
        wi = group_data.get("with_image", {})
        wo = group_data.get("without_image", {})
        if wi and wo:
            delta_hit = wi["hit_rate_at_k"] - wo["hit_rate_at_k"]
            delta_p = wi["precision_at_k"] - wo["precision_at_k"]
            delta_mrr = wi["mrr"] - wo["mrr"]
            sign = lambda v: f"+{v:.4f}" if v >= 0 else f"{v:.4f}"
            print(f"  {'Δ (with - without)':<20} "
                  f"{sign(delta_p):>8} "
                  f"{'':>8} "
                  f"{sign(delta_hit):>8} "
                  f"{sign(delta_mrr):>8}")

            if group == "needs_image":
                if delta_hit > 0:
                    print(f"\n  ✅ Image pipeline GIÚP tìm đúng context "
                          f"(Hit@k +{delta_hit:.2%})")
                else:
                    print(f"\n  ⚠ Image pipeline không cải thiện Hit@k")
            else:
                if delta_hit >= 0 and delta_p >= -0.05:
                    print(f"\n  ✅ Image pipeline KHÔNG làm giảm chất lượng "
                          f"retrieval cho câu text/table")
                else:
                    print(f"\n  ⚠ Image pipeline có thể gây nhiễu "
                          f"cho câu text/table")


def _print_category_breakdown(detail_rows: list[dict]):
    """In breakdown theo category nhỏ."""
    print(f"\n\n{'─'*75}")
    print("BREAKDOWN THEO CATEGORY")
    print(f"{'─'*75}")

    cats = defaultdict(lambda: {"with_image": [], "without_image": []})
    for row in detail_rows:
        cat = row["category"]
        cats[cat]["with_image"].append(row["with_image"]["hit@k"])
        cats[cat]["without_image"].append(row["without_image"]["hit@k"])

    print(f"\n  {'Category':<15} {'N':>4}  "
          f"{'Hit@k (with)':>14} {'Hit@k (without)':>16} {'Δ':>8}")
    print(f"  {'─'*62}")

    for cat in ["text", "table", "visual", "cross_modal", "reasoning"]:
        if cat not in cats:
            continue
        wi_vals = cats[cat]["with_image"]
        wo_vals = cats[cat]["without_image"]
        n = len(wi_vals)
        avg_wi = sum(wi_vals) / n if n else 0
        avg_wo = sum(wo_vals) / n if n else 0
        delta = avg_wi - avg_wo
        sign = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {cat:<15} {n:>4}  {avg_wi:>14.4f} {avg_wo:>16.4f} {sign:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá vai trò Image Pipeline (Dual vs Text-only)"
    )
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/eval_image_pipeline.json")
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    args = parser.parse_args()
    run(args.questions, args.output, args.text_top_k, args.image_top_k)
