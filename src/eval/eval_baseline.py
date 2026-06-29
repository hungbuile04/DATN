# src/eval/eval_baseline.py

"""
So sánh Naive RAG vs Full System (Proposed).

═══════════════════════════════════════════════════════════════
  Naive RAG (baseline):
    - Dense search thuần → KHÔNG ticker-aware
    - Text+table collection only → KHÔNG image pipeline
    - Single LLM call → KHÔNG multi-agent

  Full System (proposed):
    - Dense search + Ticker-aware (boost + filter)
    - Dual pipeline (text+table + image)
    - Multi-Agent 4 tầng (Critical → Text → Image → Sum)
═══════════════════════════════════════════════════════════════

Output: bảng so sánh Retrieval metrics + LLM-as-Judge scores.

Usage:
    # Bước 1: Chạy sinh câu trả lời (lâu ~1-2h cho 70 câu)
    python src/eval/eval_baseline.py --questions src/eval/questions.json

    # Bước 2: Chạy thêm Judge chấm điểm (resume câu trả lời cũ)
    python src/eval/eval_baseline.py --judge --resume

    # Chạy nhanh trên tập nhỏ để test
    python src/eval/eval_baseline.py --questions src/eval/question_test.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG
from src.retrieval.retriever import DualRetriever, RetrievalResult, RetrievedChunk
from src.eval.generator import AnswerGenerator
from src.agents.orchestrator import MultiAgentOrchestrator


# ══════════════════════════════════════════════
# Retrieval Metrics
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
# Safe helpers
# ══════════════════════════════════════════════

def _safe_call(fn, *args, retries=3, wait_base=15, **kwargs):
    """Retry wrapper cho API calls."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait = wait_base * (attempt + 1)
            print(f"    ⚠ Error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"    Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def run(questions_path: str, output_path: str,
        text_top_k: int = 5, image_top_k: int = 3,
        use_judge: bool = False, judge_model: str = "google/gemini-2.5-flash",
        resume: bool = False):

    google_key     = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")
    if not openrouter_key:
        raise EnvironmentError("Thiếu OPENROUTER_API_KEY trong .env")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    total_k = text_top_k + image_top_k  # = 8

    # ── Khởi tạo components ──
    print("Initializing DualRetriever...")
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )

    print("Initializing AnswerGenerator (single LLM)...")
    generator = AnswerGenerator(
        api_key=openrouter_key,
        text_model=CFG["agents"]["llm_model"],
        vision_model=CFG["agents"]["vision_model"],
    )

    print("Initializing MultiAgentOrchestrator (4 agents)...")
    orchestrator = MultiAgentOrchestrator(
        api_key=openrouter_key,
        text_model=CFG["agents"]["llm_model"],
        vision_model=CFG["agents"]["vision_model"],
    )

    # ── Judge (nếu bật) ──
    judge = None
    if use_judge:
        from src.eval.judge import LLMJudge
        print(f"LLM-as-a-Judge ENABLED (model={judge_model})")
        judge = LLMJudge(api_key=openrouter_key, model=judge_model)

    # ── Lưu ref gốc để monkey-patch ──
    _original_detect = retriever._detect_tickers
    _original_infer  = retriever._infer_tickers_from_chunks

    # ── Load kết quả cũ nếu resume ──
    existing = {}
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                for item in old_data.get("detail", []):
                    existing[item["id"]] = item
            print(f"Resumed {len(existing)} entries from {output_path}")
        except Exception as e:
            print(f"Cannot read {output_path}: {e}")

    # ── Chạy từng câu hỏi ──
    detail_rows = []
    retrieval_metrics = {
        "naive":    {"p@k": [], "hit@k": [], "mrr": []},
        "proposed": {"p@k": [], "hit@k": [], "mrr": []},
    }
    judge_scores = {
        "naive":    [],
        "proposed": [],
    }

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid      = q.get("id", f"q{i:03d}")
        doc      = q.get("doc", "")
        requires = q.get("requires", [])
        category = q.get("category", "")
        expected = q.get("expected_answer_hint", "")

        print(f"\n{'='*65}")
        print(f"[{i}/{len(questions)}] {qid} ({category})")
        print(f"  Q: {question[:70]}...")

        entry = existing.get(qid, {
            "id": qid,
            "question": question,
            "category": category,
            "requires": requires,
            "doc": doc,
            "naive": {},
            "proposed": {},
        })

        # ════════════════════════════════════════
        # A) NAIVE RAG
        # ════════════════════════════════════════
        if not (resume and entry.get("naive", {}).get("answer")):
            try:
                print(f"  ── NAIVE RAG ──")

                # Tắt ticker-aware
                retriever._detect_tickers = lambda query: []
                retriever._infer_tickers_from_chunks = lambda chunks: []

                # Retrieve: text+table only, KHÔNG image
                naive_result = _safe_call(
                    retriever.retrieve, question,
                    text_top_k=total_k, image_top_k=0,
                )
                naive_chunks = naive_result.text_chunks

                # Retrieval metrics
                p  = precision_at_k(naive_chunks, doc, requires, total_k)
                h  = hit_rate_at_k(naive_chunks, doc, requires, total_k)
                m  = mrr(naive_chunks, doc, requires)
                print(f"     Retrieval: P@{total_k}={p:.2f} Hit={h} MRR={m:.3f} "
                      f"[{len(naive_chunks)} chunks]")

                # Generation: single LLM
                naive_gen = _safe_call(
                    generator.generate, question, naive_result, mode="text_table",
                )
                print(f"     Answer: {naive_gen['answer'][:100]}...")

                entry["naive"] = {
                    "answer":     naive_gen["answer"],
                    "model":      naive_gen["model"],
                    "has_image":  False,
                    "retrieval":  {"p@k": p, "hit@k": h, "mrr": m},
                    "n_chunks":   len(naive_chunks),
                    "chunk_types": [c.chunk_type for c in naive_chunks],
                    "top3_docs":  [c.doc for c in naive_chunks[:3]],
                }

            except Exception as e:
                print(f"     ✗ NAIVE failed: {e}")
                entry["naive"] = {"error": str(e)}
        else:
            print(f"  ── NAIVE RAG ── (skipped, exists)")

        # ════════════════════════════════════════
        # B) FULL SYSTEM (PROPOSED)
        # ════════════════════════════════════════
        if not (resume and entry.get("proposed", {}).get("answer")):
            try:
                print(f"  ── FULL SYSTEM ──")

                # Khôi phục ticker-aware
                retriever._detect_tickers = _original_detect
                retriever._infer_tickers_from_chunks = _original_infer

                # Retrieve: dual pipeline (text + image)
                full_result = _safe_call(
                    retriever.retrieve, question,
                    text_top_k=text_top_k, image_top_k=image_top_k,
                )
                full_chunks = full_result.text_chunks + full_result.image_chunks

                # Retrieval metrics
                p  = precision_at_k(full_chunks, doc, requires, total_k)
                h  = hit_rate_at_k(full_chunks, doc, requires, total_k)
                m  = mrr(full_chunks, doc, requires)
                print(f"     Retrieval: P@{total_k}={p:.2f} Hit={h} MRR={m:.3f} "
                      f"[text={len(full_result.text_chunks)} img={len(full_result.image_chunks)}]")

                # Generation: Multi-Agent 4 tầng
                ma_gen = _safe_call(orchestrator.run, question, full_result)
                print(f"     Answer: {ma_gen['answer'][:100]}...")

                entry["proposed"] = {
                    "answer":      ma_gen["answer"],
                    "model":       ma_gen["model"],
                    "has_image":   ma_gen["has_image"],
                    "confidence":  ma_gen["confidence"],
                    "retrieval":   {"p@k": p, "hit@k": h, "mrr": m},
                    "n_text":      len(full_result.text_chunks),
                    "n_image":     len(full_result.image_chunks),
                    "chunk_types": [c.chunk_type for c in full_chunks],
                    "top3_docs":   [c.doc for c in full_chunks[:3]],
                }

            except Exception as e:
                print(f"     ✗ PROPOSED failed: {e}")
                entry["proposed"] = {"error": str(e)}
        else:
            print(f"  ── FULL SYSTEM ── (skipped, exists)")

        # ════════════════════════════════════════
        # C) JUDGE (nếu bật)
        # ════════════════════════════════════════
        if judge and expected:
            for mode in ["naive", "proposed"]:
                m_data = entry.get(mode, {})
                if not m_data.get("answer"):
                    continue
                if resume and m_data.get("judge"):
                    print(f"     Judge [{mode}] skipped (exists: {m_data['judge'].get('total')}/20)")
                    continue
                try:
                    print(f"     Judge [{mode}]...", end=" ", flush=True)
                    score = judge.evaluate(question, m_data["answer"], expected)
                    m_data["judge"] = score
                    print(f"→ {score['total']}/20")
                except Exception as e:
                    print(f"error: {e}")

        # Thu thập metrics
        for mode in ["naive", "proposed"]:
            m_data = entry.get(mode, {})
            ret = m_data.get("retrieval", {})
            if ret:
                retrieval_metrics[mode]["p@k"].append(ret["p@k"])
                retrieval_metrics[mode]["hit@k"].append(ret["hit@k"])
                retrieval_metrics[mode]["mrr"].append(ret["mrr"])
            if m_data.get("judge"):
                judge_scores[mode].append(m_data["judge"])

        detail_rows.append(entry)

        # Incremental save
        _save(output_path, detail_rows, retrieval_metrics, judge_scores)

        time.sleep(0.3)

    # Khôi phục
    retriever._detect_tickers = _original_detect
    retriever._infer_tickers_from_chunks = _original_infer

    # ── In bảng tổng kết ──
    _print_summary(retrieval_metrics, judge_scores, len(questions))

    # ── Breakdown theo category ──
    _print_category_breakdown(detail_rows)

    print(f"\n✓ Results saved → {output_path}")


def _save(output_path, detail_rows, retrieval_metrics, judge_scores):
    """Lưu kết quả (incremental)."""
    summary = {}
    for mode in ["naive", "proposed"]:
        rm = retrieval_metrics[mode]
        n = len(rm["p@k"])
        if n == 0:
            continue
        s = {
            "precision_at_k": round(sum(rm["p@k"]) / n, 4),
            "hit_rate_at_k":  round(sum(rm["hit@k"]) / n, 4),
            "mrr":            round(sum(rm["mrr"]) / n, 4),
            "n":              n,
        }
        js = judge_scores[mode]
        if js:
            s["judge_avg_total"] = round(sum(j["total"] for j in js) / len(js), 2)
        summary[mode] = s

    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "detail": detail_rows,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def _print_summary(retrieval_metrics, judge_scores, total_q):
    """In bảng so sánh chính."""
    print(f"\n\n{'═'*75}")
    print(f"TỔNG KẾT: Naive RAG vs Full System (Proposed)")
    print(f"{'═'*75}")

    # ── Retrieval metrics ──
    print(f"\n  📦 RETRIEVAL METRICS")
    print(f"  {'Mode':<20} {'P@k':>8} {'Hit@k':>8} {'MRR':>8}  {'N':>4}")
    print(f"  {'─'*55}")

    for mode, label in [("naive", "Naive RAG"), ("proposed", "Full System")]:
        rm = retrieval_metrics[mode]
        n = len(rm["p@k"])
        if n == 0:
            continue
        p = sum(rm["p@k"]) / n
        h = sum(rm["hit@k"]) / n
        m = sum(rm["mrr"]) / n
        print(f"  {label:<20} {p:>8.4f} {h:>8.4f} {m:>8.4f}  {n:>4}")

    # Delta
    for metric_name, metric_key in [("P@k", "p@k"), ("Hit@k", "hit@k"), ("MRR", "mrr")]:
        n_vals = retrieval_metrics["naive"][metric_key]
        p_vals = retrieval_metrics["proposed"][metric_key]
        if n_vals and p_vals:
            dn = sum(n_vals) / len(n_vals)
            dp = sum(p_vals) / len(p_vals)
            delta = dp - dn
            sign = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            if metric_key == "p@k":
                print(f"\n  Δ Precision:  {sign}  ({'↑ Proposed tốt hơn' if delta > 0 else '↓ Naive tốt hơn'})")
            elif metric_key == "hit@k":
                print(f"  Δ Hit Rate:   {sign}  ({'↑ Proposed tốt hơn' if delta > 0 else '↓ Naive tốt hơn'})")
            elif metric_key == "mrr":
                print(f"  Δ MRR:        {sign}  ({'↑ Proposed tốt hơn' if delta > 0 else '↓ Naive tốt hơn'})")

    # ── Judge scores ──
    if judge_scores["naive"] or judge_scores["proposed"]:
        print(f"\n\n  🏆 LLM-AS-JUDGE SCORES (trung bình)")
        print(f"  {'Mode':<20} {'Correct':>9} {'Complete':>9} "
              f"{'Relevant':>9} {'Faithful':>9} {'TOTAL':>8}")
        print(f"  {'─'*70}")

        for mode, label in [("naive", "Naive RAG"), ("proposed", "Full System")]:
            js = judge_scores[mode]
            if not js:
                continue
            n = len(js)
            avg_c = sum(j.get("correctness", 0) for j in js) / n
            avg_o = sum(j.get("completeness", 0) for j in js) / n
            avg_r = sum(j.get("relevance", 0) for j in js) / n
            avg_f = sum(j.get("faithfulness", 0) for j in js) / n
            avg_t = sum(j.get("total", 0) for j in js) / n
            print(f"  {label:<20} {avg_c:>9.2f} {avg_o:>9.2f} "
                  f"{avg_r:>9.2f} {avg_f:>9.2f} {avg_t:>8.2f}/20")

        # Delta total
        naive_js = judge_scores["naive"]
        proposed_js = judge_scores["proposed"]
        if naive_js and proposed_js:
            naive_avg = sum(j["total"] for j in naive_js) / len(naive_js)
            prop_avg  = sum(j["total"] for j in proposed_js) / len(proposed_js)
            delta = prop_avg - naive_avg
            print(f"\n  Δ Total Score: {delta:+.2f}/20  "
                  f"({'↑ Full System vượt trội' if delta > 0 else '⚠ Cần kiểm tra'})")
            if delta > 0:
                pct = (delta / naive_avg) * 100 if naive_avg > 0 else 0
                print(f"  → Full System cải thiện {pct:.1f}% so với Naive RAG")


def _print_category_breakdown(detail_rows):
    """In breakdown theo category."""
    print(f"\n\n{'─'*75}")
    print("BREAKDOWN THEO CATEGORY")
    print(f"{'─'*75}")

    cats = defaultdict(lambda: {
        "naive_hit": [], "proposed_hit": [],
        "naive_judge": [], "proposed_judge": [],
    })

    for row in detail_rows:
        cat = row.get("category", "unknown")
        n_ret = row.get("naive", {}).get("retrieval", {})
        p_ret = row.get("proposed", {}).get("retrieval", {})
        if n_ret:
            cats[cat]["naive_hit"].append(n_ret.get("hit@k", 0))
        if p_ret:
            cats[cat]["proposed_hit"].append(p_ret.get("hit@k", 0))
        n_judge = row.get("naive", {}).get("judge", {})
        p_judge = row.get("proposed", {}).get("judge", {})
        if n_judge:
            cats[cat]["naive_judge"].append(n_judge.get("total", 0))
        if p_judge:
            cats[cat]["proposed_judge"].append(p_judge.get("total", 0))

    # Header
    has_judge = any(cats[c]["naive_judge"] or cats[c]["proposed_judge"] for c in cats)
    if has_judge:
        print(f"\n  {'Category':<15} {'N':>4}  "
              f"{'Hit(naive)':>11} {'Hit(full)':>10} {'Δ Hit':>7}  "
              f"{'Judge(naive)':>13} {'Judge(full)':>12} {'Δ Judge':>8}")
        print(f"  {'─'*90}")
    else:
        print(f"\n  {'Category':<15} {'N':>4}  "
              f"{'Hit(naive)':>11} {'Hit(full)':>10} {'Δ Hit':>7}")
        print(f"  {'─'*55}")

    for cat in ["text", "table", "visual", "cross_modal", "reasoning"]:
        if cat not in cats:
            continue
        c = cats[cat]
        n = len(c["naive_hit"])
        if n == 0:
            continue

        nh = sum(c["naive_hit"]) / n
        ph = sum(c["proposed_hit"]) / n
        dh = ph - nh
        sign_h = f"+{dh:.2f}" if dh >= 0 else f"{dh:.2f}"

        line = f"  {cat:<15} {n:>4}  {nh:>11.4f} {ph:>10.4f} {sign_h:>7}"

        if has_judge and c["naive_judge"] and c["proposed_judge"]:
            nj = sum(c["naive_judge"]) / len(c["naive_judge"])
            pj = sum(c["proposed_judge"]) / len(c["proposed_judge"])
            dj = pj - nj
            sign_j = f"+{dj:.1f}" if dj >= 0 else f"{dj:.1f}"
            line += f"  {nj:>13.1f} {pj:>12.1f} {sign_j:>8}"

        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="So sánh Naive RAG vs Full System (Proposed)"
    )
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/eval_baseline.json")
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    parser.add_argument("--judge", action="store_true",
                        help="Bật LLM-as-a-Judge chấm điểm")
    parser.add_argument("--judge_model", type=str,
                        default="google/gemini-2.5-flash")
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục từ kết quả cũ (skip câu đã có)")
    args = parser.parse_args()

    run(
        args.questions, args.output,
        args.text_top_k, args.image_top_k,
        use_judge=args.judge,
        judge_model=args.judge_model,
        resume=args.resume,
    )
