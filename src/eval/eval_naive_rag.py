# src/eval/eval_naive_rag.py

"""
Chạy Naive RAG baseline và so sánh với kết quả Full System đã có.

Naive RAG = Dense search thuần (KHÔNG ticker-aware, KHÔNG image, Single LLM).

Kết quả Full System được đọc từ file comparison.json đã chạy trước đó.

Usage:
    # Chạy naive RAG + so sánh
    python src/eval/eval_naive_rag.py

    # Chạy + Judge chấm điểm
    python src/eval/eval_naive_rag.py --judge

    # Resume nếu bị dừng giữa chừng
    python src/eval/eval_naive_rag.py --resume

    # Chạy trên tập nhỏ
    python src/eval/eval_naive_rag.py --questions src/eval/question_test.json
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


# ══════════════════════════════════════════════
# Retrieval Metrics
# ══════════════════════════════════════════════

def precision_at_k(chunks, doc, requires, k):
    top = chunks[:k]
    if not top:
        return 0.0
    return sum(1 for c in top if _is_relevant(c, doc, requires)) / len(top)

def hit_rate_at_k(chunks, doc, requires, k):
    return 1 if any(_is_relevant(c, doc, requires) for c in chunks[:k]) else 0

def mrr(chunks, doc, requires):
    for rank, c in enumerate(chunks, 1):
        if _is_relevant(c, doc, requires):
            return 1.0 / rank
    return 0.0

def _is_relevant(chunk, doc, requires):
    if doc and doc not in chunk.doc:
        return False
    if requires and chunk.chunk_type not in requires:
        return False
    return True


def run(questions_path, output_path, full_system_path,
        text_top_k=5, image_top_k=3,
        use_judge=False, judge_model="google/gemini-2.5-flash",
        resume=False):

    google_key     = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY")
    if not openrouter_key:
        raise EnvironmentError("Thiếu OPENROUTER_API_KEY")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    total_k = text_top_k + image_top_k

    # ── Load kết quả Full System đã có ──
    full_system_data = {}
    if os.path.exists(full_system_path):
        with open(full_system_path, encoding="utf-8") as f:
            raw = json.load(f)
            # runner.py output là list[dict], mỗi dict có id + modes
            for item in raw:
                qid = item.get("id", "")
                ma = item.get("modes", {}).get("multi_agent", {})
                if ma:
                    full_system_data[qid] = ma
        print(f"Loaded {len(full_system_data)} Full System results from {full_system_path}")
    else:
        print(f"⚠ Không tìm thấy {full_system_path} — chỉ chạy Naive RAG")

    # ── Khởi tạo ──
    print("Initializing DualRetriever...")
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )

    generator = AnswerGenerator(
        api_key=openrouter_key,
        text_model=CFG["agents"]["llm_model"],
        vision_model=CFG["agents"]["vision_model"],
    )

    judge = None
    if use_judge:
        from src.eval.judge import LLMJudge
        print(f"Judge ENABLED ({judge_model})")
        judge = LLMJudge(api_key=openrouter_key, model=judge_model)

    # Monkey-patch: TẮT ticker-aware
    retriever._detect_tickers = lambda query: []
    retriever._infer_tickers_from_chunks = lambda chunks: []
    print("🚫 Ticker-aware: DISABLED (Naive RAG mode)")

    # ── Load kết quả cũ nếu resume ──
    existing = {}
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                for item in json.load(f).get("detail", []):
                    existing[item["id"]] = item
            print(f"Resumed {len(existing)} entries")
        except Exception:
            pass

    # ── Chạy Naive RAG ──
    detail_rows = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid      = q.get("id", f"q{i:03d}")
        doc      = q.get("doc", "")
        requires = q.get("requires", [])
        category = q.get("category", "")
        expected = q.get("expected_answer_hint", "")

        print(f"\n[{i}/{len(questions)}] {qid}: {question[:60]}...")

        # Skip nếu đã có
        if resume and qid in existing and existing[qid].get("naive", {}).get("answer"):
            print(f"  → skipped (exists)")
            detail_rows.append(existing[qid])
            continue

        entry = {"id": qid, "question": question, "category": category,
                 "requires": requires, "doc": doc}

        try:
            # Retrieve: text+table only, KHÔNG image, KHÔNG ticker
            result = retriever.retrieve(question, text_top_k=total_k, image_top_k=0)
            chunks = result.text_chunks

            p = precision_at_k(chunks, doc, requires, total_k)
            h = hit_rate_at_k(chunks, doc, requires, total_k)
            m = mrr(chunks, doc, requires)

            # Generate: Single LLM
            gen = generator.generate(question, result, mode="text_table")

            print(f"  P={p:.2f} Hit={h} MRR={m:.3f} [{len(chunks)} chunks]")
            print(f"  → {gen['answer'][:90]}...")

            entry["naive"] = {
                "answer":      gen["answer"],
                "model":       gen["model"],
                "retrieval":   {"p@k": p, "hit@k": h, "mrr": m},
                "n_chunks":    len(chunks),
                "chunk_types": [c.chunk_type for c in chunks],
                "top3_docs":   [c.doc for c in chunks[:3]],
            }

        except Exception as e:
            print(f"  ✗ Error: {e}")
            entry["naive"] = {"error": str(e)}
            time.sleep(15)

        # Judge naive answer
        if judge and expected and entry.get("naive", {}).get("answer"):
            if not (resume and existing.get(qid, {}).get("naive", {}).get("judge")):
                try:
                    print(f"  Judge...", end=" ", flush=True)
                    score = judge.evaluate(question, entry["naive"]["answer"], expected)
                    entry["naive"]["judge"] = score
                    print(f"→ {score['total']}/20")
                except Exception as e:
                    print(f"error: {e}")
            else:
                entry["naive"]["judge"] = existing[qid]["naive"]["judge"]

        detail_rows.append(entry)

        # Incremental save
        _save(output_path, detail_rows)
        time.sleep(0.3)

    # ── In bảng so sánh ──
    _print_comparison(detail_rows, full_system_data, total_k)

    print(f"\n✓ Saved → {output_path}")


def _save(output_path, detail_rows):
    n = len([r for r in detail_rows if r.get("naive", {}).get("retrieval")])
    if n == 0:
        summary = {}
    else:
        rets = [r["naive"]["retrieval"] for r in detail_rows
                if r.get("naive", {}).get("retrieval")]
        summary = {
            "precision_at_k": round(sum(r["p@k"] for r in rets) / len(rets), 4),
            "hit_rate_at_k":  round(sum(r["hit@k"] for r in rets) / len(rets), 4),
            "mrr":            round(sum(r["mrr"] for r in rets) / len(rets), 4),
            "n": len(rets),
        }
        judges = [r["naive"]["judge"] for r in detail_rows
                  if r.get("naive", {}).get("judge")]
        if judges:
            summary["judge_avg"] = round(
                sum(j["total"] for j in judges) / len(judges), 2)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "summary": summary, "detail": detail_rows},
                  f, ensure_ascii=False, indent=2)


def _print_comparison(detail_rows, full_system_data, total_k):
    """In bảng so sánh Naive RAG vs Full System."""
    print(f"\n\n{'═'*75}")
    print(f"SO SÁNH: Naive RAG  vs  Full System (Proposed)")
    print(f"{'═'*75}")

    # ── Thu thập naive metrics ──
    naive_rets = [r["naive"]["retrieval"] for r in detail_rows
                  if r.get("naive", {}).get("retrieval")]
    naive_judges = [r["naive"]["judge"] for r in detail_rows
                    if r.get("naive", {}).get("judge")]

    # ── Thu thập full system judge từ comparison.json ──
    full_judges = []
    for r in detail_rows:
        qid = r["id"]
        fs = full_system_data.get(qid, {})
        if fs.get("judge"):
            full_judges.append(fs["judge"])

    # ── Retrieval ──
    if naive_rets:
        n = len(naive_rets)
        np = sum(r["p@k"] for r in naive_rets) / n
        nh = sum(r["hit@k"] for r in naive_rets) / n
        nm = sum(r["mrr"] for r in naive_rets) / n

        print(f"\n  📦 RETRIEVAL (Naive RAG — {n} câu)")
        print(f"     P@{total_k} = {np:.4f}    Hit@{total_k} = {nh:.4f}    MRR = {nm:.4f}")
        print(f"\n  (Full System retrieval metrics: xem file comparison.json)")

    # ── Judge comparison ──
    if naive_judges:
        n_nj = len(naive_judges)
        avg_nc = sum(j.get("correctness", 0) for j in naive_judges) / n_nj
        avg_no = sum(j.get("completeness", 0) for j in naive_judges) / n_nj
        avg_nr = sum(j.get("relevance", 0) for j in naive_judges) / n_nj
        avg_nf = sum(j.get("faithfulness", 0) for j in naive_judges) / n_nj
        avg_nt = sum(j.get("total", 0) for j in naive_judges) / n_nj

        print(f"\n\n  🏆 LLM-AS-JUDGE")
        print(f"  {'Mode':<20} {'Correct':>8} {'Complete':>9} "
              f"{'Relevant':>9} {'Faithful':>9} {'TOTAL':>9}")
        print(f"  {'─'*70}")
        print(f"  {'Naive RAG':<20} {avg_nc:>8.2f} {avg_no:>9.2f} "
              f"{avg_nr:>9.2f} {avg_nf:>9.2f} {avg_nt:>9.2f}/20")

        if full_judges:
            n_fj = len(full_judges)
            avg_fc = sum(j.get("correctness", 0) for j in full_judges) / n_fj
            avg_fo = sum(j.get("completeness", 0) for j in full_judges) / n_fj
            avg_fr = sum(j.get("relevance", 0) for j in full_judges) / n_fj
            avg_ff = sum(j.get("faithfulness", 0) for j in full_judges) / n_fj
            avg_ft = sum(j.get("total", 0) for j in full_judges) / n_fj
            print(f"  {'Full System':<20} {avg_fc:>8.2f} {avg_fo:>9.2f} "
                  f"{avg_fr:>9.2f} {avg_ff:>9.2f} {avg_ft:>9.2f}/20")

            delta = avg_ft - avg_nt
            print(f"\n  Δ Total: {delta:+.2f}/20", end="")
            if delta > 0:
                pct = (delta / avg_nt * 100) if avg_nt > 0 else 0
                print(f"  → Full System tốt hơn {pct:.1f}%")
            else:
                print()

    # ── Breakdown theo category ──
    print(f"\n\n{'─'*75}")
    print("BREAKDOWN THEO CATEGORY")
    print(f"{'─'*75}")

    cats = defaultdict(lambda: {"naive_hit": [], "naive_judge": [],
                                "full_judge": []})
    for row in detail_rows:
        cat = row.get("category", "unknown")
        nr = row.get("naive", {}).get("retrieval", {})
        if nr:
            cats[cat]["naive_hit"].append(nr.get("hit@k", 0))
        nj = row.get("naive", {}).get("judge", {})
        if nj:
            cats[cat]["naive_judge"].append(nj.get("total", 0))
        fs = full_system_data.get(row["id"], {})
        if fs.get("judge"):
            cats[cat]["full_judge"].append(fs["judge"].get("total", 0))

    has_full = any(cats[c]["full_judge"] for c in cats)
    header = f"  {'Category':<15} {'N':>3}  {'Hit(naive)':>11}"
    if has_full:
        header += f"  {'Judge(naive)':>13} {'Judge(full)':>12} {'Δ':>6}"
    print(header)
    print(f"  {'─'*70}")

    for cat in ["text", "table", "visual", "cross_modal", "reasoning"]:
        if cat not in cats:
            continue
        c = cats[cat]
        n = len(c["naive_hit"])
        if n == 0:
            continue
        nh = sum(c["naive_hit"]) / n
        line = f"  {cat:<15} {n:>3}  {nh:>11.4f}"
        if has_full and c["naive_judge"] and c["full_judge"]:
            nj = sum(c["naive_judge"]) / len(c["naive_judge"])
            fj = sum(c["full_judge"]) / len(c["full_judge"])
            d = fj - nj
            line += f"  {nj:>13.1f} {fj:>12.1f} {d:>+6.1f}"
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chạy Naive RAG baseline")
    p.add_argument("--questions", default="src/eval/questions.json")
    p.add_argument("--output", default="results/eval_naive_rag.json")
    p.add_argument("--full_system", default="results/comparison.json",
                   help="File chứa kết quả Full System đã chạy")
    p.add_argument("--text_top_k", type=int, default=5)
    p.add_argument("--image_top_k", type=int, default=3)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--judge_model", default="google/gemini-2.5-flash")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    run(args.questions, args.output, args.full_system,
        args.text_top_k, args.image_top_k,
        args.judge, args.judge_model, args.resume)
