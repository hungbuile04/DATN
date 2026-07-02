# src/eval/runner.py

"""
Chạy thí nghiệm so sánh 4 chế độ retrieval + generation.

4 modes:
    text_only       — baseline: chỉ text chunks → single LLM
    text_table      — text + table → single LLM
    full_multimodal — text + table + image → single vision LLM
    multi_agent     — text + table + image → 4 agents (Text, Table, Image, Sum)

Usage:
    python src/eval/runner.py \\
        --questions src/eval/questions.json \\
        --output    results/comparison.json \\
        --text_top_k 5 --image_top_k 3

    # Bật LLM-as-a-Judge scoring:
    python src/eval/runner.py --judge
    python src/eval/runner.py --judge --judge_model google/gemini-2.5-flash

API keys:
    GOOGLE_API_KEY     → cho dual retriever (Gemini Embedding 2)
    OPENROUTER_API_KEY → cho LLM generation + judge
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG
from src.retrieval.retriever import DualRetriever
from src.retrieval.query_router import classify_query
from src.eval.modes import TextOnlyRetriever, TextTableRetriever, FullMultimodalRetriever
from src.eval.generator import AnswerGenerator
from src.agents.orchestrator import MultiAgentOrchestrator
from src.eval.judge import LLMJudge, print_judge_summary


def run_comparison(
    questions_path: str,
    output_path: str,
    text_top_k: int = 5,
    image_top_k: int = 3,
    use_judge: bool = False,
    judge_model: str = "openai/gpt-4o-mini",
    skip_baselines: bool = False,
    skip_multi_agent: bool = False,
    judge_target: str = "all",
    resume: bool = False,
) -> list[dict]:
    """Chạy so sánh 4 chế độ trên toàn bộ câu hỏi."""
    google_key     = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")
    if not openrouter_key:
        raise EnvironmentError("Thiếu OPENROUTER_API_KEY trong .env")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    # ── Khởi tạo dual retriever ──
    print("Initializing DualRetriever...")
    base = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )

    retrievers = {
        "text_only":       TextOnlyRetriever(base),
        "text_table":      TextTableRetriever(base),
        "full_multimodal": FullMultimodalRetriever(base),
    }

    # ── Generator (OpenRouter) — cho 3 modes đầu ──
    generator = AnswerGenerator(
        api_key=openrouter_key,
        text_model=CFG["agents"]["llm_model"],
        vision_model=CFG["agents"]["vision_model"],
    )

    # ── Multi-Agent Orchestrator — cho mode thứ 4 ──
    orchestrator = MultiAgentOrchestrator(
        api_key=openrouter_key,
        text_model=CFG["agents"]["llm_model"],
        vision_model=CFG["agents"]["vision_model"],
    )

    # ── LLM Judge (nếu bật) ──
    judge = None
    if use_judge:
        print(f"LLM-as-a-Judge ENABLED (model={judge_model})")
        judge = LLMJudge(
            api_key=openrouter_key,
            model=judge_model,
        )

    # Tải output cũ nếu có để resume
    existing_results = {}
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                for item in old_data:
                    existing_results[item["id"]] = item
            print(f"Resumed {len(existing_results)} entries from {output_path}")
        except Exception as e:
            print(f"Không thể đọc file {output_path} để resume: {e}")

    # ── Chạy từng câu hỏi ──
    results: list[dict] = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        question      = q["question"]
        expected_hint = q.get("expected_answer_hint", "")
        query_type    = classify_query(question)

        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {question}")
        print(f"  category={q.get('category','')} | query_type={query_type}")

        q_id = q.get("id", f"q{i:03d}")

        entry: dict = {
            "id":                   q_id,
            "question":             question,
            "category":             q.get("category", ""),
            "requires":             q.get("requires", []),
            "expected_answer_hint": expected_hint,
            "query_type":           query_type,
            "timestamp":            datetime.now().isoformat(),
            "modes":                {},
        }

        if resume and q_id in existing_results:
            entry["modes"] = existing_results[q_id].get("modes", {})

        try:
            # ── Modes 1-3: single generator ──
            if not skip_baselines:
                for mode_name, retriever in retrievers.items():
                    if resume and mode_name in entry["modes"] and "answer" in entry["modes"][mode_name]:
                        print(f"  → [{mode_name}] skipped (exists)")
                        continue

                    print(f"  → [{mode_name}] retrieving...", end=" ", flush=True)

                    if mode_name == "full_multimodal":
                        result = retriever.retrieve(
                            question,
                            text_top_k=text_top_k,
                            image_top_k=image_top_k,
                        )
                    else:
                        result = retriever.retrieve(question, top_k=text_top_k)

                    gen = generator.generate(question, result, mode=mode_name)

                    text_types  = [c.chunk_type for c in result.text_chunks]
                    image_types = [c.chunk_type for c in result.image_chunks]
                    print(f"text={text_types} img={image_types} {'📷' if gen['has_image'] else ''}")
                    print(f"     {gen['answer'][:120]}...")

                    old_judge = entry["modes"].get(mode_name, {}).get("judge", None)
                    entry["modes"][mode_name] = {
                        "answer":       gen["answer"],
                        "model":        gen["model"],
                        "has_image":    gen["has_image"],
                        "num_images":   gen["num_images"],
                        "num_text":     len(result.text_chunks),
                        "num_img":      len(result.image_chunks),
                        "text_types":   text_types,
                        "image_types":  image_types,
                        "text_chunks":  [_chunk_to_dict(c) for c in result.text_chunks],
                        "image_chunks": [_chunk_to_dict(c) for c in result.image_chunks],
                    }
                    if old_judge:
                        entry["modes"][mode_name]["judge"] = old_judge

            # ── Mode 4: multi-agent ──
            if not skip_multi_agent:
                if not (resume and "multi_agent" in entry["modes"] and "answer" in entry["modes"]["multi_agent"]):
                    print(f"  → [multi_agent] retrieving + reasoning...")
                    ma_result = base.retrieve(
                        question,
                        text_top_k=text_top_k,
                        image_top_k=image_top_k,
                    )
                    ma_gen = orchestrator.run(question, ma_result)

                    text_types  = [c.chunk_type for c in ma_result.text_chunks]
                    image_types = [c.chunk_type for c in ma_result.image_chunks]
                    print(f"     text={text_types} img={image_types}")
                    print(f"     {ma_gen['answer'][:120]}...")

                    old_judge = entry["modes"].get("multi_agent", {}).get("judge", None)
                    entry["modes"]["multi_agent"] = {
                        "answer":        ma_gen["answer"],
                        "model":         ma_gen["model"],
                        "has_image":     ma_gen["has_image"],
                        "num_images":    ma_gen["num_images"],
                        "confidence":    ma_gen["confidence"],
                        "reasoning":     ma_gen["reasoning"],
                        "agent_outputs": ma_gen["agent_outputs"],
                        "num_text":      len(ma_result.text_chunks),
                        "num_img":       len(ma_result.image_chunks),
                        "text_types":    text_types,
                        "image_types":   image_types,
                        "text_chunks":   [_chunk_to_dict(c) for c in ma_result.text_chunks],
                        "image_chunks":  [_chunk_to_dict(c) for c in ma_result.image_chunks],
                    }
                    if old_judge:
                        entry["modes"]["multi_agent"]["judge"] = old_judge
                else:
                    print(f"  → [multi_agent] skipped (exists)")

            # ── Judge scoring (nếu bật) ──
            if judge and expected_hint:
                print(f"  → Judging modes...", end=" ")
                
                MODES_TO_JUDGE = []
                if judge_target in ["all", "baselines"]:
                    MODES_TO_JUDGE.extend(["text_only", "text_table", "full_multimodal"])
                if judge_target in ["all", "multi_agent"]:
                    MODES_TO_JUDGE.append("multi_agent")
                    
                print(f"Targets: {MODES_TO_JUDGE}")
                
                for mode_name in MODES_TO_JUDGE:
                    m = entry["modes"].get(mode_name, {})
                    if not m:
                        continue
                        
                    if resume and m.get("judge"):
                        print(f"    [{mode_name}] judge skipped (exists: {m['judge'].get('total')}/20)")
                        continue
                        
                    answer_text = m.get("answer", "")
                    if not answer_text:
                        continue
                        
                    print(f"    [{mode_name}]", end=" ", flush=True)
                    score = judge.evaluate(question, answer_text, expected_hint)
                    m["judge"] = score
                    print(f"→ {score['total']}/20")

        except (Exception) as e:
            print(f"\n  ⚠ ERROR at question {i}: {type(e).__name__}: {e}")
            print(f"    Retrying in 15s...")
            time.sleep(15)
            try:
                # Retry 1 lần cho câu này
                if not skip_baselines:
                    for mode_name, retriever in retrievers.items():
                        if mode_name in entry["modes"] and "answer" in entry["modes"][mode_name]:
                            continue
                        print(f"  → [RETRY {mode_name}] retrieving...", end=" ", flush=True)
                        if mode_name == "full_multimodal":
                            result = retriever.retrieve(question, text_top_k=text_top_k, image_top_k=image_top_k)
                        else:
                            result = retriever.retrieve(question, top_k=text_top_k)
                        gen = generator.generate(question, result, mode=mode_name)
                        text_types  = [c.chunk_type for c in result.text_chunks]
                        image_types = [c.chunk_type for c in result.image_chunks]
                        print(f"OK")
                        entry["modes"][mode_name] = {
                            "answer": gen["answer"], "model": gen["model"],
                            "has_image": gen["has_image"], "num_images": gen["num_images"],
                            "num_text": len(result.text_chunks), "num_img": len(result.image_chunks),
                            "text_types": text_types, "image_types": image_types,
                            "text_chunks": [_chunk_to_dict(c) for c in result.text_chunks],
                            "image_chunks": [_chunk_to_dict(c) for c in result.image_chunks],
                        }
                if not skip_multi_agent:
                    if "multi_agent" not in entry["modes"] or "answer" not in entry["modes"]["multi_agent"]:
                        print(f"  → [RETRY multi_agent]...", end=" ", flush=True)
                        ma_result = base.retrieve(question, text_top_k=text_top_k, image_top_k=image_top_k)
                        ma_gen = orchestrator.run(question, ma_result)
                        print(f"OK")
                        entry["modes"]["multi_agent"] = {
                            "answer": ma_gen["answer"], "model": ma_gen["model"],
                            "has_image": ma_gen["has_image"], "num_images": ma_gen["num_images"],
                            "confidence": ma_gen["confidence"], "reasoning": ma_gen["reasoning"],
                            "agent_outputs": ma_gen["agent_outputs"],
                            "num_text": len(ma_result.text_chunks), "num_img": len(ma_result.image_chunks),
                            "text_types": [c.chunk_type for c in ma_result.text_chunks],
                            "image_types": [c.chunk_type for c in ma_result.image_chunks],
                            "text_chunks": [_chunk_to_dict(c) for c in ma_result.text_chunks],
                            "image_chunks": [_chunk_to_dict(c) for c in ma_result.image_chunks],
                        }
            except Exception as e2:
                print(f"  ✗ Retry also failed: {type(e2).__name__}: {e2}")
                print(f"    Skipping question {i}, saving partial results...")
                entry["modes"]["_error"] = str(e2)

        results.append(entry)

        # ── Incremental save (lưu sau mỗi câu) ──
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved → {Path(output_path)}")
    _print_summary(results)

    if use_judge:
        print_judge_summary(results)

    return results


def _print_summary(results: list[dict]):
    """In bảng so sánh tóm tắt."""
    MODES = ["text_only", "text_table", "full_multimodal", "multi_agent"]
    SEP   = "─" * 70

    print(f"\n\n{'='*70}")
    print("KẾT QUẢ SO SÁNH (4 Modes)")
    print(f"{'='*70}")

    for r in results:
        print(f"\n[{r['id']}] {r['question']}")
        print(f"  Category: {r['category']} | Query type: {r['query_type']}")
        if r.get("expected_answer_hint"):
            print(f"  Expected: {r['expected_answer_hint'][:100]}")
        print(SEP)
        for mode in MODES:
            m = r["modes"].get(mode, {})
            txt = ", ".join(m.get("text_types", []))
            img = ", ".join(m.get("image_types", []))
            img_icon = " 📷" if m.get("has_image") else ""
            conf     = f" conf={m['confidence']:.2f}" if "confidence" in m else ""
            answer_pre = (m.get("answer") or "")[:200]
            print(f"  {mode:<20} text=[{txt}] img=[{img}]{img_icon}{conf}")
            print(f"    → {answer_pre}")
        print(SEP)


def _chunk_to_dict(c) -> dict:
    return {
        "chunk_id":       c.chunk_id,
        "chunk_type":     c.chunk_type,
        "score":          round(c.score, 6),
        "page":           c.page,
        "doc":            c.doc,
        "has_chart":      c.has_chart,
        "img_path":       c.img_path,
        "content_preview": c.content[:300],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="So sánh 4 chế độ retrieval + generation."
    )
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/comparison.json")
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    parser.add_argument("--judge", action="store_true",
                        help="Bật LLM-as-a-Judge chấm điểm tự động")
    parser.add_argument("--judge_model", type=str,
                        default="openai/gpt-4o-mini",
                        help="Model dùng làm judge")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Bỏ qua việc chạy 3 mode baselines (text_only, text_table, full_multimodal)")
    parser.add_argument("--skip_multi_agent", action="store_true",
                        help="Bỏ qua việc chạy mode multi_agent")
    parser.add_argument("--judge_target", type=str, choices=["all", "baselines", "multi_agent"],
                        default="all", help="Chỉ định mode nào được chấm điểm")
    parser.add_argument("--resume", action="store_true",
                        help="Đọc file output cũ để ghi đè/thêm kết quả vào thay vì chạy lại từ đầu")
    args = parser.parse_args()
    
    run_comparison(
        args.questions, args.output,
        args.text_top_k, args.image_top_k,
        use_judge=args.judge,
        judge_model=args.judge_model,
        skip_baselines=args.skip_baselines,
        skip_multi_agent=args.skip_multi_agent,
        judge_target=args.judge_target,
        resume=args.resume
    )
