# src/eval/ablation_agents.py

"""
Ablation: Đóng góp của từng agent trong 3-agent pipeline.

5 biến thể so sánh:
  full            — Text + Image + Sum (đầy đủ, 3 LLM calls)
  no_text         — bỏ TextAgent
  no_image        — bỏ ImageAgent
  no_sum          — bỏ SumAgent (ghép thô)
  single_llm      — 1 LLM call (full_multimodal baseline, không chia agent)

Usage:
    python src/eval/ablation_agents.py --questions src/eval/question_test.json --output results/eval_ablation.json --judge
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
from src.agents.agents import (
    TextAgent, ImageAgent, SumAgent,
    AgentOutput,
)
from src.eval.generator import AnswerGenerator
from src.eval.judge import LLMJudge, print_judge_summary

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _empty_output(agent_name: str) -> AgentOutput:
    return AgentOutput(
        agent=agent_name, analysis="", confidence=0.0,
        sources=[], has_data=False,
    )


ALL_VARIANTS = ["full", "no_text", "no_image", "no_sum", "single_llm"]

MAX_RETRIES = 3
RETRY_WAITS = [10, 30, 60]   # seconds


def _retry_call(fn, *args, label="API call", **kwargs):
    """Gọi fn với retry + exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
            print(f"\n  ⚠ {label} failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"    Retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ✗ All {MAX_RETRIES} attempts failed, skipping.")
                raise


def run(questions_path: str, output_path: str,
        text_top_k: int = 5, image_top_k: int = 3,
        use_judge: bool = False, judge_model: str = "openai/gpt-4o-mini",
        only_full: bool = False,
        variants: list[str] | None = None):

    google_key     = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY")
    if not openrouter_key:
        raise EnvironmentError("Thiếu OPENROUTER_API_KEY")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # ── Resume: load kết quả cũ nếu có ──
    out = Path(output_path)
    done_ids: set[str] = set()
    results: list[dict] = []
    if out.exists():
        with open(out, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"  ↻ Resume: đã có {len(done_ids)} câu, skip những câu đã chạy")

    # ── Init ──
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key, cfg=CFG,
    )

    client = OpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE_URL)
    text_model   = CFG["agents"]["llm_model"]
    vision_model = CFG["agents"]["vision_model"]

    text_agent     = TextAgent(client, text_model)
    image_agent    = ImageAgent(client, vision_model)
    sum_agent      = SumAgent(client, text_model)

    generator = AnswerGenerator(
        api_key=openrouter_key,
        text_model=text_model,
        vision_model=vision_model,
    )

    if variants:
        VARIANTS = [v for v in variants if v in ALL_VARIANTS]
        if not VARIANTS:
            raise ValueError(f"No valid variants. Choose from: {ALL_VARIANTS}")
    elif only_full:
        VARIANTS = ["full"]
    else:
        VARIANTS = list(ALL_VARIANTS)
    print(f"Variants: {VARIANTS}")

    # ── LLM Judge (nếu bật) ──
    judge = None
    if use_judge:
        print(f"LLM-as-a-Judge ENABLED (model={judge_model})")
        judge = LLMJudge(
            api_key=openrouter_key,
            model=judge_model,
        )

    total = len(questions)

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q["id"]

        # Skip nếu đã chạy rồi
        if qid in done_ids:
            print(f"\n  [{i}/{total}] {qid}: SKIP (đã có)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {qid}: {question[:60]}")

        # Retrieve 1 lần (có retry)
        try:
            result = _retry_call(
                retriever.retrieve, question,
                text_top_k=text_top_k, image_top_k=image_top_k,
                label=f"Retrieve {qid}",
            )
        except Exception:
            print(f"  ✗ Skip {qid} — retrieve failed")
            continue

        text_chunks  = result.text_chunks
        image_chunks = result.image_chunks
        print(f"  T={len(text_chunks)} I={len(image_chunks)}")

        # ── Chạy full pipeline 1 lần (dùng lại cho các variant) ──
        try:
            # Cross-context: tóm tắt captions từ image chunks cho TextAgent
            image_hints = ""
            if image_chunks:
                hints = [f"[{i+1}] {c.caption}" for i, c in enumerate(image_chunks) if c.caption]
                if hints:
                    image_hints = "\n".join(hints)

            text_out = _retry_call(
                text_agent.analyze, question, text_chunks,
                image_hints=image_hints,
                label=f"TextAgent {qid}",
            )
            print(f"  Text: conf={text_out.confidence:.2f}")

            image_out = _retry_call(
                image_agent.analyze, question, image_chunks,
                label=f"ImageAgent {qid}",
            )
            print(f"  Image: conf={image_out.confidence:.2f}")
        except Exception:
            print(f"  ✗ Skip {qid} — agent call failed")
            continue

        # Raw chunks cho SumAgent kiểm chứng
        all_chunks = list(text_chunks) + list(image_chunks)

        # ── Build variants ──
        entry = {
            "id": qid,
            "question": question,
            "category": q.get("category", ""),
            "expected_answer_hint": q.get("expected_answer_hint", ""),
            "timestamp": datetime.now().isoformat(),
            "modes": {},
        }

        for variant in VARIANTS:
            print(f"  → [{variant}]", end=" ", flush=True)

            if variant == "full":
                gen = sum_agent.synthesize(question, text_out, image_out,
                                          raw_chunks=all_chunks)

            elif variant == "no_text":
                gen = sum_agent.synthesize(question, _empty_output("text"), image_out,
                                          raw_chunks=all_chunks)

            elif variant == "no_image":
                gen = sum_agent.synthesize(question, text_out, _empty_output("image"),
                                          raw_chunks=all_chunks)

            elif variant == "no_sum":
                # Ghép thô không qua SumAgent
                parts = []
                for out in [text_out, image_out]:
                    if out.has_data and out.analysis:
                        parts.append(out.analysis)
                gen = {
                    "answer": " ".join(parts) if parts else "Không đủ thông tin.",
                    "confidence": max((o.confidence for o in [text_out, image_out] if o.has_data), default=0),
                    "reasoning": "Ghép thô không qua SumAgent",
                }

            elif variant == "single_llm":
                gen_single = generator.generate(question, result, mode="full_multimodal")
                gen = {
                    "answer": gen_single["answer"],
                    "confidence": 0.0,
                    "reasoning": "Single LLM (1 call duy nhất)",
                }

            answer = gen.get("answer", "")[:120]
            conf = gen.get("confidence", 0)
            print(f"conf={conf:.2f} → {answer}...")

            entry["modes"][variant] = {
                "answer": gen.get("answer", ""),
                "confidence": gen.get("confidence", 0),
                "reasoning": gen.get("reasoning", ""),
            }

        # ── Judge scoring (nếu bật) ──
        expected_hint = q.get("expected_answer_hint", "")
        if judge and expected_hint:
            # Build context string từ retrieved chunks để Judge đánh giá Faithfulness
            context_parts = []
            for c in text_chunks:
                context_parts.append(f"[{c.chunk_type}] {c.content[:500]}")
            for c in image_chunks:
                cap = c.caption[:300] if c.caption else c.content[:300]
                context_parts.append(f"[image] {cap}")
            context_str = "\n---\n".join(context_parts)

            print(f"  → Judging {len(VARIANTS)} variants...")
            for variant in VARIANTS:
                m = entry["modes"].get(variant, {})
                answer_text = m.get("answer", "")
                print(f"    [{variant}]", end=" ", flush=True)
                try:
                    score = _retry_call(
                        judge.evaluate,
                        question, answer_text, expected_hint,
                        context=context_str,
                        label=f"Judge {qid}/{variant}",
                    )
                    m["judge"] = score
                    print(f"→ {score['total']}/20")
                except Exception:
                    m["judge"] = {"correctness": 0, "completeness": 0,
                                 "reasoning": 0, "faithfulness": 0, "total": 0}
                    print(f"→ FAILED (scored 0)")

        results.append(entry)

        # ── Incremental save (lưu sau mỗi câu) ──
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results → {out}")

    if use_judge:
        print_judge_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation: 3-Agent Contribution")
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/ablation_agents.json")
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    parser.add_argument("--judge", action="store_true",
                        help="Bật LLM-as-a-Judge chấm điểm tự động")
    parser.add_argument("--judge_model", type=str,
                        default="openai/gpt-4o-mini",
                        help="Model dùng làm judge")
    parser.add_argument("--only_full", action="store_true",
                        help="Chỉ đánh giá biến thể full (không chạy ablation)")
    parser.add_argument("--variants", type=str, default="",
                        help="Danh sách variants cách nhau bằng dấu phẩy, VD: full,single_llm")
    args = parser.parse_args()

    variant_list = [v.strip() for v in args.variants.split(",") if v.strip()] or None

    run(
        args.questions, args.output,
        args.text_top_k, args.image_top_k,
        use_judge=args.judge,
        judge_model=args.judge_model,
        only_full=args.only_full,
        variants=variant_list,
    )
