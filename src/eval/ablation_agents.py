# src/eval/ablation_agents.py

"""
Ablation: Đóng góp của từng agent trong 4-agent pipeline.

5 biến thể — bỏ từng agent so với full:
  full            — Critical + Text + Image + Sum (đầy đủ, 4 LLM calls)
  no_critical     — bỏ CriticalAgent (Text/Image không có guidance)
  no_text         — bỏ TextAgent
  no_image        — bỏ ImageAgent
  no_sum          — bỏ SumAgent (ghép thô)
  single_llm      — 1 LLM call (full_multimodal baseline)

Usage:
    conda run -n base python src/eval/ablation_agents.py
    conda run -n base python src/eval/ablation_agents.py --judge
    conda run -n base python src/eval/ablation_agents.py --judge --judge_model google/gemini-2.5-flash
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG
from src.retrieval.retriever import DualRetriever
from src.agents.agents import (
    CriticalAgent, TextAgent, ImageAgent, SumAgent,
    AgentOutput, _format_chunks,
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


def run(questions_path: str, output_path: str,
        text_top_k: int = 5, image_top_k: int = 3,
        use_judge: bool = False, judge_model: str = "google/gemini-2.5-flash"):

    google_key     = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY")
    if not openrouter_key:
        raise EnvironmentError("Thiếu OPENROUTER_API_KEY")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # ── Init ──
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key, cfg=CFG,
    )

    client = OpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE_URL)
    text_model   = CFG["agents"]["llm_model"]
    vision_model = CFG["agents"]["vision_model"]

    critical_agent = CriticalAgent(client, text_model)
    text_agent     = TextAgent(client, text_model)
    image_agent    = ImageAgent(client, vision_model)
    sum_agent      = SumAgent(client, text_model)

    generator = AnswerGenerator(
        api_key=openrouter_key,
        text_model=text_model,
        vision_model=vision_model,
    )

    VARIANTS = ["full", "no_critical", "no_text", "no_image", "no_sum", "single_llm"]

    # ── LLM Judge (nếu bật) ──
    judge = None
    if use_judge:
        print(f"LLM-as-a-Judge ENABLED (model={judge_model})")
        judge = LLMJudge(
            api_key=openrouter_key,
            model=judge_model,
        )

    results = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q["id"]

        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {qid}: {question[:60]}")

        # Retrieve 1 lần
        result = retriever.retrieve(
            question, text_top_k=text_top_k, image_top_k=image_top_k,
        )
        text_chunks  = result.text_chunks
        image_chunks = result.image_chunks
        print(f"  T={len(text_chunks)} I={len(image_chunks)}")

        # ── Chạy full pipeline 1 lần ──
        text_context  = _format_chunks(text_chunks)
        image_context = _format_chunks(image_chunks)

        # Critical
        critical_info = critical_agent.extract(
            question=question,
            text_context=text_context,
            image_context=image_context,
        )
        print(f"  Critical: text='{critical_info['text'][:40]}' img='{critical_info['image'][:40]}'")

        # Text + Image (guided)
        text_out = text_agent.analyze(question, text_chunks, critical_info["text"])
        print(f"  Text: conf={text_out.confidence:.2f}")

        image_out = image_agent.analyze(question, image_chunks, critical_info["image"])
        print(f"  Image: conf={image_out.confidence:.2f}")

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
                gen = sum_agent.synthesize(question, text_out, image_out)

            elif variant == "no_critical":
                # Text/Image không có guidance
                text_plain  = text_agent.analyze(question, text_chunks, critical_info="")
                image_plain = image_agent.analyze(question, image_chunks, critical_info="")
                gen = sum_agent.synthesize(question, text_plain, image_plain)

            elif variant == "no_text":
                gen = sum_agent.synthesize(question, _empty_output("text"), image_out)

            elif variant == "no_image":
                gen = sum_agent.synthesize(question, text_out, _empty_output("image"))

            elif variant == "no_sum":
                # Ghép thô
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
                    "reasoning": "Single LLM",
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
            print(f"  → Judging {len(VARIANTS)} variants...")
            for variant in VARIANTS:
                m = entry["modes"].get(variant, {})
                answer_text = m.get("answer", "")
                print(f"    [{variant}]", end=" ", flush=True)
                score = judge.evaluate(question, answer_text, expected_hint)
                m["judge"] = score
                print(f"→ {score['total']}/20")

        results.append(entry)

    # ── Save ──
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results → {out}")

    if use_judge:
        print_judge_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation: 4-Agent Contribution")
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/ablation_agents.json")
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    parser.add_argument("--judge", action="store_true",
                        help="Bật LLM-as-a-Judge chấm điểm tự động")
    parser.add_argument("--judge_model", type=str,
                        default="google/gemini-2.5-flash",
                        help="Model dùng làm judge")
    args = parser.parse_args()
    run(
        args.questions, args.output,
        args.text_top_k, args.image_top_k,
        use_judge=args.judge,
        judge_model=args.judge_model,
    )
