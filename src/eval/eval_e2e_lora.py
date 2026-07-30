# src/eval/eval_e2e_lora.py

"""
End-to-End Evaluation: So sánh chất lượng câu trả lời cuối cùng
khi dùng Gemini Reranker vs LoRA Reranker.

Pipeline: Retriever → Reranker → Agents → Answer → LLM-as-Judge

So sánh 2 (hoặc 3) cấu hình:
    A) no_rerank:     Pipeline không rerank
    B) gemini_rerank: Pipeline + Gemini Flash Reranker (API)
    C) lora_rerank:   Pipeline + LoRA Cross-Encoder Reranker (local)

Đánh giá bằng LLMJudge (4 tiêu chí: correctness, completeness,
relevance, faithfulness — mỗi tiêu chí 0-5 → total 0-20).

Usage:
    python -m src.eval.eval_e2e_lora --questions src/eval/questions_hard.json --n 20
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
from src.agents.agents import build_context
from src.eval.judge import LLMJudge, print_judge_summary
from src.eval.runner import ask_with_retriever


def run_e2e_eval(
    questions: list[dict],
    n: int = 20,
    skip_gemini: bool = False,
) -> list[dict]:
    """
    Chạy full pipeline cho mỗi câu hỏi với các cấu hình reranker khác nhau,
    sau đó chấm điểm bằng LLM-as-Judge.

    Args:
        questions:   list eval questions
        n:           số câu hỏi tối đa (để tiết kiệm API)
        skip_gemini: bỏ qua Gemini reranker

    Returns:
        list of result dicts
    """
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")
    if not openrouter_key:
        raise EnvironmentError("Thiếu OPENROUTER_API_KEY trong .env")

    chroma_path = str(CFG["paths"]["vector_db"])

    # ── Init Retrievers ──
    print("Initializing retrievers...")

    # Dense only (no rerank)
    retriever_dense = DualRetriever(
        chroma_path=chroma_path,
        api_key=google_key,
        cfg=CFG,
        openrouter_key=openrouter_key,
        use_reranker=False,
    )

    # Gemini reranker
    retriever_gemini = None
    if not skip_gemini:
        retriever_gemini = DualRetriever(
            chroma_path=chroma_path,
            api_key=google_key,
            cfg=CFG,
            openrouter_key=openrouter_key,
            use_reranker=True,
            reranker_type="gemini",
        )

    # LoRA reranker
    retriever_lora = DualRetriever(
        chroma_path=chroma_path,
        api_key=google_key,
        cfg=CFG,
        use_reranker=True,
        reranker_type="lora",
    )

    # ── Init Judge ──
    judge = LLMJudge(
        api_key=openrouter_key,
        model="openai/gpt-4o-mini",
        sleep_sec=1.0,
    )

    # ── Run ──
    questions = questions[:n]
    results = []

    print(f"\n{'=' * 70}")
    print(f"E2E Evaluation: {len(questions)} questions")
    print(f"{'=' * 70}")

    configs = {"no_rerank": retriever_dense, "lora_rerank": retriever_lora}
    if retriever_gemini:
        configs["gemini_rerank"] = retriever_gemini

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q["id"]
        expected = q.get("expected_answer_hint", "")
        doc = q.get("doc", "")

        if not doc or not expected:
            continue

        print(f"\n  [{i}/{len(questions)}] {qid}: {question[:60]}...")

        entry = {
            "id": qid,
            "question": question,
            "expected_answer_hint": expected,
            "doc": doc,
            "category": q.get("category", ""),
            "modes": {},
        }

        for mode_name, retriever in configs.items():
            try:
                t0 = time.perf_counter()
                answer, context = ask_with_retriever(retriever, question)
                latency = (time.perf_counter() - t0) * 1000

                # Judge
                judge_score = judge.evaluate(
                    question=question,
                    answer=answer,
                    expected_hint=expected,
                    context=context,
                )

                entry["modes"][mode_name] = {
                    "answer": answer[:500],
                    "judge": judge_score,
                    "latency_ms": round(latency, 1),
                }

                print(
                    f"    {mode_name:<18} "
                    f"total={judge_score['total']:>2}/20  "
                    f"({judge_score['correctness']}/{judge_score['completeness']}/"
                    f"{judge_score['relevance']}/{judge_score['faithfulness']})  "
                    f"{latency:.0f}ms"
                )

            except Exception as e:
                entry["modes"][mode_name] = {"_error": str(e)}
                print(f"    {mode_name:<18} ERROR: {e}")

            time.sleep(1.0)  # Rate limit

        results.append(entry)

    # ── Print Summary ──
    print_judge_summary(results)

    return results


def ask_with_retriever(retriever: DualRetriever, question: str) -> tuple[str, str]:
    """
    Chạy full pipeline: Retrieve → Build Context → Agent → Answer.

    Returns:
        (answer_text, context_text)
    """
    from openai import OpenAI

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = CFG.get("agents", {}).get("llm_model", "google/gemini-2.5-flash")

    # Retrieve
    result = retriever.retrieve(
        query=question,
        text_top_k=CFG["retrieval"]["text_top_k"],
        image_top_k=CFG["retrieval"]["image_top_k"],
    )

    # Build context
    all_chunks = result.text_chunks + result.image_chunks
    context_parts = []
    for chunk in all_chunks:
        context_parts.append(
            f"[{chunk.chunk_type}|{chunk.doc}|p{chunk.page}] {chunk.content[:800]}"
        )
    context_text = "\n---\n".join(context_parts)

    # Generate answer via LLM
    client = OpenAI(
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
    )

    system_prompt = (
        "Bạn là chuyên gia phân tích tài chính. Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. "
        "Nếu thông tin không có trong ngữ cảnh, hãy nói rõ. Trả lời chính xác, có dẫn chứng số liệu."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Ngữ cảnh:\n{context_text}\n\nCâu hỏi: {question}"},
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content or ""
    return answer, context_text


def run(questions_path: str, output_path: str, n: int = 20,
        skip_gemini: bool = False):
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    results = run_e2e_eval(questions, n=n, skip_gemini=skip_gemini)

    all_output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "e2e_lora_vs_gemini_reranker",
        "config": {"n_questions": len(results)},
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E2E: So sánh Gemini vs LoRA Reranker (LLM-as-Judge)"
    )
    parser.add_argument("--questions", default="src/eval/questions_hard.json")
    parser.add_argument("--output", default="results/eval_e2e_lora.json")
    parser.add_argument("--n", type=int, default=20,
                        help="Số câu hỏi tối đa (tiết kiệm API)")
    parser.add_argument("--skip_gemini", action="store_true",
                        help="Bỏ qua Gemini reranker")
    args = parser.parse_args()
    run(args.questions, args.output, args.n, args.skip_gemini)
