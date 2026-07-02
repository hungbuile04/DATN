# src/eval/runner_dual_encode.py

"""
Ablation: Single Encode vs Dual Encode (MDocAgent-style).

So sánh hiệu quả retrieval khi:
  - single_encode: text & image pipeline dùng chung 1 query vector
  - dual_encode:   image pipeline dùng vector riêng (prefix "biểu đồ hình ảnh...")

Chạy trên 2 modes nhạy cảm nhất với image retrieval:
  - full_multimodal (single vision LLM)
  - multi_agent     (4 agents)

Usage:
    python src/eval/runner_dual_encode.py \
        --questions src/eval/questions.json \
        --output    results/ablation_dual_encode.json \
        --text_top_k 5 --image_top_k 3
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
from src.retrieval.query_router import classify_query
from src.eval.generator import AnswerGenerator
from src.agents.orchestrator import MultiAgentOrchestrator


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


def _run_mode(
    mode_label: str,
    question: str,
    retriever: DualRetriever,
    generator: AnswerGenerator,
    orchestrator: MultiAgentOrchestrator,
    text_top_k: int,
    image_top_k: int,
    dual_encode: bool,
) -> dict:
    """Chạy 1 câu hỏi trên 1 mode (full_multimodal hoặc multi_agent)."""
    encode_label = "dual" if dual_encode else "single"

    result = retriever.retrieve(
        question,
        text_top_k=text_top_k,
        image_top_k=image_top_k,
        dual_encode=dual_encode,
    )

    text_types  = [c.chunk_type for c in result.text_chunks]
    image_types = [c.chunk_type for c in result.image_chunks]

    if mode_label == "full_multimodal":
        gen = generator.generate(question, result, mode="full_multimodal")
        output = {
            "answer":     gen["answer"],
            "model":      gen["model"],
            "has_image":  gen["has_image"],
            "num_images": gen["num_images"],
        }
    else:  # multi_agent
        gen = orchestrator.run(question, result)
        output = {
            "answer":        gen["answer"],
            "model":         gen["model"],
            "has_image":     gen["has_image"],
            "num_images":    gen["num_images"],
            "confidence":    gen["confidence"],
            "reasoning":     gen["reasoning"],
            "agent_outputs": gen["agent_outputs"],
        }

    output.update({
        "encode":       encode_label,
        "num_text":     len(result.text_chunks),
        "num_img":      len(result.image_chunks),
        "text_types":   text_types,
        "image_types":  image_types,
        "text_chunks":  [_chunk_to_dict(c) for c in result.text_chunks],
        "image_chunks": [_chunk_to_dict(c) for c in result.image_chunks],
    })

    answer_pre = (output["answer"] or "")[:120]
    img_flag = " 📷" if output["has_image"] else ""
    print(f"    [{encode_label:6}] T={output['num_text']} I={output['num_img']}{img_flag}")
    print(f"           {answer_pre}...")

    return output


def run_ablation(
    questions_path: str,
    output_path: str,
    text_top_k: int = 5,
    image_top_k: int = 3,
) -> list[dict]:
    """Chạy ablation single vs dual encode."""
    google_key     = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")
    if not openrouter_key:
        raise EnvironmentError("Thiếu OPENROUTER_API_KEY trong .env")

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {questions_path}")

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

    orchestrator = MultiAgentOrchestrator(
        api_key=openrouter_key,
        text_model=CFG["agents"]["llm_model"],
        vision_model=CFG["agents"]["vision_model"],
    )

    # ── Chạy ──
    results: list[dict] = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        question   = q["question"]
        query_type = classify_query(question)

        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {question}")
        print(f"  category={q.get('category','')} | query_type={query_type}")

        entry: dict = {
            "id":                   q.get("id", f"q{i:03d}"),
            "question":             question,
            "category":             q.get("category", ""),
            "requires":             q.get("requires", []),
            "expected_answer_hint": q.get("expected_answer_hint", ""),
            "query_type":           query_type,
            "timestamp":            datetime.now().isoformat(),
            "modes":                {},
        }

        # Chạy 4 biến thể: {full_multimodal, multi_agent} x {single, dual}
        import time
        for mode in ["full_multimodal", "multi_agent"]:
            for dual in [False, True]:
                # Throttle giữa các variants — tránh 429 quota/minute
                time.sleep(3)
                encode_label = "dual" if dual else "single"
                key = f"{mode}__{encode_label}"

                print(f"  → [{key}]")
                entry["modes"][key] = _run_mode(
                    mode_label=mode,
                    question=question,
                    retriever=retriever,
                    generator=generator,
                    orchestrator=orchestrator,
                    text_top_k=text_top_k,
                    image_top_k=image_top_k,
                    dual_encode=dual,
                )

        results.append(entry)

    # ── Save ──
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved → {out}")
    _print_summary(results)
    return results


def _print_summary(results: list[dict]):
    """Bảng so sánh nhanh."""
    KEYS = [
        "full_multimodal__single",
        "full_multimodal__dual",
        "multi_agent__single",
        "multi_agent__dual",
    ]

    print(f"\n\n{'='*80}")
    print("ABLATION: Single Encode vs Dual Encode")
    print(f"{'='*80}")

    for r in results:
        print(f"\n[{r['id']}] {r['question'][:70]}")
        hint = r.get("expected_answer_hint", "")
        if hint:
            print(f"  Expected: {hint}")
        for key in KEYS:
            m = r["modes"].get(key, {})
            ans = (m.get("answer") or "")[:150]
            n_img = m.get("num_img", 0)
            conf = f" conf={m['confidence']:.2f}" if "confidence" in m else ""
            print(f"  {key:<32} I={n_img}{conf}")
            print(f"    → {ans}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation: Single vs Dual Encode"
    )
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/ablation_dual_encode.json")
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    args = parser.parse_args()
    run_ablation(args.questions, args.output, args.text_top_k, args.image_top_k)
