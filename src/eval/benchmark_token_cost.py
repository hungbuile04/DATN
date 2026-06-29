"""
Benchmark token usage và chi phí API cho Single LLM vs Multi-Agent.

Chạy trên 12 câu hỏi khó (questions_hard.json), đo:
  - Input tokens / Output tokens cho từng agent
  - Tổng tokens mỗi chế độ
  - Chi phí ước tính theo bảng giá Gemini 2.5 Flash

Giá Gemini 2.5 Flash (OpenRouter, 2025):
  Input : $0.15 / 1M tokens  → $1.5e-7 / token
  Output: $0.60 / 1M tokens  → $6.0e-7 / token

Usage:
    python src/eval/benchmark_token_cost.py
"""

import json
import os
import sys
import time
import base64
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from config.settings import CFG
from src.retrieval.retriever import DualRetriever, RetrievalResult, RetrievedChunk

# ── Giá Gemini 2.5 Flash trên OpenRouter (USD / token) ──
PRICE_INPUT  = 0.15 / 1_000_000   # $0.15 / 1M input tokens
PRICE_OUTPUT = 0.60 / 1_000_000   # $0.60 / 1M output tokens

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

N_QUESTIONS = 12   # Chạy 12 câu khó đầu tiên (giống benchmark_latency_hard)


def cost(inp: int, out: int) -> float:
    return inp * PRICE_INPUT + out * PRICE_OUTPUT


def _encode_image(img_path: str) -> str | None:
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except (FileNotFoundError, OSError):
        return None


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(không có dữ liệu)"
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.chunk_type.upper()} | {c.doc} | Trang {c.page}"
        parts.append(f"{header}\n{c.content}")
    return "\n\n---\n\n".join(parts)


# ── System prompts (lấy từ agents.py / generator.py) ──
SYSTEM_SINGLE = (
    "Bạn là trợ lý phân tích báo cáo tài chính chuyên nghiệp.\n"
    "Trả lời dựa HOÀN TOÀN vào context được cung cấp bên dưới.\n"
    "Trả lời bằng tiếng Việt, chi tiết, đầy đủ, chính xác."
)

SYSTEM_TEXT_AGENT = (
    "Bạn là TextAgent — chuyên phân tích văn bản và bảng số liệu tài chính.\n"
    "Trả lời dựa HOÀN TOÀN vào context văn bản được cung cấp.\n"
    "Trích dẫn nguồn [X, Trang Y] sau mỗi số liệu.\n"
    "Trả lời bằng tiếng Việt, chi tiết."
)

SYSTEM_IMAGE_AGENT = (
    "Bạn là ImageAgent — chuyên phân tích biểu đồ tài chính.\n"
    "Mô tả chi tiết những gì quan sát được từ biểu đồ: xu hướng, giá trị cụ thể, đỉnh/đáy.\n"
    "Trả lời bằng tiếng Việt, chi tiết."
)

SYSTEM_SUM_AGENT = (
    "Bạn là SumAgent — tổng hợp câu trả lời từ các agent khác.\n"
    "Kết hợp thông tin từ TextAgent và ImageAgent thành câu trả lời hoàn chỉnh.\n"
    "Giữ tất cả trích dẫn nguồn. Nếu hai agent mâu thuẫn, ghi rõ sự khác biệt.\n"
    "Trả lời bằng tiếng Việt, toàn diện."
)


def run():
    google_key     = os.environ.get("GOOGLE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not google_key or not openrouter_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY hoặc OPENROUTER_API_KEY")

    client = OpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE_URL)

    model_text   = CFG["agents"]["llm_model"]
    model_vision = CFG["agents"]["vision_model"]
    text_top_k   = 6
    image_top_k  = 4

    print(f"Text model  : {model_text}")
    print(f"Vision model: {model_vision}")
    print(f"Top-k       : text={text_top_k}, image={image_top_k}")

    # Load questions
    with open("src/eval/questions_hard.json", encoding="utf-8") as f:
        all_questions = json.load(f)
    questions = all_questions[:N_QUESTIONS]
    print(f"\nRunning on {len(questions)} questions from questions_hard.json\n")

    # Retriever
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )

    results = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        qid = q.get("id", f"q{i:03d}")
        print(f"\n{'='*65}")
        print(f"[{i}/{len(questions)}] {qid}: {question[:70]}...")

        # ── Retrieve ──
        result: RetrievalResult = retriever.retrieve(
            question, text_top_k=text_top_k, image_top_k=image_top_k
        )
        text_chunks  = result.text_chunks
        image_chunks = result.image_chunks

        text_ctx  = _format_context(text_chunks)
        image_ctx = _format_context(image_chunks)

        # ═══════════════════════════
        # A) SINGLE LLM
        # ═══════════════════════════
        print("  [Single LLM]", end=" ", flush=True)

        # Build multimodal message
        content_single: list[dict] = [{
            "type": "text",
            "text": f"Text context:\n{text_ctx}\n\nBiểu đồ (caption):\n{image_ctx}\n\nCâu hỏi: {question}",
        }]
        n_imgs_single = 0
        seen: set[str] = set()
        for c in image_chunks:
            if c.img_path and c.img_path not in seen:
                b64 = _encode_image(c.img_path)
                if b64:
                    content_single.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                    n_imgs_single += 1
                    seen.add(c.img_path)
            if n_imgs_single >= 3:
                break

        resp_single = client.chat.completions.create(
            model=model_vision,
            messages=[
                {"role": "system", "content": SYSTEM_SINGLE},
                {"role": "user",   "content": content_single},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        single_inp = resp_single.usage.prompt_tokens
        single_out = resp_single.usage.completion_tokens
        single_cost = cost(single_inp, single_out)
        print(f"in={single_inp} out={single_out} cost=${single_cost:.5f}")
        time.sleep(2)

        # ═══════════════════════════
        # B) MULTI-AGENT
        # ═══════════════════════════
        # TextAgent
        print("  [TextAgent] ", end=" ", flush=True)
        resp_text = client.chat.completions.create(
            model=model_text,
            messages=[
                {"role": "system", "content": SYSTEM_TEXT_AGENT},
                {"role": "user",   "content": f"Text context:\n{text_ctx}\n\nCâu hỏi: {question}"},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        text_inp = resp_text.usage.prompt_tokens
        text_out = resp_text.usage.completion_tokens
        text_answer = resp_text.choices[0].message.content
        print(f"in={text_inp} out={text_out} cost=${cost(text_inp, text_out):.5f}")
        time.sleep(2)

        # ImageAgent
        print("  [ImageAgent]", end=" ", flush=True)
        content_img: list[dict] = [{
            "type": "text",
            "text": f"Biểu đồ (caption):\n{image_ctx}\n\nCâu hỏi: {question}",
        }]
        n_imgs_ma = 0
        seen2: set[str] = set()
        for c in image_chunks:
            if c.img_path and c.img_path not in seen2:
                b64 = _encode_image(c.img_path)
                if b64:
                    content_img.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                    n_imgs_ma += 1
                    seen2.add(c.img_path)
            if n_imgs_ma >= 3:
                break

        resp_img = client.chat.completions.create(
            model=model_vision,
            messages=[
                {"role": "system", "content": SYSTEM_IMAGE_AGENT},
                {"role": "user",   "content": content_img},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        img_inp = resp_img.usage.prompt_tokens
        img_out = resp_img.usage.completion_tokens
        img_answer = resp_img.choices[0].message.content
        print(f"in={img_inp} out={img_out} cost=${cost(img_inp, img_out):.5f}")
        time.sleep(2)

        # SumAgent
        print("  [SumAgent]  ", end=" ", flush=True)
        sum_user = (
            f"Câu hỏi: {question}\n\n"
            f"TextAgent:\n{text_answer}\n\n"
            f"ImageAgent:\n{img_answer}\n\n"
            "Hãy tổng hợp thành câu trả lời hoàn chỉnh."
        )
        resp_sum = client.chat.completions.create(
            model=model_text,
            messages=[
                {"role": "system", "content": SYSTEM_SUM_AGENT},
                {"role": "user",   "content": sum_user},
            ],
            temperature=0.1,
            max_tokens=3500,
        )
        sum_inp = resp_sum.usage.prompt_tokens
        sum_out = resp_sum.usage.completion_tokens
        print(f"in={sum_inp} out={sum_out} cost=${cost(sum_inp, sum_out):.5f}")
        time.sleep(2)

        # Tổng multi-agent
        ma_inp_total = text_inp + img_inp + sum_inp
        ma_out_total = text_out + img_out + sum_out
        ma_cost_total = cost(ma_inp_total, ma_out_total)

        print(f"  → Multi-Agent total: in={ma_inp_total} out={ma_out_total} cost=${ma_cost_total:.5f}")
        print(f"  → Single LLM total : in={single_inp} out={single_out} cost=${single_cost:.5f}")

        results.append({
            "qid": qid,
            "category": q.get("category", ""),
            "n_text_chunks": len(text_chunks),
            "n_image_chunks": len(image_chunks),
            "n_images_attached": n_imgs_single,
            "single_llm": {
                "input_tokens":  single_inp,
                "output_tokens": single_out,
                "total_tokens":  single_inp + single_out,
                "cost_usd":      round(single_cost, 6),
            },
            "multi_agent": {
                "text_agent":  {"input": text_inp,   "output": text_out},
                "image_agent": {"input": img_inp,    "output": img_out},
                "sum_agent":   {"input": sum_inp,    "output": sum_out},
                "input_tokens":  ma_inp_total,
                "output_tokens": ma_out_total,
                "total_tokens":  ma_inp_total + ma_out_total,
                "cost_usd":      round(ma_cost_total, 6),
            },
        })

    # ── Tổng kết ──
    n = len(results)
    avg_single_inp = sum(r["single_llm"]["input_tokens"]  for r in results) / n
    avg_single_out = sum(r["single_llm"]["output_tokens"] for r in results) / n
    avg_single_cost = sum(r["single_llm"]["cost_usd"]     for r in results) / n

    avg_ma_inp  = sum(r["multi_agent"]["input_tokens"]  for r in results) / n
    avg_ma_out  = sum(r["multi_agent"]["output_tokens"] for r in results) / n
    avg_ma_cost = sum(r["multi_agent"]["cost_usd"]      for r in results) / n

    # Per-agent avg
    avg_ta_inp = sum(r["multi_agent"]["text_agent"]["input"]  for r in results) / n
    avg_ta_out = sum(r["multi_agent"]["text_agent"]["output"] for r in results) / n
    avg_ia_inp = sum(r["multi_agent"]["image_agent"]["input"] for r in results) / n
    avg_ia_out = sum(r["multi_agent"]["image_agent"]["output"]for r in results) / n
    avg_sa_inp = sum(r["multi_agent"]["sum_agent"]["input"]   for r in results) / n
    avg_sa_out = sum(r["multi_agent"]["sum_agent"]["output"]  for r in results) / n

    print(f"\n\n{'='*70}")
    print(f"TỔNG KẾT TOKEN & CHI PHÍ (trung bình / câu hỏi, n={n})")
    print(f"{'='*70}")
    print(f"\n  {'Chế độ':<22} {'Input':>8} {'Output':>8} {'Total':>8}  {'Cost (USD)':>12}")
    print(f"  {'─'*65}")
    print(f"  {'Single LLM':<22} {avg_single_inp:>8.0f} {avg_single_out:>8.0f} {avg_single_inp+avg_single_out:>8.0f}  ${avg_single_cost:>11.5f}")
    print(f"  {'Multi-Agent (tổng)':<22} {avg_ma_inp:>8.0f} {avg_ma_out:>8.0f} {avg_ma_inp+avg_ma_out:>8.0f}  ${avg_ma_cost:>11.5f}")
    print(f"\n  Chi tiết Multi-Agent:")
    print(f"  {'  TextAgent':<22} {avg_ta_inp:>8.0f} {avg_ta_out:>8.0f}   ${cost(avg_ta_inp,avg_ta_out):>11.5f}")
    print(f"  {'  ImageAgent':<22} {avg_ia_inp:>8.0f} {avg_ia_out:>8.0f}   ${cost(avg_ia_inp,avg_ia_out):>11.5f}")
    print(f"  {'  SumAgent':<22} {avg_sa_inp:>8.0f} {avg_sa_out:>8.0f}   ${cost(avg_sa_inp,avg_sa_out):>11.5f}")
    print(f"\n  Tỷ lệ token Multi/Single: {(avg_ma_inp+avg_ma_out)/(avg_single_inp+avg_single_out):.1f}x")
    print(f"  Tỷ lệ cost  Multi/Single: {avg_ma_cost/avg_single_cost:.1f}x")
    print(f"\n  Giá tham chiếu (Gemini 2.5 Flash via OpenRouter):")
    print(f"    Input : $0.15 / 1M tokens")
    print(f"    Output: $0.60 / 1M tokens")

    # ── Save ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_questions": n,
            "text_top_k": text_top_k,
            "image_top_k": image_top_k,
            "model_text": model_text,
            "model_vision": model_vision,
            "price_input_per_token":  PRICE_INPUT,
            "price_output_per_token": PRICE_OUTPUT,
        },
        "summary": {
            "single_llm": {
                "avg_input_tokens":  round(avg_single_inp),
                "avg_output_tokens": round(avg_single_out),
                "avg_total_tokens":  round(avg_single_inp + avg_single_out),
                "avg_cost_usd":      round(avg_single_cost, 6),
            },
            "multi_agent": {
                "avg_input_tokens":  round(avg_ma_inp),
                "avg_output_tokens": round(avg_ma_out),
                "avg_total_tokens":  round(avg_ma_inp + avg_ma_out),
                "avg_cost_usd":      round(avg_ma_cost, 6),
                "per_agent": {
                    "text_agent":  {"avg_input": round(avg_ta_inp), "avg_output": round(avg_ta_out)},
                    "image_agent": {"avg_input": round(avg_ia_inp), "avg_output": round(avg_ia_out)},
                    "sum_agent":   {"avg_input": round(avg_sa_inp), "avg_output": round(avg_sa_out)},
                },
            },
            "ratio_tokens_multi_vs_single": round((avg_ma_inp+avg_ma_out)/(avg_single_inp+avg_single_out), 2),
            "ratio_cost_multi_vs_single":   round(avg_ma_cost/avg_single_cost, 2),
        },
        "details": results,
    }

    out_path = Path("results/benchmark_token_cost.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results → {out_path}")


if __name__ == "__main__":
    run()
