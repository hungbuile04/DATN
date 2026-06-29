"""
Benchmark latency & tài nguyên hệ thống RAG.

Đo thời gian từng giai đoạn:
  1. Embedding query
  2. Dense retrieval (ChromaDB)
  3. Reranker (nếu bật)
  4. Multi-Agent reasoning (Text + Image + Sum)
  5. Single LLM reasoning

Sử dụng:
    cd source_code_2
    python benchmark_latency.py
    python benchmark_latency.py --n 15 --mode both        # 15 câu, cả 2 chế độ
    python benchmark_latency.py --n 15 --hard              # bộ câu hỏi khó
    python benchmark_latency.py --n 15 --hard --mode both  # khó + cả 2 chế độ
"""

import sys
import os
import json
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG
from src.retrieval.retriever import DualRetriever
from src.agents.orchestrator import MultiAgentOrchestrator
from src.eval.generator import AnswerGenerator


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def fmt(seconds: float) -> str:
    """Format thời gian."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def percentile(data: list, p: int) -> float:
    """Tính percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def get_disk_usage(path: str) -> str:
    """Dung lượng thư mục."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    if total < 1024 * 1024:
        return f"{total / 1024:.1f} KB"
    return f"{total / (1024*1024):.1f} MB"


# ──────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────

def run_benchmark(n_questions: int = 20, mode: str = "multi", use_hard: bool = False):
    """
    mode: "multi" | "single" | "both"
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")

    if not openrouter_key:
        raise EnvironmentError("Thiếu OPENROUTER_API_KEY")

    # ── Khởi tạo ──
    print("=" * 60)
    print("🔧 Khởi tạo hệ thống...")
    print("=" * 60)

    t0 = time.time()
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )
    t_init = time.time() - t0
    print(f"  Retriever init: {fmt(t_init)}")

    orchestrator = None
    generator = None

    if mode in ("multi", "both"):
        orchestrator = MultiAgentOrchestrator(
            api_key=openrouter_key,
            text_model=CFG["agents"]["llm_model"],
            vision_model=CFG["agents"]["vision_model"],
        )
        print(f"  Multi-Agent Orchestrator: OK")

    if mode in ("single", "both"):
        generator = AnswerGenerator(
            api_key=openrouter_key,
            text_model=CFG["agents"]["llm_model"],
            vision_model=CFG["agents"]["vision_model"],
        )
        print(f"  Single LLM Generator: OK")

    # ── Load câu hỏi ──
    if use_hard:
        q_path = Path("src/eval/questions_hard.json")
        dataset_name = "hard (100 câu)"
    else:
        q_path = Path("src/eval/questions.json")
        dataset_name = "standard (200 câu)"
    print(f"\n📂 Dataset: {q_path.name} — {dataset_name}")
    with open(q_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    # Lấy đa dạng category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for q in all_questions:
        by_cat[q["category"]].append(q)

    selected = []
    per_cat = max(1, n_questions // len(by_cat))
    for cat, qs in by_cat.items():
        selected.extend(qs[:per_cat])
    selected = selected[:n_questions]

    print(f"\n📊 Benchmark: {len(selected)} câu hỏi")
    cats = {}
    for q in selected:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    for c, n in cats.items():
        print(f"  {c}: {n}")

    # ── Đo từng giai đoạn ──
    results = []
    text_top_k = CFG["retrieval"]["text_top_k"]
    image_top_k = CFG["retrieval"]["image_top_k"]

    print(f"\n{'='*60}")
    print(f"⏱️  Bắt đầu benchmark (top_k: text={text_top_k}, image={image_top_k})")
    print(f"{'='*60}\n")

    for i, q in enumerate(selected):
        qid = q["id"]
        question = q["question"]
        print(f"[{i+1}/{len(selected)}] {qid}: {question[:50]}...")

        record = {"id": qid, "category": q["category"]}

        # ── Giai đoạn 1: Retrieval ──
        t0 = time.time()
        result = retriever.retrieve(question, text_top_k=text_top_k, image_top_k=image_top_k)
        t_retrieval = time.time() - t0
        record["retrieval"] = t_retrieval
        record["n_text"] = len(result.text_chunks)
        record["n_image"] = len(result.image_chunks)
        print(f"  Retrieval: {fmt(t_retrieval)} ({len(result.text_chunks)}T + {len(result.image_chunks)}I)")

        # ── Giai đoạn 2a: Multi-Agent ──
        if orchestrator and mode in ("multi", "both"):
            t0 = time.time()
            try:
                final = orchestrator.run(question, result)
                t_multi = time.time() - t0
                record["multi_agent"] = t_multi
                print(f"  Multi-Agent: {fmt(t_multi)}")
            except Exception as e:
                t_multi = time.time() - t0
                record["multi_agent"] = t_multi
                record["multi_error"] = str(e)
                print(f"  Multi-Agent: ERROR ({fmt(t_multi)}) - {e}")

        # ── Giai đoạn 2b: Single LLM ──
        if generator and mode in ("single", "both"):
            gen_mode = "full_multimodal" if result.image_chunks else "text_table"
            t0 = time.time()
            try:
                gen_result = generator.generate(question, result, mode=gen_mode)
                t_single = time.time() - t0
                record["single_llm"] = t_single
                print(f"  Single LLM: {fmt(t_single)}")
            except Exception as e:
                t_single = time.time() - t0
                record["single_llm"] = t_single
                record["single_error"] = str(e)
                print(f"  Single LLM: ERROR ({fmt(t_single)}) - {e}")

        # ── Tổng ──
        t_total = record["retrieval"] + record.get("multi_agent", record.get("single_llm", 0))
        record["total"] = t_total
        print(f"  → Total: {fmt(t_total)}")

        results.append(record)

        # Rate limiting
        time.sleep(0.5)

    # ── Thống kê tổng hợp ──
    print(f"\n{'='*60}")
    print(f"📈 KẾT QUẢ BENCHMARK")
    print(f"{'='*60}\n")

    def stats_block(name: str, values: list):
        if not values:
            return
        avg = statistics.mean(values)
        med = statistics.median(values)
        p95 = percentile(values, 95)
        mn = min(values)
        mx = max(values)
        print(f"  {name:25s}  TB={fmt(avg):>8s}  Med={fmt(med):>8s}  P95={fmt(p95):>8s}  Min={fmt(mn):>8s}  Max={fmt(mx):>8s}")

    retrieval_times = [r["retrieval"] for r in results]
    stats_block("Retrieval (Dense Search)", retrieval_times)

    if mode in ("multi", "both"):
        multi_times = [r["multi_agent"] for r in results if "multi_agent" in r]
        stats_block("Multi-Agent (3 agents)", multi_times)
        total_multi = [r["retrieval"] + r.get("multi_agent", 0) for r in results if "multi_agent" in r]
        stats_block("Tổng E2E (Multi-Agent)", total_multi)

    if mode in ("single", "both"):
        single_times = [r["single_llm"] for r in results if "single_llm" in r]
        stats_block("Single LLM", single_times)
        total_single = [r["retrieval"] + r.get("single_llm", 0) for r in results if "single_llm" in r]
        stats_block("Tổng E2E (Single LLM)", total_single)

    # ── Tài nguyên hệ thống ──
    print(f"\n{'='*60}")
    print(f"💾 TÀI NGUYÊN HỆ THỐNG")
    print(f"{'='*60}\n")

    db_path = str(CFG["paths"]["vector_db"])
    processed_path = str(CFG["paths"]["pdf_processed"])

    text_count = retriever.text_col.count()
    image_count = retriever.image_col.count()

    print(f"  Tài liệu PDF        : {len(list(Path(CFG['paths']['pdf_raw']).glob('*.pdf')))}")
    print(f"  Mã cổ phiếu          : {len(retriever._known_tickers)}")
    print(f"  Text/Table chunks    : {text_count}")
    print(f"  Image chunks         : {image_count}")
    print(f"  Tổng chunks          : {text_count + image_count}")
    print(f"  Vector DB size       : {get_disk_usage(db_path)}")
    print(f"  Processed data size  : {get_disk_usage(processed_path)}")
    print(f"  Embedding dim        : {CFG['embedding']['embed_dim']}")
    print(f"  Embedding model      : {CFG['embedding']['model']}")
    print(f"  LLM model            : {CFG['agents']['llm_model']}")
    print(f"  API calls / query    : {'4-5 (Multi-Agent)' if mode != 'single' else '1-2 (Single LLM)'}")

    # ── Lưu kết quả ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": q_path.name,
        "config": {
            "n_questions": len(selected),
            "mode": mode,
            "text_top_k": text_top_k,
            "image_top_k": image_top_k,
            "llm_model": CFG["agents"]["llm_model"],
            "embed_model": CFG["embedding"]["model"],
        },
        "summary": {
            "retrieval_avg": statistics.mean(retrieval_times),
            "retrieval_p95": percentile(retrieval_times, 95),
        },
        "resources": {
            "text_chunks": text_count,
            "image_chunks": image_count,
            "total_chunks": text_count + image_count,
            "vector_db_size": get_disk_usage(db_path),
            "processed_size": get_disk_usage(processed_path),
        },
        "details": results,
    }

    if mode in ("multi", "both") and multi_times:
        output["summary"]["multi_agent_avg"] = statistics.mean(multi_times)
        output["summary"]["multi_agent_p95"] = percentile(multi_times, 95)
        output["summary"]["total_e2e_multi_avg"] = statistics.mean(total_multi)

    if mode in ("single", "both") and single_times:
        output["summary"]["single_llm_avg"] = statistics.mean(single_times)
        output["summary"]["single_llm_p95"] = percentile(single_times, 95)
        output["summary"]["total_e2e_single_avg"] = statistics.mean(total_single)

    suffix = "hard" if use_hard else "standard"
    out_path = Path(f"results/benchmark_latency_{suffix}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📁 Kết quả đã lưu: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark latency hệ thống RAG")
    parser.add_argument("--n", type=int, default=20, help="Số câu hỏi (default: 20)")
    parser.add_argument("--mode", choices=["multi", "single", "both"], default="multi",
                        help="Chế độ reasoning: multi (Multi-Agent), single (Single LLM), both")
    parser.add_argument("--hard", action="store_true",
                        help="Dùng bộ câu hỏi khó (questions_hard.json)")
    args = parser.parse_args()
    run_benchmark(n_questions=args.n, mode=args.mode, use_hard=args.hard)
