# src/eval/analysis_stats.py

"""
Phân tích bổ sung cho đồ án:
  1. Statistical significance (paired t-test) cho các so sánh chính
  2. Cost analysis (USD/1000 câu hỏi)
  3. Error analysis tự động (phân loại câu trả lời điểm thấp)

Usage:
    python src/eval/analysis_stats.py --input results/comparison.json
    python src/eval/analysis_stats.py --input results/eval_naive_rag.json --naive
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ══════════════════════════════════════════════
# 1. Statistical Significance (Paired t-test)
# ══════════════════════════════════════════════

def run_significance_tests(results: list[dict]):
    """Paired t-test giữa các mode."""
    from scipy import stats

    print(f"\n{'═'*70}")
    print("1. STATISTICAL SIGNIFICANCE (Paired t-test)")
    print(f"{'═'*70}")
    print("  H0: Không có sự khác biệt giữa 2 mode")
    print("  H1: Có sự khác biệt có ý nghĩa thống kê")
    print("  Ngưỡng: α = 0.05\n")

    # Thu thập judge scores theo mode
    mode_scores = defaultdict(list)
    for r in results:
        modes = r.get("modes", {})
        for mode_name, m_data in modes.items():
            if m_data.get("judge", {}).get("total") is not None:
                mode_scores[mode_name].append(m_data["judge"]["total"])

    if not mode_scores:
        print("  ⚠ Không tìm thấy judge scores. Chạy runner.py --judge trước.")
        return

    # Các cặp so sánh quan trọng
    comparisons = [
        ("text_only",       "multi_agent",     "Text-only vs Multi-Agent"),
        ("text_table",      "multi_agent",     "Text+Table vs Multi-Agent"),
        ("full_multimodal", "multi_agent",     "Full Multimodal vs Multi-Agent"),
        ("text_only",       "full_multimodal", "Text-only vs Full Multimodal"),
    ]

    print(f"  {'So sánh':<40} {'N':>4} {'Mean A':>8} {'Mean B':>8} "
          f"{'t-stat':>8} {'p-value':>10} {'Kết luận':<20}")
    print(f"  {'─'*110}")

    for mode_a, mode_b, label in comparisons:
        scores_a = mode_scores.get(mode_a, [])
        scores_b = mode_scores.get(mode_b, [])

        if not scores_a or not scores_b:
            print(f"  {label:<40} — thiếu dữ liệu")
            continue

        # Ghép cặp theo index (cùng câu hỏi)
        n = min(len(scores_a), len(scores_b))
        a = scores_a[:n]
        b = scores_b[:n]

        mean_a = sum(a) / n
        mean_b = sum(b) / n

        t_stat, p_value = stats.ttest_rel(a, b)

        if p_value < 0.001:
            conclusion = "*** p<0.001"
        elif p_value < 0.01:
            conclusion = "**  p<0.01"
        elif p_value < 0.05:
            conclusion = "*   p<0.05"
        else:
            conclusion = "n.s."

        print(f"  {label:<40} {n:>4} {mean_a:>8.2f} {mean_b:>8.2f} "
              f"{t_stat:>8.3f} {p_value:>10.6f} {conclusion:<20}")

    # Cohen's d (effect size) cho so sánh chính
    print(f"\n  Effect Size (Cohen's d) cho so sánh chính:")
    for mode_a, mode_b, label in comparisons:
        scores_a = mode_scores.get(mode_a, [])
        scores_b = mode_scores.get(mode_b, [])
        if not scores_a or not scores_b:
            continue
        n = min(len(scores_a), len(scores_b))
        a, b = scores_a[:n], scores_b[:n]
        diff = [b[i] - a[i] for i in range(n)]
        mean_d = sum(diff) / n
        std_d = (sum((x - mean_d)**2 for x in diff) / (n - 1)) ** 0.5
        d = mean_d / std_d if std_d > 0 else 0
        size = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
        print(f"    {label:<40} d = {d:+.3f} ({size})")


# ══════════════════════════════════════════════
# 2. Cost Analysis
# ══════════════════════════════════════════════

def run_cost_analysis():
    """Tính chi phí vận hành USD/1000 câu hỏi."""
    print(f"\n\n{'═'*70}")
    print("2. COST ANALYSIS (USD / 1000 câu hỏi)")
    print(f"{'═'*70}")

    # Giá API (tham khảo tháng 6/2026, có thể cần cập nhật)
    prices = {
        "Gemini Embedding 2": {
            "price_per_1M_tokens": 0.00,   # Free tier / rất rẻ
            "tokens_per_query": 50,         # query embedding
            "calls_per_question": 1,
            "note": "Free tier hoặc $0.00/1M tokens",
        },
        "Gemini 2.5 Flash (Vision - Caption)": {
            "input_per_1M": 0.15,
            "output_per_1M": 0.60,
            "input_tokens": 1500,   # ảnh + prompt
            "output_tokens": 200,   # caption 3-5 câu
            "calls_per_question": 0,  # caption chỉ chạy 1 lần khi indexing
            "note": "Chỉ chạy khi indexing, không tính vào runtime",
        },
    }

    # ── Naive RAG: 1 LLM call/câu ──
    naive_calls = {
        "Embedding query":        {"calls": 1, "input": 50, "output": 0},
        "LLM Generation":         {"calls": 1, "input": 3000, "output": 800},
    }

    # ── Full System: 1 embed + 4 agent calls/câu ──
    full_calls = {
        "Embedding query":        {"calls": 1, "input": 50, "output": 0},
        "CriticalAgent":          {"calls": 1, "input": 500, "output": 300},
        "TextAgent":              {"calls": 1, "input": 4000, "output": 800},
        "ImageAgent (Vision)":    {"calls": 1, "input": 3000, "output": 600},
        "SumAgent":               {"calls": 1, "input": 2500, "output": 1000},
    }

    # Giá LLM (OpenRouter, ước tính)
    llm_input_price  = 0.15   # $/1M input tokens (Gemini 2.5 Flash)
    llm_output_price = 0.60   # $/1M output tokens
    vision_input_price = 0.15
    vision_output_price = 0.60

    def calc_cost(calls_dict, n=1000):
        total = 0.0
        for name, c in calls_dict.items():
            inp = c["calls"] * c["input"] * n
            out = c["calls"] * c["output"] * n
            if "Vision" in name:
                cost = (inp / 1e6) * vision_input_price + (out / 1e6) * vision_output_price
            else:
                cost = (inp / 1e6) * llm_input_price + (out / 1e6) * llm_output_price
            total += cost
        return total

    naive_cost = calc_cost(naive_calls)
    full_cost  = calc_cost(full_calls)

    print(f"\n  Giả định giá API (Gemini 2.5 Flash qua OpenRouter):")
    print(f"    Input:  ${llm_input_price}/1M tokens")
    print(f"    Output: ${llm_output_price}/1M tokens\n")

    print(f"  {'Mode':<25} {'LLM calls/câu':>15} {'USD/1000 câu':>15}")
    print(f"  {'─'*58}")
    print(f"  {'Naive RAG':<25} {'1':>15} {'$'+f'{naive_cost:.3f}':>15}")
    print(f"  {'Full System (4 agents)':<25} {'4':>15} {'$'+f'{full_cost:.3f}':>15}")
    print(f"  {'─'*58}")
    print(f"  {'Chi phí tăng thêm':<25} {'':>15} {'$'+f'{full_cost-naive_cost:.3f}':>15}")
    print(f"  {'Tỷ lệ tăng':<25} {'':>15} {f'{full_cost/naive_cost:.1f}x':>15}")

    print(f"\n  💡 Nhận xét:")
    print(f"     Chi phí Full System ≈ ${full_cost:.2f}/1000 câu")
    print(f"     ≈ ${full_cost*30:.2f}/tháng nếu trung bình 1000 câu/ngày")
    print(f"     → Hoàn toàn khả thi cho triển khai thực tế")


# ══════════════════════════════════════════════
# 3. Error Analysis
# ══════════════════════════════════════════════

def run_error_analysis(results: list[dict], threshold: float = 12.0):
    """Phân tích câu trả lời điểm thấp."""
    print(f"\n\n{'═'*70}")
    print(f"3. ERROR ANALYSIS (câu có Judge Total ≤ {threshold}/20)")
    print(f"{'═'*70}")

    # Tìm câu điểm thấp ở mode multi_agent (hoặc mode tốt nhất)
    low_score_items = []
    for r in results:
        modes = r.get("modes", {})
        ma = modes.get("multi_agent", {})
        judge = ma.get("judge", {})
        total = judge.get("total", 999)

        if total <= threshold and total != 999:
            low_score_items.append({
                "id": r["id"],
                "question": r["question"],
                "category": r.get("category", ""),
                "requires": r.get("requires", []),
                "expected": r.get("expected_answer_hint", ""),
                "answer": ma.get("answer", "")[:200],
                "judge": judge,
                "n_text": ma.get("num_text", 0),
                "n_img": ma.get("num_img", 0),
                "text_types": ma.get("text_types", []),
            })

    if not low_score_items:
        print("  ✅ Không có câu nào dưới ngưỡng — hệ thống hoạt động tốt!")
        return

    print(f"  Tìm thấy {len(low_score_items)} câu điểm thấp\n")

    # Phân loại nguyên nhân lỗi
    error_types = defaultdict(list)

    for item in low_score_items:
        j = item["judge"]
        reasons = []

        # Lỗi truy xuất: correctness thấp + có doc trong requires
        if j.get("correctness", 5) <= 2:
            reasons.append("retrieval_miss")

        # Lỗi suy luận: correctness OK nhưng completeness thấp
        if j.get("completeness", 5) <= 2 and j.get("correctness", 0) >= 3:
            reasons.append("reasoning_incomplete")

        # Lỗi trung thực: faithfulness thấp (hallucination)
        if j.get("faithfulness", 5) <= 2:
            reasons.append("hallucination")

        # Lỗi relevance: trả lời lạc đề
        if j.get("relevance", 5) <= 2:
            reasons.append("off_topic")

        # Nếu tất cả tiêu chí đều trung bình → lỗi chung
        if not reasons:
            reasons.append("general_weak")

        for reason in reasons:
            error_types[reason].append(item)

    # In thống kê
    error_labels = {
        "retrieval_miss":       "Lỗi truy xuất (retriever không tìm đúng chunk)",
        "reasoning_incomplete": "Lỗi suy luận (thiếu thông tin trong câu trả lời)",
        "hallucination":        "Lỗi trung thực (hallucination / bịa số liệu)",
        "off_topic":            "Lỗi lạc đề (trả lời không đúng câu hỏi)",
        "general_weak":         "Yếu tổng thể (không có lỗi rõ ràng)",
    }

    print(f"  {'Loại lỗi':<55} {'Số câu':>8}")
    print(f"  {'─'*65}")
    for err_type in ["retrieval_miss", "reasoning_incomplete",
                      "hallucination", "off_topic", "general_weak"]:
        items = error_types.get(err_type, [])
        if items:
            print(f"  {error_labels[err_type]:<55} {len(items):>8}")

    # In chi tiết top 10
    low_score_items.sort(key=lambda x: x["judge"]["total"])
    print(f"\n  TOP {min(10, len(low_score_items))} CÂU ĐIỂM THẤP NHẤT:")
    print(f"  {'─'*70}")

    for item in low_score_items[:10]:
        j = item["judge"]
        print(f"\n  [{item['id']}] ({item['category']}) Total={j['total']}/20")
        print(f"    C={j.get('correctness','?')} O={j.get('completeness','?')} "
              f"R={j.get('relevance','?')} F={j.get('faithfulness','?')}")
        print(f"    Q: {item['question'][:80]}...")
        print(f"    Expected: {item['expected'][:80]}...")
        print(f"    Answer: {item['answer'][:100]}...")
        print(f"    Chunks: text={item['n_text']} img={item['n_img']} "
              f"types={item['text_types']}")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phân tích thống kê + chi phí + lỗi"
    )
    parser.add_argument("--input", default="results/comparison.json",
                        help="File kết quả từ runner.py")
    parser.add_argument("--threshold", type=float, default=12.0,
                        help="Ngưỡng điểm Judge để coi là 'lỗi'")
    parser.add_argument("--skip_stats", action="store_true",
                        help="Bỏ qua t-test (nếu chưa cài scipy)")
    args = parser.parse_args()

    # Load results
    with open(args.input, encoding="utf-8") as f:
        results = json.load(f)

    # runner.py output là list[dict]
    if isinstance(results, dict):
        results = results.get("detail", [results])

    print(f"Loaded {len(results)} entries from {args.input}")

    # 1. Statistical tests
    if not args.skip_stats:
        try:
            run_significance_tests(results)
        except ImportError:
            print("\n⚠ Cần cài scipy: pip install scipy")
            print("  Hoặc chạy với --skip_stats để bỏ qua")

    # 2. Cost analysis
    run_cost_analysis()

    # 3. Error analysis
    run_error_analysis(results, threshold=args.threshold)
