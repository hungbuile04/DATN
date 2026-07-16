# src/eval/judge_manual.py

"""
Chấm điểm thủ công bằng LLM Judge.

Cách dùng:
    # CLI trực tiếp (chấm 1 câu)
    python src/eval/judge_manual.py \
        -q "Câu hỏi?" \
        -a "Câu trả lời" \
        -e "Đáp án mong đợi"

    # Tương tác (nhập nhiều câu)
    python src/eval/judge_manual.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.eval.judge import LLMJudge


def judge_single(judge, question, answer, expected):
    score = judge.evaluate(question, answer, expected, context="")
    score["faithfulness"] = 5
    score["total"] = (
        score["correctness"] + score["completeness"]
        + score["relevance"] + score["faithfulness"]
    )
    print(f"\n  {'─'*40}")
    print(f"  Correctness  (đúng):      {score['correctness']}/5")
    print(f"  Completeness (đầy đủ):    {score['completeness']}/5")
    print(f"  Relevance    (liên quan):  {score['relevance']}/5")
    print(f"  Faithfulness (trung thực): {score['faithfulness']}/5")
    print(f"  {'─'*40}")
    print(f"  TỔNG:                      {score['total']}/20")
    # print(f"  Giải thích: {score['explanation']}")
    return score


def read_multiline(prompt):
    print(f"{prompt} (Enter 2 lần để kết thúc):")
    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 1 and lines:
                    break
                if not lines:
                    continue
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", type=str, help="Câu hỏi")
    parser.add_argument("-a", "--answer", type=str, help="Câu trả lời")
    parser.add_argument("-e", "--expected", type=str, help="Đáp án mong đợi")
    parser.add_argument("--judge_model", default="google/gemini-2.5-flash")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("❌ Thiếu OPENROUTER_API_KEY trong .env")
        sys.exit(1)

    judge = LLMJudge(api_key=api_key, model=args.judge_model, sleep_sec=1.0)
    print(f"✓ Judge: {args.judge_model}")

    # ── Chế độ CLI trực tiếp ──
    if args.question and args.answer:
        expected = args.expected or args.answer
        print(f"\n❓ {args.question[:80]}")
        print(f"💬 {args.answer[:120]}")
        judge_single(judge, args.question, args.answer, expected)
        return

    # ── Chế độ tương tác ──
    print("Nhập câu hỏi → câu trả lời → đáp án. Gõ 'quit' để thoát.\n")
    scores = []
    count = 0
    while True:
        count += 1
        print(f"\n📝 Câu {count}")
        question = read_multiline("❓ Câu hỏi")
        if question.lower() in ("quit", "exit", "q"):
            break
        answer = read_multiline("💬 Câu trả lời")
        if answer.lower() in ("quit", "exit", "q"):
            break
        expected = read_multiline("✅ Đáp án mong đợi")
        if expected.lower() in ("quit", "exit", "q"):
            break
        print("⏳ Đang chấm...")
        scores.append(judge_single(judge, question, answer, expected))

    if scores:
        avg = sum(s["total"] for s in scores) / len(scores)
        print(f"\nTỔNG KẾT: {len(scores)} câu | TB: {avg:.1f}/20")


if __name__ == "__main__":
    main()
