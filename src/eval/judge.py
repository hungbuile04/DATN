# src/eval/judge.py

"""
LLM-as-a-Judge — chấm điểm câu trả lời tự động.

Dùng một LLM mạnh (judge) để đánh giá câu trả lời theo 4 tiêu chí:
    1. Correctness  — Đúng so với expected_answer_hint (0-5)
    2. Completeness — Trả lời đầy đủ các khía cạnh câu hỏi (0-5)
    3. Relevance    — Bám sát câu hỏi, không lan man (0-5)
    4. Faithfulness — Trung thành với context, không hallucinate (0-5)

Trả về dict:
    {
        "correctness":  int,
        "completeness": int,
        "relevance":    int,
        "faithfulness": int,
        "total":        int (0-20),
        "explanation":  str,
    }

Sử dụng:
    from src.eval.judge import LLMJudge
    judge = LLMJudge(api_key=..., model=...)
    score = judge.evaluate(question, answer, expected_hint)
"""

import json
import time
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

JUDGE_SYSTEM_PROMPT = """\
Bạn là giám khảo chuyên đánh giá câu trả lời trong lĩnh vực phân tích báo cáo tài chính.

Cho: Câu hỏi, Câu trả lời cần chấm, và Gợi ý đáp án đúng (expected_answer_hint).

Chấm theo 4 tiêu chí (mỗi tiêu chí 0-5 điểm):

1. **Correctness** (Độ chính xác): Câu trả lời có khớp với expected_answer_hint không?
   - 5: Hoàn toàn chính xác, đúng con số/dữ kiện
   - 4: Gần đúng, sai sót nhỏ không ảnh hưởng kết luận
   - 3: Đúng hướng nhưng thiếu chi tiết hoặc con số sai nhẹ
   - 2: Đúng một phần, thiếu nhiều hoặc có sai sót đáng kể
   - 1: Sai hầu hết nhưng có chút liên quan
   - 0: Hoàn toàn sai hoặc không trả lời

2. **Completeness** (Đầy đủ): Câu trả lời bao phủ đủ các khía cạnh?
   - 5: Đầy đủ mọi khía cạnh câu hỏi yêu cầu
   - 3: Trả lời được phần chính nhưng thiếu chi tiết bổ sung
   - 1: Chỉ trả lời được một phần nhỏ
   - 0: Không trả lời được gì

3. **Relevance** (Liên quan): Câu trả lời bám sát câu hỏi?
   - 5: Hoàn toàn bám sát, không lan man
   - 3: Bám sát nhưng có phần thừa
   - 1: Lan man nhiều, ít liên quan
   - 0: Hoàn toàn không liên quan

4. **Faithfulness** (Trung thực): Câu trả lời có vẻ dựa trên dữ liệu thực hay bịa đặt?
   - 5: Rõ ràng dựa trên dữ liệu, có dẫn chứng cụ thể
   - 3: Có vẻ dựa trên dữ liệu nhưng không rõ nguồn
   - 1: Nhiều phần có vẻ suy đoán/bịa
   - 0: Rõ ràng hallucinate

QUAN TRỌNG:
- expected_answer_hint là GỢI Ý, không phải đáp án duy nhất đúng.
  Câu trả lời có thể đúng theo cách khác miễn là hợp lý.
- Nếu câu trả lời nêu thêm thông tin hữu ích ngoài hint, KHÔNG trừ điểm.
- Nếu câu trả lời nói "Không có trong tài liệu" mà hint có đáp án → Correctness = 0.

Trả lời ĐÚNG JSON, KHÔNG thêm gì khác:
{
  "correctness": 0-5,
  "completeness": 0-5,
  "relevance": 0-5,
  "faithfulness": 0-5,
  "explanation": "Giải thích ngắn gọn (1-2 câu) lý do chấm điểm"
}"""


class LLMJudge:
    """
    LLM-as-a-Judge scorer.

    Args:
        api_key:    OpenRouter API key
        model:      Model dùng làm judge (nên dùng model mạnh)
        sleep_sec:  Delay giữa các lần gọi API (tránh rate limit)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.5-flash",
        sleep_sec: float = 1.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self.model = model
        self.sleep_sec = sleep_sec

    def evaluate(
        self,
        question: str,
        answer: str,
        expected_hint: str,
    ) -> dict:
        """
        Chấm điểm 1 câu trả lời.

        Returns:
            {
                "correctness": int,  "completeness": int,
                "relevance": int,    "faithfulness": int,
                "total": int,        "explanation": str,
            }
        """
        if not answer or not answer.strip():
            return self._empty_score("Không có câu trả lời")

        if not expected_hint or not expected_hint.strip():
            return self._empty_score("Không có expected_answer_hint để chấm")

        user_msg = (
            f"Câu hỏi:\n{question}\n\n"
            f"Câu trả lời cần chấm:\n{answer}\n\n"
            f"Gợi ý đáp án đúng (expected_answer_hint):\n{expected_hint}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_json(raw)

            score = {
                "correctness":  _clamp(parsed.get("correctness", 0), 0, 5),
                "completeness": _clamp(parsed.get("completeness", 0), 0, 5),
                "relevance":    _clamp(parsed.get("relevance", 0), 0, 5),
                "faithfulness": _clamp(parsed.get("faithfulness", 0), 0, 5),
                "explanation":  parsed.get("explanation", ""),
            }
            score["total"] = (
                score["correctness"] + score["completeness"]
                + score["relevance"] + score["faithfulness"]
            )
            return score

        except Exception as e:
            return self._empty_score(f"Judge error: {e}")

        finally:
            time.sleep(self.sleep_sec)

    def evaluate_batch(
        self,
        items: list[dict],
    ) -> list[dict]:
        """
        Chấm nhiều câu lần lượt.

        Args:
            items: list of {"question", "answer", "expected_hint"}

        Returns:
            list of score dicts
        """
        scores = []
        for i, item in enumerate(items, 1):
            print(f"    Judge [{i}/{len(items)}]", end=" ", flush=True)
            s = self.evaluate(
                item["question"],
                item["answer"],
                item["expected_hint"],
            )
            print(f"→ {s['total']}/20")
            scores.append(s)
        return scores

    @staticmethod
    def _empty_score(reason: str) -> dict:
        return {
            "correctness": 0, "completeness": 0,
            "relevance": 0,   "faithfulness": 0,
            "total": 0,       "explanation": reason,
        }


def print_judge_summary(results: list[dict], mode_key: str = "modes"):
    """
    In bảng tổng kết điểm judge cho tất cả modes.

    Args:
        results: list entries, mỗi entry có [mode_key][mode_name]["judge"]
        mode_key: key chứa dict các modes (mặc định "modes")
    """
    if not results:
        return

    # Thu thập tất cả mode names
    all_modes = set()
    for r in results:
        for k, v in r.get(mode_key, {}).items():
            if k != "_error" and isinstance(v, dict):
                all_modes.add(k)
    all_modes = sorted(all_modes)

    if not all_modes:
        return

    CRITERIA = ["correctness", "completeness", "relevance", "faithfulness", "total"]

    print(f"\n\n{'='*80}")
    print("BẢNG ĐIỂM LLM-AS-A-JUDGE")
    print(f"{'='*80}")

    # ── Per-question scores ──
    for r in results:
        qid = r.get("id", "?")
        question = r.get("question", "")[:60]
        print(f"\n[{qid}] {question}")
        print(f"  Expected: {r.get('expected_answer_hint', '')[:80]}")
        print(f"  {'Mode':<20} {'Corr':>5} {'Comp':>5} {'Relv':>5} {'Faith':>5} {'Total':>6}")
        print(f"  {'─'*52}")
        for mode in all_modes:
            m = r.get(mode_key, {}).get(mode, {})
            if not isinstance(m, dict):
                continue
            j = m.get("judge", {})
            if j:
                print(
                    f"  {mode:<20}"
                    f" {j.get('correctness','-'):>5}"
                    f" {j.get('completeness','-'):>5}"
                    f" {j.get('relevance','-'):>5}"
                    f" {j.get('faithfulness','-'):>5}"
                    f" {j.get('total','-'):>6}"
                )

    # ── Aggregate scores ──
    print(f"\n\n{'─'*80}")
    print("TỔNG KẾT TRUNG BÌNH")
    print(f"{'─'*80}")
    print(f"  {'Mode':<20} {'Corr':>5} {'Comp':>5} {'Relv':>5} {'Faith':>5} {'Total':>6}  {'N':>3}")
    print(f"  {'─'*58}")

    for mode in all_modes:
        scores = []
        for r in results:
            m = r.get(mode_key, {}).get(mode, {})
            if not isinstance(m, dict):
                continue
            j = m.get("judge")
            if j and isinstance(j.get("total"), (int, float)):
                scores.append(j)

        if not scores:
            continue

        n = len(scores)
        avgs = {}
        for c in CRITERIA:
            vals = [s[c] for s in scores if isinstance(s.get(c), (int, float))]
            avgs[c] = sum(vals) / len(vals) if vals else 0

        print(
            f"  {mode:<20}"
            f" {avgs['correctness']:>5.2f}"
            f" {avgs['completeness']:>5.2f}"
            f" {avgs['relevance']:>5.2f}"
            f" {avgs['faithfulness']:>5.2f}"
            f" {avgs['total']:>6.2f}"
            f"  {n:>3}"
        )

    # ── Win rate ──
    if len(all_modes) >= 2:
        print(f"\n{'─'*80}")
        print("WIN RATE (mode nào thắng theo total score)")
        print(f"{'─'*80}")

        wins = {m: 0 for m in all_modes}
        ties = 0
        valid = 0

        for r in results:
            mode_scores = {}
            for mode in all_modes:
                m = r.get(mode_key, {}).get(mode, {})
                if not isinstance(m, dict):
                    continue
                j = m.get("judge")
                if j and isinstance(j.get("total"), (int, float)):
                    mode_scores[mode] = j["total"]

            if len(mode_scores) < 2:
                continue
            valid += 1
            max_score = max(mode_scores.values())
            winners = [m for m, s in mode_scores.items() if s == max_score]
            if len(winners) == 1:
                wins[winners[0]] += 1
            else:
                ties += 1

        if valid > 0:
            for mode in all_modes:
                pct = wins[mode] / valid * 100
                print(f"  {mode:<20} {wins[mode]:>3}/{valid} ({pct:>5.1f}%)")
            print(f"  {'Ties':<20} {ties:>3}/{valid} ({ties/valid*100:>5.1f}%)")


# ── Helpers ──

def _clamp(val, lo, hi):
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return lo


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
