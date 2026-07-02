# src/retrieval/reranker.py

"""
Gemini Flash Reranker — Cross-encoder reranking bằng LLM.

Nhận danh sách candidates từ Dense Search, dùng Gemini Flash
để đánh giá lại mức độ liên quan (query, chunk) → sắp xếp lại.

Tích hợp vào DualRetriever:
    Dense Search (top_k × 4 candidates)
    → Reranker (LLM scoring)
    → Top-k (kết quả chính xác hơn)

Fallback: nếu LLM API lỗi → trả về thứ tự cũ (dense score).
"""

import json
import re


from openai import OpenAI

# ──────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────

RERANK_PROMPT_TEMPLATE = """\
Bạn là chuyên gia phân tích tài chính. Đánh giá mức độ liên quan của mỗi đoạn văn bản với câu hỏi.

**Câu hỏi:** __QUERY__

**Các đoạn văn bản:**
__CANDIDATES__

**Tiêu chí chấm điểm (0.0 – 1.0):**
- 0.9-1.0: Chứa số liệu/dữ kiện trực tiếp trả lời câu hỏi (đúng mã cổ phiếu, đúng chỉ số, đúng kỳ)
- 0.6-0.8: Chứa thông tin liên quan trực tiếp nhưng chưa đủ để trả lời
- 0.3-0.5: Liên quan gián tiếp (cùng công ty nhưng khác chỉ số)
- 0.0-0.2: Không liên quan hoặc sai mã cổ phiếu

Trả về JSON array. Ví dụ: [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.3}]
"""


# ──────────────────────────────────────────────
# GeminiReranker
# ──────────────────────────────────────────────

class GeminiReranker:
    """
    Reranker sử dụng Gemini Flash qua OpenRouter API.

    Flow:
        1. Nhận list (chunk_id, content, dense_score)
        2. Truncate content → 400 chars/chunk (tiết kiệm tokens)
        3. Gửi prompt cho Gemini Flash → nhận relevance scores
        4. Kết hợp: final_score = 0.4 × dense_score + 0.6 × llm_score
        5. Sort lại theo final_score

    Fallback: API lỗi → giữ thứ tự cũ (dense score).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.5-flash",
        max_chars_per_chunk: int = 500,
        dense_weight: float = 0.7,
        llm_weight: float = 0.3,
    ):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
        self.max_chars = max_chars_per_chunk
        self.dense_weight = dense_weight
        self.llm_weight = llm_weight

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
    ) -> list[tuple[str, float]]:
        """
        Rerank candidates bằng Gemini Flash.

        Args:
            query:      câu hỏi người dùng
            candidates: list of (chunk_id, content, dense_score)

        Returns:
            list of (chunk_id, final_score) — đã sắp xếp giảm dần
        """
        if not candidates:
            return []

        # Nếu ≤ 2 candidates, không cần rerank (tốn API call vô ích)
        if len(candidates) <= 2:
            return [(cid, score) for cid, _, score in candidates]

        try:
            llm_scores = self._call_llm(query, candidates)
            return self._merge_scores(candidates, llm_scores)
        except Exception as e:
            print(f"  ⚠ Reranker fallback (LLM error): {e}")
            # Fallback: giữ thứ tự cũ
            return [(cid, score) for cid, _, score in candidates]

    def _call_llm(
        self, query: str, candidates: list[tuple[str, str, float]]
    ) -> list[float]:
        """Gọi Gemini Flash để scoring relevance."""
        # Build candidates text (truncated)
        parts = []
        for i, (chunk_id, content, _) in enumerate(candidates):
            snippet = content[:self.max_chars].replace("\n", " ").strip()
            if len(content) > self.max_chars:
                snippet += "..."
            parts.append(f"[{i}] {snippet}")

        candidates_text = "\n\n".join(parts)
        prompt = RERANK_PROMPT_TEMPLATE.replace("__QUERY__", query).replace("__CANDIDATES__", candidates_text)

        # LLM call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )

        raw = response.choices[0].message.content.strip()
        return self._parse_scores(raw, len(candidates))

    def _parse_scores(self, raw: str, n: int) -> list[float]:
        """Parse JSON scores từ LLM output. Robust: handle nhiều format edge cases và JSON bị cắt."""
        # Bỏ markdown code fences nếu có
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        # Phương pháp 1: Parse JSON chuẩn
        scores = [0.5] * n
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        idx = item.get("id", item.get("index", item.get("idx", -1)))
                        sc = item.get("score", item.get("relevance", 0.5))
                        if isinstance(idx, int) and 0 <= idx < n:
                            scores[idx] = max(0.0, min(1.0, float(sc)))
                return scores
        except json.JSONDecodeError:
            pass

        # Phương pháp 2: Regex từng object — xử lý JSON bị cắt ngang
        pattern = r'"id"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*([\d.]+)'
        matches = re.findall(pattern, cleaned)
        if matches:
            for idx_str, score_str in matches:
                idx = int(idx_str)
                sc = float(score_str)
                if 0 <= idx < n:
                    scores[idx] = max(0.0, min(1.0, sc))
            return scores

        # Phương pháp 3: Không parse được → giữ default 0.5
        print(f"  ⚠ Cannot parse reranker output, using default scores")
        return scores

    def _merge_scores(
        self,
        candidates: list[tuple[str, str, float]],
        llm_scores: list[float],
    ) -> list[tuple[str, float]]:
        """Kết hợp dense_score + llm_score → final_score, sort giảm dần."""
        merged = []
        for i, (chunk_id, _, dense_score) in enumerate(candidates):
            llm_score = llm_scores[i] if i < len(llm_scores) else 0.5
            final = self.dense_weight * dense_score + self.llm_weight * llm_score
            merged.append((chunk_id, final))

        return sorted(merged, key=lambda x: x[1], reverse=True)
