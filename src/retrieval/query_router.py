# src/retrieval/query_router.py

"""
Query Router — phân loại intent của query.

Output: "factual" | "visual" | "comparative"

Logic: keyword matching tiếng Việt (không cần LLM → nhanh, không tốn API).
Priority: visual > comparative > factual (fallback).

Ví dụ:
    "biểu đồ lãi suất"       → visual
    "so sánh GDP các quý"     → comparative
    "tỷ lệ tăng trưởng 2025"  → factual
"""

import re

# ──────────────────────────────────────────────
# Keyword lists
# ──────────────────────────────────────────────

VISUAL_KEYWORDS = [
    # biểu đồ / hình ảnh
    "biểu đồ", "đồ thị", "hình", "chart", "graph", "figure",
    # diễn biến trực quan
    "diễn biến", "xu hướng", "dịch chuyển", "trực quan",
    "biến động", "dao động",
    # câu hỏi quan sát
    "nhìn vào", "cho thấy", "theo đồ thị", "theo biểu đồ",
    "tăng giảm thế nào", "biến động thế nào",
]

COMPARATIVE_KEYWORDS = [
    # so sánh rõ ràng
    "so sánh", "so với", "đối chiếu", "khác nhau", "giống nhau",
    # cùng kỳ / period comparison
    "cùng kỳ", "svck", "yoy", "year-over-year", "qoq", "mom",
    "năm ngoái", "năm trước", "quý",
    # ranking / top
    "top", "xếp hạng", "dẫn đầu", "đứng đầu", "cao nhất", "thấp nhất",
    # nhiều đối tượng
    "các ngành", "các lĩnh vực", "nhóm ngành", "phân ngành",
]


# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────

def classify_query(query: str) -> str:
    """
    Phân loại query intent.

    Args:
        query: câu hỏi từ người dùng (tiếng Việt hoặc mixed)

    Returns:
        "visual" | "comparative" | "factual"
    """
    q = query.lower()

    # Priority 1: visual (cần ảnh/biểu đồ)
    if _match_any(q, VISUAL_KEYWORDS):
        return "visual"

    # Priority 2: comparative (cần bảng/so sánh)
    if _match_any(q, COMPARATIVE_KEYWORDS):
        return "comparative"

    # Fallback: factual
    return "factual"


def _match_any(text: str, keywords: list[str]) -> bool:
    """Kiểm tra text có chứa bất kỳ keyword nào không."""
    for kw in keywords:
        # Word boundary-aware: tránh match partial (vd "top" trong "topology")
        pattern = re.escape(kw)
        if re.search(pattern, text):
            return True
    return False
