# src/eval/modes.py

"""
Ba chế độ retrieval để so sánh trong thí nghiệm ablation.

    TextOnlyRetriever       — baseline: chỉ text chunks (không table, không image)
    TextTableRetriever      — text + table chunks (không image)
    FullMultimodalRetriever — text + table + image (dual pipeline đầy đủ)

Cả 3 dùng chung DualRetriever, chỉ khác nhau ở loại chunk đưa vào context.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.retriever import DualRetriever, RetrievedChunk, RetrievalResult


class TextOnlyRetriever:
    """
    Baseline — chỉ dùng text chunks (loại bỏ table và image).
    """

    MODE = "text_only"

    def __init__(self, base: DualRetriever):
        self._r = base

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        # Chỉ text pipeline, KHÔNG image pipeline
        result = self._r.retrieve(query, text_top_k=top_k * 2, image_top_k=0)

        # Lọc chỉ giữ text chunks (loại bỏ table)
        text_only = [c for c in result.text_chunks if c.chunk_type == "text"][:top_k]

        return RetrievalResult(
            text_chunks=text_only,
            image_chunks=[],
        )


class TextTableRetriever:
    """
    Text + Table — thêm bảng số liệu nhưng không có image.
    """

    MODE = "text_table"

    def __init__(self, base: DualRetriever):
        self._r = base

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        # Text pipeline bao gồm cả text + table, KHÔNG image
        result = self._r.retrieve(query, text_top_k=top_k, image_top_k=0)

        return RetrievalResult(
            text_chunks=result.text_chunks,
            image_chunks=[],
        )


class FullMultimodalRetriever:
    """
    Full multimodal — Dual pipeline đầy đủ (text + table + image).
    """

    MODE = "full_multimodal"

    def __init__(self, base: DualRetriever):
        self._r = base

    def retrieve(
        self,
        query: str,
        text_top_k: int = 5,
        image_top_k: int = 3,
    ) -> RetrievalResult:
        return self._r.retrieve(
            query,
            text_top_k=text_top_k,
            image_top_k=image_top_k,
        )
