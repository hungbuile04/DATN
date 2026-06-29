# src/eval/generator.py

"""
Sinh câu trả lời từ retrieved chunks dùng LLM / Vision LLM.

Nhận RetrievalResult (text_chunks + image_chunks riêng biệt):
    text_only / text_table  → chỉ text context → text LLM
    full_multimodal         → text context + ảnh gốc → vision LLM
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI
from src.retrieval.retriever import RetrievedChunk, RetrievalResult

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = (
    "Bạn là trợ lý phân tích báo cáo tài chính chuyên nghiệp.\n"
    "Trả lời dựa HOÀN TOÀN vào context được cung cấp bên dưới.\n"
    "ƯU TIÊN text context. Biểu đồ/ảnh chỉ dùng để BỔ SUNG — "
    "nếu ảnh không liên quan đến câu hỏi, hãy BỎ QUA ảnh và trả lời từ text.\n\n"
    "QUAN TRỌNG:\n"
    "- Nếu câu hỏi chứa số liệu hoặc thông tin SAI so với context, "
    "hãy CHỈ RA lỗi sai và đưa ra con số ĐÚNG từ tài liệu.\n"
    "- Nếu câu hỏi không nhắc tên công ty/mã cổ phiếu cụ thể, "
    "hãy XÁC ĐỊNH công ty nào phù hợp nhất dựa trên đặc điểm mô tả "
    "trong câu hỏi và thông tin trong context.\n\n"
    "TRÍCH DẪN NGUỒN — BẮT BUỘC:\n"
    "- Mỗi khi nêu số liệu hoặc nhận định, gắn nhãn nguồn [X, Trang Y] vào sau.\n"
    "  (với X là số thứ tự đoạn văn trong context, Y là số trang ghi trong header)\n"
    "  Ví dụ: \"NIM của HDB đạt 4,5% [1, Trang 8] trong khi CTG đạt 3,2% [3, Trang 5].\"\n"
    "- Cuối câu trả lời, LUÔN thêm phần:\n"
    "  **Nguồn tham khảo:**\n"
    "  - [X] Tên tài liệu | Trang Y\n\n"
    "Nếu context không đủ để trả lời, hãy nói rõ: "
    "'Thông tin không có trong tài liệu được cung cấp.'\n"
    "Trả lời bằng tiếng Việt, chi tiết, đầy đủ, chính xác."
)


class AnswerGenerator:
    """
    Sinh câu trả lời cho một câu hỏi.

    Args:
        api_key:      OpenRouter API key
        text_model:   model cho text-only generation
        vision_model: model cho multimodal generation (hỗ trợ ảnh)
    """

    def __init__(self, api_key: str, text_model: str, vision_model: str):
        self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self.text_model   = text_model
        self.vision_model = vision_model

    def generate(
        self,
        question: str,
        result: RetrievalResult,
        mode: str,
    ) -> dict:
        """
        Sinh câu trả lời từ RetrievalResult.

        Args:
            question: câu hỏi gốc
            result:   RetrievalResult (text_chunks + image_chunks)
            mode:     "text_only" | "text_table" | "full_multimodal"

        Returns:
            {
                "answer":    str,
                "model":     str,
                "has_image": bool,
                "num_images":int,
            }
        """
        if mode == "full_multimodal" and result.image_chunks:
            return self._generate_vision(question, result)
        else:
            return self._generate_text(question, result.text_chunks)

    def _generate_text(self, question: str, chunks: list[RetrievedChunk]) -> dict:
        """LLM thuần text."""
        context = _format_context(chunks)
        user_msg = f"Context:\n{context}\n\nCâu hỏi: {question}"

        resp = self.client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        return {
            "answer":     resp.choices[0].message.content,
            "model":      self.text_model,
            "has_image":  False,
            "num_images": 0,
        }

    def _generate_vision(self, question: str, result: RetrievalResult) -> dict:
        """Vision LLM — text context + ảnh gốc từ image pipeline."""
        text_context = _format_context(result.text_chunks)
        image_context = _format_context(result.image_chunks)

        # Build multimodal message
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"Text context:\n{text_context}\n\n"
                    f"Biểu đồ/Hình ảnh (caption kèm nguồn):\n{image_context}\n\n"
                    f"Câu hỏi: {question}"
                ),
            },
        ]

        # Attach ảnh gốc từ image chunks
        num_images = 0
        seen: set[str] = set()
        for c in result.image_chunks:
            if c.img_path and c.img_path not in seen:
                b64 = _encode_image(c.img_path)
                if b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                    num_images += 1
                    seen.add(c.img_path)
            if num_images >= 3:
                break

        resp = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        return {
            "answer":     resp.choices[0].message.content,
            "model":      self.vision_model,
            "has_image":  num_images > 0,
            "num_images": num_images,
        }


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format danh sách chunk thành chuỗi context cho LLM, kèm nhãn để trích dẫn."""
    if not chunks:
        return "(không có dữ liệu)"
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.chunk_type.upper()} | {c.doc} | Trang {c.page}"
        parts.append(f"{header}\n{c.content}")
    return "\n\n---\n\n".join(parts)


def _encode_image(img_path: str) -> str | None:
    """Encode ảnh sang base64."""
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except (FileNotFoundError, OSError):
        return None
