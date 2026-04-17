# src/agents/agents.py

"""
4 Agents cho Multi-Agent RAG (MDocAgent-inspired).

Luồng 3 stages:
    Stage 1: CriticalAgent (text + image + question) → critical info {text, image}
    Stage 2: TextAgent     (text+table chunks + Tc)   → text-based answer
             ImageAgent    (image chunks + Ic)         → image-based answer
    Stage 3: SumAgent      (ghép thô aT + aI → chỉnh sửa mạch lạc)

Thiết kế:
    - CriticalAgent: xác định keypoints → guide TextAgent + ImageAgent focus đúng hướng
    - TextAgent: text + table gộp (không tách TableAgent riêng)
    - ImageAgent: OCR + visual analysis, guided by critical info
    - SumAgent: editor — nhận bản nháp ghép thô, chỉnh mạch lạc, KHÔNG viết lại
"""

import base64
import json
from pathlib import Path
from dataclasses import dataclass

from openai import OpenAI
from src.retrieval.retriever import RetrievedChunk


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ──────────────────────────────────────────────
# Agent output
# ──────────────────────────────────────────────

@dataclass
class AgentOutput:
    """Output từ mỗi agent."""
    agent:      str            # "text" | "image"
    analysis:   str            # phân tích chi tiết
    confidence: float          # 0.0 - 1.0
    sources:    list[dict]     # [{chunk_id, page, doc}]
    has_data:   bool           # True nếu agent nhận được chunks


# ──────────────────────────────────────────────
# Critical Agent (Stage 1)
# ──────────────────────────────────────────────

CRITICAL_AGENT_PROMPT = """\
Dựa trên câu hỏi và context, hãy xác định thông tin quan trọng cần tìm.

Cho MỖI nguồn (text và image), liệt kê 2-3 điểm cần tập trung.
ĐỂ RỘNG — không thu hẹp quá mức. Bao gồm cả thông tin trực tiếp lẫn liên quan.

[FUZZ ENTITY] Nếu câu hỏi KHÔNG đề cập tên công ty hoặc mã cổ phiếu cụ thể, hãy
xác định công ty nào trong context phù hợp nhất với đặc điểm mô tả trong câu hỏi
(ví dụ: "công ty sản xuất sữa" → VNM, "ngân hàng có NIM cao nhất" → xem context).
Đưa tên công ty đã xác định vào phần "text" để TextAgent biết cần tìm về ai.

Ví dụ câu hỏi "NIM thay đổi thế nào từ Q2 sang Q3?":
  text: "Tìm giá trị NIM ở Q2/2025 VÀ Q3/2025, nguyên nhân thay đổi, so sánh với các ngân hàng khác"
  image: "Tìm biểu đồ NIM theo quý, đọc giá trị trục Y tại Q2 và Q3, xu hướng đường"

Trả lời ĐÚNG định dạng JSON, KHÔNG thêm gì khác:
{"text": "các điểm cần tìm trong text/bảng", "image": "các điểm cần nhìn trong biểu đồ"}"""


class CriticalAgent:
    """Extract critical info để guide TextAgent + ImageAgent."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def extract(
        self, question: str,
        text_context: str,
        image_context: str,
    ) -> dict:
        """
        Returns: {"text": "...", "image": "..."}
        """
        user_msg = (
            f"Câu hỏi: {question}\n\n"
            f"Text context (tóm tắt):\n{text_context[:1500]}\n\n"
            f"Image captions:\n{image_context[:500]}"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CRITICAL_AGENT_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or ""
        parsed = _parse_json(raw)

        return {
            "text":  parsed.get("text", ""),
            "image": parsed.get("image", ""),
        }


# ──────────────────────────────────────────────
# Text Agent (Stage 2a) — text + table combined
# ──────────────────────────────────────────────

TEXT_AGENT_PROMPT = """\
Bạn là Text Agent — chuyên phân tích nội dung văn bản và bảng số liệu báo cáo tài chính.

Nhiệm vụ:
1. Trích xuất dữ kiện quan trọng: Tập trung vào số liệu, nhận định, khuyến nghị liên quan đến câu hỏi.
2. Hiểu ngữ cảnh: Chú ý đến ý nghĩa và chi tiết trong văn bản và bảng.
3. Trả lời rõ ràng: Dùng thông tin đã trích xuất để đưa ra câu trả lời ngắn gọn, chính xác.

Bạn được cung cấp CRITICAL INFO — đây là gợi ý về phần quan trọng nhất cần tập trung.
Hãy ưu tiên tìm thông tin liên quan đến critical info trong context.

QUAN TRỌNG — Hai khả năng đặc biệt:
[SELF-CORRECTION] CHỈ kích hoạt khi bạn CHẮC CHẮN 100% rằng câu hỏi có CON SỐ CỤ THỂ sai.
Phải thỏa 3 điều kiện:
  (a) Câu hỏi KHẲNG ĐỊNH một con số (VD: "GPM = 45%", "doanh thu 10 nghìn tỷ")
  (b) Context có con số KHÁC RÕ RÀNG cho cùng chỉ tiêu đó
  (c) Hai con số KHÔNG thể cùng đúng
Ví dụ ĐÚNG: "Câu hỏi nêu GPM = 45% nhưng bảng trang 12 ghi GPM = 40,4%." → self-correct.
KHÔNG self-correct khi:
  - Câu hỏi chỉ nhắc tên quý/năm mà không khẳng định con số (VD: "NIM Q3/2025" → KHÔNG sửa)
  - Khác biệt do làm tròn (16,6% vs 17%) → KHÔNG sửa
  - Bạn không tìm thấy con số trong context → KHÔNG sửa, để trống
Khi không chắc chắn → ĐỂ TRỐNG trường self_correction.

[FUZZ ENTITY] Nếu câu hỏi KHÔNG nhắc tên công ty hoặc mã cổ phiếu cụ thể,
hãy XÁC ĐỊNH công ty phù hợp nhất dựa trên đặc điểm mô tả trong câu hỏi và context.
Nêu rõ: "Dựa trên đặc điểm [X], công ty được đề cập là [tên/mã]."

Nếu context không đủ để trả lời, nêu rõ: 'Thông tin không có trong tài liệu được cung cấp.'

Lưu ý: Bạn CHỈ có thông tin từ text/bảng. Các agents khác bổ sung thông tin từ biểu đồ.

Trả lời dạng JSON:
{
  "analysis": "Phân tích chi tiết từ văn bản và bảng (2-5 câu, nêu con số cụ thể)",
  "confidence": 0.0-1.0,
  "key_facts": ["fact1", "fact2"],
  "self_correction": "Ghi rõ nếu câu hỏi có thông tin sai, để trống nếu không",
  "entity_resolved": "Tên công ty đã xác định nếu câu hỏi mơ hồ, để trống nếu không cần"
}

Nếu context không đủ để trả lời, đặt confidence thấp và nêu rõ thiếu gì.
CHỈ trả lời JSON, không thêm gì khác."""


class TextAgent:
    """Phân tích text + table chunks, guided by critical info."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def analyze(
        self, question: str,
        chunks: list[RetrievedChunk],
        critical_info: str = "",
    ) -> AgentOutput:
        if not chunks:
            return AgentOutput(
                agent="text", analysis="", confidence=0.0,
                sources=[], has_data=False,
            )

        context = _format_chunks(chunks)

        critical_section = ""
        if critical_info:
            critical_section = f"\n\nCRITICAL INFO (tập trung vào đây): {critical_info}"

        user_msg = f"Context:\n{context}{critical_section}\n\nCâu hỏi: {question}"

        raw = self._call_llm(user_msg)
        parsed = _parse_json(raw)

        return AgentOutput(
            agent="text",
            analysis=parsed.get("analysis", raw),
            confidence=float(parsed.get("confidence", 0.5)),
            sources=[
                {"chunk_id": c.chunk_id, "page": c.page, "doc": c.doc}
                for c in chunks
            ],
            has_data=True,
        )

    def _call_llm(self, user_msg: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": TEXT_AGENT_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


# ──────────────────────────────────────────────
# Image Agent (Stage 2b)
# ──────────────────────────────────────────────

IMAGE_AGENT_PROMPT = """\
Bạn là Image Agent chuyên phân tích và trích xuất thông tin từ hình ảnh biểu đồ tài chính.
Hình ảnh có thể bao gồm biểu đồ đường, biểu đồ cột, biểu đồ tròn, hoặc ảnh chụp trang báo cáo.

Nhiệm vụ:
1. ĐỌC TEXT trong ảnh (OCR): Trích xuất nhãn trục, legend, tiêu đề, số liệu trên biểu đồ.
   Nêu rõ các CON SỐ CỤ THỂ mà bạn đọc được (VD: "GPM = 40,4% ở Q4/2025").
2. PHÂN TÍCH trực quan: Nhận diện xu hướng, điểm cực đại/cực tiểu, giao điểm,
   vị trí tương đối giữa các đường/cột.
3. KẾT HỢP text và visual để trả lời câu hỏi chính xác.

Bạn được cung cấp CRITICAL INFO — đây là gợi ý về phần quan trọng nhất cần nhìn trong biểu đồ.
Hãy tập trung vào vùng/chi tiết liên quan đến critical info.

Lưu ý quan trọng:
- CHỈ trả lời dựa trên thông tin NHÌN THẤY trong ảnh. Không suy đoán thêm.
- Nếu ảnh không rõ hoặc không liên quan đến câu hỏi, đặt confidence thấp và nói rõ.
- Các agents khác có thể bổ sung thông tin từ text — bạn chỉ cần tập trung vào ảnh.

Trả lời dạng JSON:
{
  "analysis": "Phân tích chi tiết từ biểu đồ (2-5 câu, nêu con số cụ thể đọc được)",
  "confidence": 0.0-1.0,
  "chart_type": "line|bar|pie|other",
  "trend": "tăng|giảm|ổn định|biến động"
}

CHỈ trả lời JSON, không thêm gì khác."""


class ImageAgent:
    """Phân tích biểu đồ, guided by critical info."""

    def __init__(self, client: OpenAI, vision_model: str):
        self.client = client
        self.vision_model = vision_model

    def analyze(
        self, question: str,
        chunks: list[RetrievedChunk],
        critical_info: str = "",
    ) -> AgentOutput:
        if not chunks:
            return AgentOutput(
                agent="image", analysis="", confidence=0.0,
                sources=[], has_data=False,
            )

        content: list[dict] = []

        caption_text = "\n".join(
            f"[{i+1}] {c.caption}" for i, c in enumerate(chunks) if c.caption
        )

        critical_section = ""
        if critical_info:
            critical_section = f"\n\nCRITICAL INFO (tập trung nhìn vào đây): {critical_info}"

        content.append({
            "type": "text",
            "text": f"Image captions:\n{caption_text}{critical_section}\n\nCâu hỏi: {question}",
        })

        n_img = 0
        seen: set[str] = set()
        for c in chunks:
            if c.img_path and c.img_path not in seen:
                b64 = _encode_image(c.img_path)
                if b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                    n_img += 1
                    seen.add(c.img_path)
            if n_img >= 3:
                break

        raw = self._call_vision(content)
        parsed = _parse_json(raw)

        return AgentOutput(
            agent="image",
            analysis=parsed.get("analysis", raw),
            confidence=float(parsed.get("confidence", 0.5)),
            sources=[
                {"chunk_id": c.chunk_id, "page": c.page, "doc": c.doc,
                 "img_path": c.img_path}
                for c in chunks
            ],
            has_data=True,
        )

    def _call_vision(self, content: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": IMAGE_AGENT_PROMPT},
                {"role": "user",   "content": content},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


# ──────────────────────────────────────────────
# Sum Agent (Stage 3) — editor, not writer
# ──────────────────────────────────────────────

SUM_AGENT_PROMPT = """\
Bạn là editor — nhận bản nháp ghép từ nhiều agents và chỉnh sửa thành câu trả lời mạch lạc.

Bạn nhận được BẢN NHÁP đã ghép sẵn từ Text Agent và Image Agent.
Bản nháp có thể rời rạc, lặp lại, hoặc có mâu thuẫn nhỏ.

Nhiệm vụ:
1. GIỮ NGUYÊN tất cả thông tin, số liệu, con số cụ thể từ bản nháp.
   KHÔNG được bỏ bất kỳ dữ kiện nào — chỉ sắp xếp lại cho mạch lạc.
2. Loại bỏ phần LẶP LẠI (nếu 2 agents nói cùng 1 điều, giữ 1 lần).
3. Nếu có MÂU THUẪN, giữ cả 2 phía và ghi chú nguồn
   (VD: "Theo text: 16,6%, theo biểu đồ: ~17%").
4. Viết lại thành 1 đoạn văn liền mạch, tự nhiên, đầy đủ.

QUAN TRỌNG — Kiểm tra trước khi viết câu trả lời:
[SELF-CORRECTION] CHỈ khi bản nháp có trường self_correction KHÔNG RỖNG
VÀ nội dung đó nêu rõ CẢ con số sai LẪN con số đúng (VD: "45% vs 40,4%"),
thì mới đặt lưu ý lên đầu. Nếu self_correction rỗng hoặc mơ hồ → BỎ QUA.
Ví dụ mở đầu: "Lưu ý: Câu hỏi nêu GPM = 45% nhưng theo tài liệu, con số đúng là 40,4%."

[FUZZ ENTITY] Nếu bản nháp đã xác định được tên công ty (có trường entity_resolved),
hãy nhắc rõ công ty đó trong câu trả lời cuối.

KHÔNG tóm tắt. KHÔNG rút gọn. KHÔNG thêm thông tin mới.
Chỉ CHỈNH SỬA cho đọc được — giữ nguyên nội dung.

Trả lời dạng JSON:
{
  "answer": "Câu trả lời đã chỉnh sửa mạch lạc (tiếng Việt)",
  "confidence": 0.0-1.0,
  "reasoning": "Ghi chú ngắn về cách chỉnh sửa"
}

CHỈ trả lời JSON, không thêm gì khác."""


class SumAgent:
    """Editor — nhận bản nháp ghép thô, chỉnh mạch lạc, giữ nguyên nội dung."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def synthesize(
        self,
        question: str,
        text_output: AgentOutput,
        image_output: AgentOutput,
    ) -> dict:
        # Ghép thô trước
        draft_parts = []
        for out in [text_output, image_output]:
            if out.has_data and out.analysis:
                draft_parts.append(out.analysis)

        draft = " ".join(draft_parts) if draft_parts else "Không đủ thông tin."

        user_msg = (
            f"Bản nháp (ghép từ Text Agent và Image Agent):\n{draft}\n\n"
            f"Câu hỏi gốc: {question}"
        )

        raw = self._call_llm(user_msg)
        parsed = _parse_json(raw)

        all_sources = text_output.sources + image_output.sources

        active = [o for o in [text_output, image_output] if o.has_data]
        avg_conf = sum(o.confidence for o in active) / len(active) if active else 0.0
        final_conf = float(parsed.get("confidence", avg_conf))

        return {
            "answer":     parsed.get("answer", raw),
            "sources":    all_sources,
            "confidence": final_conf,
            "reasoning":  parsed.get("reasoning", ""),
            "agent_outputs": {
                "text":  {"analysis": text_output.analysis,
                          "confidence": text_output.confidence,
                          "has_data": text_output.has_data},
                "image": {"analysis": image_output.analysis,
                          "confidence": image_output.confidence,
                          "has_data": image_output.has_data},
            },
        }

    def _call_llm(self, user_msg: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUM_AGENT_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


# ──────────────────────────────────────────────
# Backward compatibility
# ──────────────────────────────────────────────

# TableAgent = TextAgent (gộp)
TableAgent = TextAgent

# GeneralAgent stub cho code cũ
class GeneralAgent:
    """Deprecated — giữ lại cho backward compatibility."""
    def __init__(self, client, model):
        pass
    def analyze(self, question, text_chunks, image_chunks):
        return AgentOutput(agent="general", analysis="", confidence=0.0,
                          sources=[], has_data=False)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(không có dữ liệu)"
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.chunk_type.upper()} | Trang {c.page} | {c.doc}"
        parts.append(f"{header}\n{c.content}")
    return "\n\n---\n\n".join(parts)


def _encode_image(img_path: str) -> str | None:
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except (FileNotFoundError, OSError):
        return None


def _parse_json(text: str) -> dict:
    """Parse JSON từ LLM response, xử lý markdown code block."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"analysis": text, "confidence": 0.5}
