# src/agents/agents.py

"""
3 Agents cho Multi-Agent RAG.

Luồng 2 stages:
    Stage 1: TextAgent     (text+table chunks)   → text-based answer
             ImageAgent    (image chunks)         → image-based answer
    Stage 2: SumAgent      (ghép thô aT + aI → chỉnh sửa mạch lạc)

Thiết kế:
    - TextAgent: text + table gộp (không tách TableAgent riêng)
    - ImageAgent: OCR + visual analysis
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
# Text Agent — text + table combined
# ──────────────────────────────────────────────

TEXT_AGENT_PROMPT = """\
Bạn là Text Agent — chuyên phân tích nội dung văn bản và bảng số liệu báo cáo tài chính.

Nhiệm vụ:
1. Trích xuất dữ kiện quan trọng: Tập trung vào số liệu, nhận định, khuyến nghị liên quan đến câu hỏi.
2. Hiểu ngữ cảnh: Chú ý đến ý nghĩa và chi tiết trong văn bản và bảng.
3. Trả lời rõ ràng: Dùng thông tin đã trích xuất để đưa ra câu trả lời ngắn gọn, chính xác.

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
    """Phân tích text + table chunks."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def analyze(
        self, question: str,
        chunks: list[RetrievedChunk],
        critical_info: str = "",          # giữ tham số cho backward compat, nhưng bỏ qua
    ) -> AgentOutput:
        if not chunks:
            return AgentOutput(
                agent="text", analysis="", confidence=0.0,
                sources=[], has_data=False,
            )

        context = _format_chunks(chunks)
        user_msg = f"Context:\n{context}\n\nCâu hỏi: {question}"

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
            max_tokens=1500,
        )
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""


# ──────────────────────────────────────────────
# Image Agent
# ──────────────────────────────────────────────

IMAGE_AGENT_PROMPT = """\
Bạn là Image Agent chuyên phân tích hình ảnh biểu đồ tài chính và trích xuất thông tin trực quan.

Vai trò của bạn là CUNG CẤP TÍN HIỆU TỪ HÌNH ẢNH để hỗ trợ các agent khác, KHÔNG phải để phủ định hoặc loại bỏ thông tin.

---

## NHIỆM VỤ CHÍNH

### 1. ĐỌC TEXT & SỐ LIỆU (OCR)

* Trích xuất các thông tin nhìn thấy được:

  * tiêu đề
  * nhãn trục
  * chú giải (legend)
  * số liệu trên biểu đồ
* Nếu không có số chính xác, hãy đưa ra giá trị xấp xỉ:

  * ví dụ: "~4.8%", "khoảng 5%", "gần 10"

---

### 2. PHÂN TÍCH TRỰC QUAN

* Xác định xu hướng:

  * tăng / giảm / ổn định / biến động
* So sánh tương đối:

  * A cao hơn B
  * đạt đỉnh / giảm mạnh / giao nhau
* Nhận diện pattern quan trọng ngay cả khi không có số cụ thể

---

### 3. ƯU TIÊN TÍN HIỆU HỮU ÍCH

* Nếu không có số chính xác:

  * vẫn phải cung cấp:

    * khoảng giá trị
    * xu hướng
    * quan hệ tương đối

* KHÔNG được trả lời "không có thông tin" nếu vẫn có thể suy ra xu hướng hoặc so sánh

---

### 4. XỬ LÝ KHÔNG CHẮC CHẮN

* Khi không chắc:

  * dùng các từ:

    * "khoảng"
    * "có vẻ"
    * "ước tính"
    * "xấp xỉ"

* Tránh khẳng định tuyệt đối nếu dữ liệu không rõ

---

## NGUYÊN TẮC QUAN TRỌNG

* CHỈ sử dụng thông tin nhìn thấy trong ảnh
* KHÔNG dùng kiến thức bên ngoài
* KHÔNG bịa số liệu chính xác
* KHÔNG phủ định quá sớm (tránh kiểu: "biểu đồ không có dữ liệu")
* Vai trò của bạn là BỔ SUNG THÔNG TIN, không phải kiểm duyệt

---

## FORMAT OUTPUT (BẮT BUỘC JSON)

{
  "analysis": "2–4 câu mô tả các số liệu (hoặc xấp xỉ), xu hướng và so sánh quan sát được từ biểu đồ",
  "confidence": 0.0-1.0,
  "chart_type": "line|bar|pie|other",
  "trend": "tăng|giảm|ổn định|biến động|không_rõ"
}

---

## VÍ DỤ TỐT

* "Biểu đồ cho thấy NIM giảm từ khoảng ~6% xuống ~4.5% từ Q2 sang Q3."
* "ROE duy trì quanh mức ~20–25% và tương đối ổn định."
* "Cột A luôn thấp hơn cột B trong toàn bộ giai đoạn."

---

## VÍ DỤ CẦN TRÁNH

* "Biểu đồ không cung cấp thông tin" (khi vẫn có thể suy ra xu hướng)
* "Không có dữ liệu" (khi có thể ước lượng hoặc so sánh)
* Bịa số liệu cụ thể không nhìn thấy

---

Mục tiêu của bạn là: trích xuất tối đa tín hiệu hữu ích từ hình ảnh để hỗ trợ suy luận ở bước sau.
CHỈ trả lời JSON, không thêm gì khác."""


class ImageAgent:
    """Phân tích biểu đồ."""

    def __init__(self, client: OpenAI, vision_model: str):
        self.client = client
        self.vision_model = vision_model

    def analyze(
        self, question: str,
        chunks: list[RetrievedChunk],
        critical_info: str = "",          # giữ tham số cho backward compat, nhưng bỏ qua
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

        content.append({
            "type": "text",
            "text": f"Image captions:\n{caption_text}\n\nCâu hỏi: {question}",
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
            max_tokens=1500,
        )
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""


# ──────────────────────────────────────────────
# Sum Agent — editor, not writer
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
            max_tokens=2000,
        )
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""


# ──────────────────────────────────────────────
# Backward compatibility
# ──────────────────────────────────────────────

# CriticalAgent stub — giữ lại cho backward compat (import cũ không bị vỡ)
class CriticalAgent:
    """Deprecated — CriticalAgent đã bị loại bỏ khỏi pipeline chính."""
    def __init__(self, client, model):
        pass
    def extract(self, question="", text_context="", image_context=""):
        return {"text": "", "image": ""}

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


def _encode_image(img_path: str, max_long_edge: int = 1280) -> str | None:
    """
    Encode ảnh sang base64, resize nếu vượt max_long_edge.
    Giảm ~50% token cost cho Image Agent mà không mất chi tiết biểu đồ.
    """
    try:
        from PIL import Image
        import io

        img = Image.open(img_path)
        w, h = img.size
        long_edge = max(w, h)

        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except (FileNotFoundError, OSError, ImportError):
        # Fallback: đọc raw nếu PIL không có
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
