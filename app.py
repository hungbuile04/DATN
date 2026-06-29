"""
Streamlit Demo — Hệ thống RAG Phân tích Tài chính Đa phương thức (Multi-Agent)

Giao diện demo cho đồ án tốt nghiệp:
    - Hỏi đáp tài chính (Q&A)
    - Hiển thị pipeline retrieval (text + image)
    - Trực quan hóa luồng reasoning 3-agent (Text + Image → Sum)
    - Thống kê hệ thống
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Thiết lập path ──
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import json
import shutil
import re

from config.settings import CFG
from src.retrieval.retriever import DualRetriever, RetrievalResult
from src.agents.orchestrator import MultiAgentOrchestrator
from src.eval.generator import AnswerGenerator

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Phân tích Tài chính",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* CSS Variables for theming */
    :root {
        --primary: #1e3a5f;
        --secondary: #2d6a9f;
        --accent: #3498db;
        --success: #2ecc71;
        --warning: #f39c12;
        --danger: #e74c3c;
        --bg-card: #fafbfc;
        --border: #e0e0e0;
        --text-main: #333;
        --text-muted: #6c757d;
    }
    
    [data-theme="dark"] {
        --primary: #0f1c2e;
        --secondary: #1a3c5a;
        --bg-card: #1e1e1e;
        --border: #333;
        --text-main: #eee;
        --text-muted: #aaa;
    }

    /* Chỉ áp dụng font cho các thành phần custom để không làm hỏng icon của Streamlit */
    .main-header, .agent-card, .chunk-card, .stat-box {
        font-family: 'Inter', sans-serif;
    }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes progressGlow {
        0% { box-shadow: 0 0 5px var(--accent); }
        50% { box-shadow: 0 0 15px var(--accent); }
        100% { box-shadow: 0 0 5px var(--accent); }
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: fadeInUp 0.5s ease;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Agent cards */
    .agent-card {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        background: var(--bg-card);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .agent-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .agent-card.text     { border-left: 4px solid var(--accent); }
    .agent-card.image    { border-left: 4px solid var(--success); }
    .agent-card.sum      { border-left: 4px solid var(--warning); }

    /* Chunk cards */
    .chunk-card {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        background: var(--bg-card);
        animation: fadeInUp 0.4s ease;
    }
    .chunk-score {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .score-high   { background: rgba(46, 204, 113, 0.2); color: #155724; }
    .score-medium { background: rgba(243, 156, 18, 0.2); color: #856404; }
    .score-low    { background: rgba(231, 76, 60, 0.2); color: #721c24; }
    
    [data-theme="dark"] .score-high { color: #82e0aa; }
    [data-theme="dark"] .score-medium { color: #f5b041; }
    [data-theme="dark"] .score-low { color: #f1948a; }


    /* Pipeline flow */
    .pipeline-step {
        text-align: center;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }
    .pipeline-arrow {
        text-align: center;
        font-size: 1.5rem;
        color: var(--text-muted);
        padding: 0.3rem;
    }

    /* Stats */
    .stat-box {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--border) 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-box h3 { margin: 0; font-size: 1.6rem; color: var(--primary); }
    .stat-box p  { margin: 0.2rem 0 0 0; font-size: 0.85rem; color: var(--text-muted); }

    /* Chat messages adjustments */
    .stChatMessage { 
        max-width: 100% !important; 
        animation: fadeInUp 0.4s ease; 
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border);
    }
    .stChatMessage.user { 
        background-color: rgba(52, 152, 219, 0.06); 
        border-radius: 16px 16px 4px 16px; 
        border: 1px solid rgba(52, 152, 219, 0.15);
    }
    .stChatMessage.assistant { 
        background-color: var(--bg-card);
        border-radius: 16px 16px 16px 4px; 
    }
    
    /* Làm đẹp expander bên trong chat bubble */
    .stChatMessage .stExpander {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        background-color: rgba(0, 0, 0, 0.01) !important;
        margin-top: 0.5rem;
    }
    [data-theme="dark"] .stChatMessage .stExpander {
        background-color: rgba(255, 255, 255, 0.01) !important;
    }
</style>

""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Initialize system (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang khởi tạo hệ thống RAG...")
def init_system():
    """Khởi tạo Retriever, Orchestrator, và AnswerGenerator một lần."""
    google_key = os.getenv("GOOGLE_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    if not google_key or not openrouter_key:
        return None, None, None, "Thiếu GOOGLE_API_KEY hoặc OPENROUTER_API_KEY trong .env"

    # Reranker config từ config.yaml
    use_reranker = CFG.get("retrieval", {}).get("use_reranker", False)

    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
        openrouter_key=openrouter_key,
        use_reranker=use_reranker,
    )

    orchestrator = MultiAgentOrchestrator(
        api_key=openrouter_key,
        text_model=CFG["agents"]["llm_model"],
        vision_model=CFG["agents"]["vision_model"],
    )

    generator = AnswerGenerator(
        api_key=openrouter_key,
        text_model=CFG["agents"]["llm_model"],
        vision_model=CFG["agents"]["vision_model"],
    )

    return retriever, orchestrator, generator, None


def get_system_stats(retriever: DualRetriever) -> dict:
    """Lấy thống kê hệ thống."""
    text_count = retriever.text_col.count()
    image_count = retriever.image_col.count()
    tickers = sorted(retriever._known_tickers)
    pdf_dir = CFG["paths"]["pdf_raw"]
    pdf_count = len(list(Path(pdf_dir).glob("*.pdf"))) if Path(pdf_dir).exists() else 0
    return {
        "text_chunks": text_count,
        "image_chunks": image_count,
        "total_chunks": text_count + image_count,
        "tickers": tickers,
        "num_tickers": len(tickers),
        "num_pdfs": pdf_count,
    }


def get_document_list() -> list[dict]:
    """Lấy danh sách tài liệu từ metadata.json và ChromaDB."""
    meta_path = CFG["paths"]["metadata"]
    if not Path(meta_path).exists():
        return []

    with open(meta_path, encoding="utf-8") as f:
        all_entries = json.load(f)

    # Group by doc name
    docs = {}
    for entry in all_entries:
        doc_name = entry.get("doc", "unknown")
        if doc_name not in docs:
            docs[doc_name] = {
                "doc": doc_name,
                "ticker": entry.get("ticker", ""),
                "report_date": entry.get("report_date", ""),
                "text_chunks": 0,
                "table_chunks": 0,
                "image_chunks": 0,
            }
        chunk_type = entry.get("chunk_type", "")
        if chunk_type == "text":
            docs[doc_name]["text_chunks"] += 1
        elif chunk_type == "table":
            docs[doc_name]["table_chunks"] += 1
        elif chunk_type == "image":
            docs[doc_name]["image_chunks"] += 1

    result = list(docs.values())
    result.sort(key=lambda x: x["ticker"])
    return result


def upload_and_process(uploaded_file, retriever) -> tuple[bool, str]:
    """
    Xử lý file PDF upload: lưu → extract → embed → cập nhật retriever.
    Returns: (success, message)
    """
    pdf_dir = Path(CFG["paths"]["pdf_raw"])
    out_dir = str(CFG["paths"]["pdf_processed"])
    meta_path = Path(CFG["paths"]["metadata"])

    # 1. Lưu file PDF
    pdf_path = pdf_dir / uploaded_file.name
    if pdf_path.exists():
        return False, f"⚠️ File `{uploaded_file.name}` đã tồn tại trong hệ thống."

    pdf_path.write_bytes(uploaded_file.getvalue())

    try:
        # 2. Extract PDF → chunks
        from src.extraction.extractor import extract_pdf
        chunks = extract_pdf(
            str(pdf_path), out_dir,
            images_dpi=CFG["extraction"].get("dpi", 150),
        )
        if not chunks:
            return False, f"❌ Không extract được chunk nào từ `{uploaded_file.name}`."

        # 3. Cập nhật metadata.json (append)
        all_meta = []
        if meta_path.exists():
            all_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        all_meta.extend(chunks)
        meta_path.write_text(
            json.dumps(all_meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # 4. Embed chunks mới → ChromaDB
        from src.embedding.embedder import GeminiEmbedder, load_or_create_collection, _build_meta
        embed_cfg = CFG["embedding"]
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        embedder = GeminiEmbedder(api_key=api_key, model=embed_cfg.get("model", "gemini-embedding-2-preview"))

        text_col = load_or_create_collection(
            str(CFG["paths"]["vector_db"]), embed_cfg.get("text_collection", "rag_text")
        )
        image_col = load_or_create_collection(
            str(CFG["paths"]["vector_db"]), embed_cfg.get("image_collection", "rag_image")
        )

        n_text = 0
        n_image = 0

        # Text + Table chunks
        text_chunks = [c for c in chunks if c.get("chunk_type") in ("text", "table")]
        if text_chunks:
            contents = []
            for c in text_chunks:
                txt_path = Path(c["text_path"])
                contents.append(txt_path.read_text(encoding="utf-8") if txt_path.exists() else "")
            vectors = embedder.embed_texts_batch(contents)
            text_col.upsert(
                ids=[c["chunk_id"] for c in text_chunks],
                embeddings=vectors,
                documents=contents,
                metadatas=[_build_meta(c) for c in text_chunks],
            )
            n_text = len(text_chunks)

        # Image chunks
        img_chunks = [c for c in chunks if c.get("chunk_type") == "image" and c.get("img_path") and Path(c["img_path"]).exists()]
        for c in img_chunks:
            try:
                caption = embedder.caption_image(c["img_path"])
                c["caption"] = caption
                vec = embedder.embed_image(c["img_path"], caption=caption)
                image_col.upsert(
                    ids=[c["chunk_id"]],
                    embeddings=[vec],
                    documents=[caption],
                    metadatas=[_build_meta(c)],
                )
                n_image += 1
            except Exception as e:
                cid = c.get("chunk_id", "?")
                st.warning(f"⚠️ Lỗi embed ảnh {cid}: {e}")
                continue

        # 5. Reload retriever collections
        retriever.text_col = text_col
        retriever.image_col = image_col
        retriever._known_tickers = set(
            m.get("ticker", "") for m in text_col.get(include=["metadatas"])["metadatas"]
            if m.get("ticker")
        )

        ticker = chunks[0].get("ticker", "N/A") if chunks else "N/A"
        return True, (
            f"✅ Upload thành công `{uploaded_file.name}`!\n\n"
            f"- **Mã CK:** {ticker}\n"
            f"- **Text/Table chunks:** {n_text}\n"
            f"- **Image chunks:** {n_image}\n"
            f"- **Tổng chunks mới:** {n_text + n_image}"
        )

    except Exception as e:
        # Rollback: xóa file PDF nếu lỗi
        if pdf_path.exists():
            pdf_path.unlink()
        return False, f"❌ Lỗi xử lý: {e}"


@st.cache_resource(ttl=300, show_spinner=False)
def cached_retrieve(_retriever, question: str, text_top_k: int, image_top_k: int):
    """Bọc retriever.retrieve() với cache 5 phút để tăng tốc."""
    return _retriever.retrieve(
        question,
        text_top_k=text_top_k,
        image_top_k=image_top_k,
    )




def score_class(score: float) -> str:
    if score >= 0.5:
        return "score-high"
    elif score >= 0.2:
        return "score-medium"
    return "score-low"


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar(retriever):
    with st.sidebar:
        st.markdown("## ⚙️ Cấu hình")
        
        # Nút xóa lịch sử chat
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.results_history = []
            st.rerun()
            
        st.markdown("---")

        # ── Chế độ trả lời ──
        MODE_OPTIONS = {
            "multi_agent": "🤖 Multi-Agent (3 Agents)",
            "single_llm": "💬 Single LLM",
        }
        answer_mode = st.radio(
            "Chế độ trả lời",
            options=list(MODE_OPTIONS.keys()),
            format_func=lambda k: MODE_OPTIONS[k],
            index=0,
            help="**Multi-Agent:** 3-agent pipeline (Text + Image song song → Sum). "
                 "**Single LLM:** 1 lần gọi LLM duy nhất với toàn bộ context.",
        )

        # Hiển thị mô tả ngắn
        if answer_mode == "multi_agent":
            st.caption("3 LLM calls: Text + Image (song song) → Sum Agent")
        else:
            st.caption("1 LLM call: gửi toàn bộ text + image context cho LLM")

        st.markdown("")

        text_top_k = st.slider(
            "Số lượng Text chunks (top-k)",
            min_value=1, max_value=15, value=6,
            help="Số đoạn văn bản/bảng được truy xuất"
        )
        image_top_k = st.slider(
            "Số lượng Image chunks (top-k)",
            min_value=0, max_value=10, value=4,
            help="Số biểu đồ/hình ảnh được truy xuất"
        )

        st.markdown("")

        # ── Reranker toggle ──
        use_reranker = st.toggle(
            "🔄 Reranker (Gemini Flash)",
            value=CFG.get("retrieval", {}).get("use_reranker", False),
            help="Bật/tắt Gemini Flash Reranker. Reranker đánh giá lại độ liên quan của từng chunk bằng LLM, "
                 "cải thiện Precision và MRR. Tốn thêm ~0.5-1s/câu hỏi."
        )

        if use_reranker:
            st.caption("⚡ Reranker: Dense Search → LLM scoring → Top-k")
        else:
            st.caption("⚡ Dense Search → Top-k (không rerank)")

        st.markdown("---")

        # System stats
        st.markdown("## 📈 Thống kê hệ thống")
        stats = get_system_stats(retriever)

        col1, col2 = st.columns(2)
        col1.metric("PDF báo cáo", stats["num_pdfs"])
        col2.metric("Mã cổ phiếu", stats["num_tickers"])

        col3, col4 = st.columns(2)
        col3.metric("Text chunks", f"{stats['text_chunks']:,}")
        col4.metric("Image chunks", f"{stats['image_chunks']:,}")

        st.markdown(f"**Tổng:** {stats['total_chunks']:,} chunks")

        with st.expander("Danh sách mã cổ phiếu"):
            st.write(", ".join(stats["tickers"]))

        st.markdown("---")

        # Mở đầu file, bỏ phần tech stack ở sidebar.
    return text_top_k, image_top_k, answer_mode, use_reranker


# ─────────────────────────────────────────────
# Render: Retrieval Results
# ─────────────────────────────────────────────

def render_retrieval_results(result: RetrievalResult):
    """Hiển thị kết quả retrieval — dùng container thay expander để tránh lồng expander."""

    tab_text, tab_image = st.tabs([
        "Text/Table (" + str(len(result.text_chunks)) + ")",
        "Image (" + str(len(result.image_chunks)) + ")",
    ])

    with tab_text:
        if not result.text_chunks:
            st.info("Không tìm thấy text chunks phù hợp.")
        for i, chunk in enumerate(result.text_chunks):
            st.markdown(
                f"**[{i+1}]** `{chunk.chunk_type.upper()}` | **{chunk.doc}** — Trang {chunk.page} "
                f"<span class='chunk-score {score_class(chunk.score)}'>Score: {chunk.score:.4f}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Chunk ID: {chunk.chunk_id}")
            st.markdown(chunk.content[:1200] if len(chunk.content) > 1200 else chunk.content)
            if i < len(result.text_chunks) - 1:
                st.divider()

    with tab_image:
        if not result.image_chunks:
            st.info("Không tìm thấy image chunks phù hợp.")
        for i, chunk in enumerate(result.image_chunks):
            col_img, col_info = st.columns([1, 1])
            with col_img:
                if chunk.img_path and Path(chunk.img_path).exists():
                    st.image(chunk.img_path, use_container_width=True)
                else:
                    st.warning("Không tìm thấy file ảnh.")
            with col_info:
                st.markdown(
                    f"**[{i+1}]** **{chunk.doc}** — Trang {chunk.page} "
                    f"<span class='chunk-score {score_class(chunk.score)}'>Score: {chunk.score:.4f}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Chunk ID: {chunk.chunk_id}")
                if chunk.caption:
                    st.markdown("**Caption:**")
                    st.markdown(chunk.caption[:800])
            if i < len(result.image_chunks) - 1:
                st.divider()


# ─────────────────────────────────────────────
# Render: Agent Pipeline
# ─────────────────────────────────────────────

def render_agent_pipeline(final: dict):
    """Hiển thị quá trình reasoning của 3-agent pipeline."""

    # Pipeline flow diagram: Text + Image (song song) → Sum
    cols = st.columns([1, 0.3, 1, 0.3, 1])
    stages = [
        ("📄", "Text Agent", "#3498db"),
        ("+", "", ""),
        ("🖼️", "Image Agent", "#2ecc71"),
        ("→", "", ""),
        ("✏️", "Sum Agent", "#f39c12"),
    ]
    for col, (icon, name, color) in zip(cols, stages):
        with col:
            if name:
                st.markdown(f"""
                <div style="text-align:center; padding:0.5rem; border:2px solid {color};
                            border-radius:10px; background:{color}10;">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div style="font-size:0.8rem; font-weight:600; color:{color};">{name}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align:center; padding:1rem; font-size:1.5rem; color:#aaa;">
                    {icon}
                </div>
                """, unsafe_allow_html=True)

    st.caption("⚡ Text Agent và Image Agent chạy song song (parallel), giảm ~40% latency")

    st.markdown("")

    # Agent outputs
    agent_outputs = final.get("agent_outputs", {})

    text_out = agent_outputs.get("text", {})
    image_out = agent_outputs.get("image", {})

    col_text, col_image = st.columns(2)

    with col_text:
        st.markdown("**📄 Stage 1a: Text Agent**")
        if text_out.get("has_data"):
            st.markdown(text_out.get("analysis", "N/A"))
        else:
            st.warning("Text Agent không nhận được dữ liệu.")

    with col_image:
        st.markdown("**🖼️ Stage 1b: Image Agent**")
        if image_out.get("has_data"):
            st.markdown(image_out.get("analysis", "N/A"))
        else:
            st.warning("Image Agent không nhận được dữ liệu.")

    # Sum Agent reasoning
    if final.get("reasoning"):
        st.divider()
        st.markdown("**✏️ Stage 2: Sum Agent — Biên tập & tổng hợp**")
        st.markdown(final["reasoning"])


# ─────────────────────────────────────────────
# Render: Final Answer
# ─────────────────────────────────────────────

def render_answer(final: dict, total_time: float = None):
    """Hiển thị câu trả lời cuối cùng."""
    st.markdown("### 💡 Câu trả lời")

    # Dùng markdown để render **Nguồn tham khảo:** và nhãn [X, Trang Y] đúng định dạng
    st.markdown(final['answer'])


    st.markdown("")

    # Metadata inline
    time_str = f" &nbsp;|&nbsp; **⏱️ Thời gian:** {total_time:.1f}s" if total_time is not None else ""
    meta_str = (
        f"**Model:** `{final.get('model', 'N/A')}` &nbsp;|&nbsp; "
        f"**Ảnh:** {'✅' if final.get('has_image') else '❌'} ({final.get('num_images', 0)}) &nbsp;|&nbsp; "
        f"**Nguồn:** {len(final.get('sources', []))} chunks"
        f"{time_str}"
    )
    st.caption(meta_str)


# ─────────────────────────────────────────────
# Render: Sources and Details Helpers
# ─────────────────────────────────────────────

def format_answer_with_sources(answer: str, sources: list) -> str:
    """Tự động đính kèm danh sách tài liệu tham chiếu gọn gàng vào cuối câu trả lời nếu chưa có."""
    if not sources:
        return answer
    
    # Chuẩn hóa để tránh đính kèm trùng lặp
    lower_answer = answer.lower()
    if "nguồn tham" in lower_answer or "tài liệu tham" in lower_answer or "danh sách tài liệu" in lower_answer:
        return answer

    # Tách text và image sources, đánh số riêng
    text_sources = [s for s in sources if not s.get("img_path")]
    image_sources = [s for s in sources if s.get("img_path")]

    lines = []
    for i, src in enumerate(text_sources, 1):
        doc = src.get("doc", "N/A")
        page = src.get("page", "?")
        lines.append(f"[Đ{i}] {doc} | Trang {page}")
    for i, src in enumerate(image_sources, 1):
        doc = src.get("doc", "N/A")
        page = src.get("page", "?")
        lines.append(f"[HÌNH {i}] {doc} | Trang {page}")

    if not lines:
        return answer

    sources_md = "\n\n---\n**Nguồn tham khảo:**\n\n" + "\n\n".join(lines) + "\n"
    return answer + sources_md


def render_assistant_details(final: dict, result, answer_mode: str, total_time: float):
    """Render các thông tin chi tiết (timing, retrieval, pipeline) đi kèm tin nhắn."""

    # ── Metadata dòng nhỏ ──
    if total_time is not None:
        mode_label = "Multi-Agent" if answer_mode == "multi_agent" else "Single LLM"
        num_text = len(result.text_chunks) if (result and hasattr(result, "text_chunks")) else 0
        num_image = len(result.image_chunks) if (result and hasattr(result, "image_chunks")) else 0
        st.caption(
            f"⏱️ {total_time:.1f}s ({mode_label}) &nbsp;|&nbsp; "
            f"🗂️ {num_text}T + {num_image}I chunks &nbsp;|&nbsp; "
            f"`{final.get('model', 'N/A')}`"
        )

    # ── Chi tiết thu gọn — nằm trong chat bubble ──
    if result:
        num_text = len(result.text_chunks) if hasattr(result, "text_chunks") else 0
        num_image = len(result.image_chunks) if hasattr(result, "image_chunks") else 0
        with st.expander(
            f"🔍 Kết quả Retrieval — {num_text} text, {num_image} ảnh",
            expanded=False,
        ):
            render_retrieval_results(result)

    if answer_mode == "multi_agent":
        with st.expander("🤖 Multi-Agent Reasoning Pipeline", expanded=False):
            render_agent_pipeline(final)



# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Hệ thống RAG Phân tích tài chính đa phương thức</h1>
        <p>Đồ án tốt nghiệp</p>
    </div>
    """, unsafe_allow_html=True)

    # Init system
    retriever, orchestrator, generator, error = init_system()

    if error:
        st.error(f"Lỗi khởi tạo: {error}")
        st.stop()

    # Sidebar
    text_top_k, image_top_k, answer_mode, use_reranker = render_sidebar(retriever)
    # Cho phép toggle reranker runtime (không cần restart)
    retriever.use_reranker = use_reranker

    # ── Tabs chính ──
    tab_chat, tab_docs, tab_arch = st.tabs(["💬 Hỏi đáp", "📁 Tài liệu", "🏗️ Kiến trúc hệ thống"])

    # ═══════════════════════════════════════════
    # Tab 1: Hỏi đáp (Chat)
    # ═══════════════════════════════════════════
    with tab_chat:
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "results_history" not in st.session_state:
            st.session_state.results_history = []

        # Hiển thị lịch sử chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Nếu là assistant và có chứa dữ liệu phân tích đầy đủ, render lại các expander
                if msg["role"] == "assistant" and "final" in msg:
                    render_assistant_details(
                        final=msg["final"],
                        result=msg.get("result"),
                        answer_mode=msg.get("answer_mode"),
                        total_time=msg.get("total_time")
                    )

        # ── Câu hỏi gợi ý (luôn hiển thị) ──
        samples = [
            "Biên lợi nhuận gộp VNM các quý?",
            "So sánh NIM của HDB và CTG",
            "Doanh thu và LN HPG Q3/2025?",
            "Xu hướng giá MWG 6 tháng?",
            "Tỷ lệ nợ xấu Vietcombank?",
        ]

        def _set_sample(q: str):
            st.session_state["sample_question"] = q

        if not st.session_state.messages:
            st.markdown("### 💡 Gợi ý câu hỏi:")
        else:
            st.markdown("##### 💡 Gợi ý:")

        cols = st.columns(len(samples))
        for i, s in enumerate(samples):
            cols[i].button(s, key=f"sample_{i}", on_click=_set_sample, args=(s,))

        # Xử lý câu hỏi mẫu (nếu được click)
        sample_q = st.session_state.pop("sample_question", None)

    # ── Chat input đặt NGOÀI tab để Streamlit neo cố định ở đáy trang ──
    user_input = st.chat_input("Nhập câu hỏi về tài chính, cổ phiếu...")

    with tab_chat:
        question = sample_q or user_input

        if question:
            # Hiển thị câu hỏi
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Xử lý
            with st.chat_message("assistant"):
                # Step 1: Retrieval
                with st.status("Đang tìm kiếm tài liệu...", expanded=True) as status:
                    st.write("① Encoding query và search trên ChromaDB...")
                    t0 = time.time()
                    try:
                        result = cached_retrieve(
                            retriever,
                            question,
                            text_top_k=text_top_k,
                            image_top_k=image_top_k,
                        )
                        retrieval_time = time.time() - t0
                        st.write(f"✅ Tìm thấy **{len(result.text_chunks)}** text chunks, **{len(result.image_chunks)}** image chunks ({retrieval_time:.1f}s)")
                        status.update(label=f"Retrieval hoàn tất ({retrieval_time:.1f}s)", state="complete", expanded=False)
                    except Exception as e:
                        st.error(f"Lỗi retrieval: {e}")
                        status.update(label="Retrieval thất bại", state="error", expanded=True)
                        st.button("🔄 Thử lại", on_click=lambda: st.session_state.pop("sample_question", None))
                        st.stop()

                # Step 2: Generation (phân nhánh theo chế độ)
                if answer_mode == "multi_agent":
                    # ── Multi-Agent: 3-agent pipeline ──
                    with st.status("Multi-Agent đang phân tích...", expanded=True) as status:
                        st.write("② Stage 1: Text Agent + Image Agent — phân tích song song...")
                        st.write("③ Stage 2: Sum Agent — tổng hợp & biên tập...")
                        t0 = time.time()
                        try:
                            final = orchestrator.run(question, result)
                            reasoning_time = time.time() - t0
                            st.write(f"✅ Hoàn tất 3-agent reasoning ({reasoning_time:.1f}s)")
                            status.update(label=f"Multi-Agent hoàn tất ({reasoning_time:.1f}s)", state="complete", expanded=False)
                        except Exception as e:
                            st.error(f"Lỗi reasoning: {e}")
                            status.update(label="Multi-Agent thất bại", state="error", expanded=True)
                            st.stop()
                else:
                    # ── Single LLM: 1 lần gọi duy nhất ──
                    gen_mode = "full_multimodal" if result.image_chunks else "text_table"
                    with st.status(f"Single LLM đang sinh câu trả lời ({gen_mode})...", expanded=True) as status:
                        st.write(f"② Gửi {len(result.text_chunks)} text + {len(result.image_chunks)} image chunks cho LLM...")
                        t0 = time.time()
                        try:
                            gen_result = generator.generate(question, result, mode=gen_mode)
                            reasoning_time = time.time() - t0
                            # Chuẩn hóa output giống multi-agent để render chung
                            final = {
                                "answer": gen_result["answer"],
                                "model": gen_result["model"],
                                "has_image": gen_result["has_image"],
                                "num_images": gen_result["num_images"],
                                "sources": [
                                    {"chunk_id": c.chunk_id, "page": c.page, "doc": c.doc}
                                    for c in result.text_chunks
                                ] + [
                                    {"chunk_id": c.chunk_id, "page": c.page, "doc": c.doc, "img_path": c.img_path}
                                    for c in result.image_chunks
                                ],
                                "reasoning": "",
                                "agent_outputs": {},
                            }
                            st.write(f"✅ Hoàn tất ({reasoning_time:.1f}s) — model: {gen_result['model']}")
                            status.update(label=f"Single LLM hoàn tất ({reasoning_time:.1f}s)", state="complete", expanded=False)
                        except Exception as e:
                            st.error(f"Lỗi generation: {e}")
                            status.update(label="Single LLM thất bại", state="error", expanded=True)
                            st.stop()

                # ── Câu trả lời chính ──
                raw_answer = final.get("answer", "Không thể tạo câu trả lời.")
                sources = final.get("sources", [])
                
                # Định dạng câu trả lời kèm tài liệu tham chiếu trực tiếp
                answer_text = format_answer_with_sources(raw_answer, sources)
                st.markdown(answer_text)

                total_time = retrieval_time + reasoning_time
                
                # Render chi tiết đi kèm dưới dạng expanders
                render_assistant_details(
                    final=final,
                    result=result,
                    answer_mode=answer_mode,
                    total_time=total_time
                )

                # Lưu đầy đủ dữ liệu vào history để khi render lại lịch sử chat có thể hiển thị expander
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "result": result,
                    "final": final,
                    "answer_mode": answer_mode,
                    "total_time": total_time
                })
                st.session_state.results_history.append({
                    "question": question,
                    "result": result,
                    "final": final,
                    "answer_mode": answer_mode,
                    "retrieval_time": retrieval_time,
                    "reasoning_time": reasoning_time,
                })

    # ═══════════════════════════════════════════
    # Tab 2: Quản lý Tài liệu
    # ═══════════════════════════════════════════
    with tab_docs:
        st.markdown("## 📁 Quản lý Tài liệu")

        # ── Upload section ──
        st.markdown("### 📤 Upload tài liệu mới")
        st.caption("Upload file PDF báo cáo SSI Research. Hệ thống sẽ tự động: Extract text/table/image → Embedding → Lưu vào Vector DB.")

        uploaded_file = st.file_uploader(
            "Chọn file PDF",
            type=["pdf"],
            help="Định dạng tên: MÃ_YYYY_MM_DD_SSIResearch.pdf (VD: VIC_2026_05_15_SSIResearch.pdf)",
        )

        if uploaded_file:
            st.info(f"📄 File: `{uploaded_file.name}` ({uploaded_file.size / 1024:.0f} KB)")
            if st.button("🚀 Xử lý & Embed tài liệu", type="primary", use_container_width=True):
                with st.status("Đang xử lý tài liệu...", expanded=True) as status:
                    st.write("📄 Lưu file PDF...")
                    st.write("🔍 Extracting text, tables, images...")
                    st.write("🧬 Embedding chunks vào Vector DB...")

                    success, message = upload_and_process(uploaded_file, retriever)

                    if success:
                        status.update(label="✅ Upload thành công!", state="complete")
                        st.success(message)
                        # Clear cache để sidebar cập nhật stats
                        st.cache_resource.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        status.update(label="❌ Upload thất bại", state="error")
                        st.error(message)

        st.markdown("---")

        # ── Document list ──
        st.markdown("### 📋 Danh sách tài liệu trong hệ thống")

        doc_list = get_document_list()

        if doc_list:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📄 Tổng tài liệu", len(doc_list))
            col2.metric("📝 Text chunks", sum(d["text_chunks"] for d in doc_list))
            col3.metric("📊 Table chunks", sum(d["table_chunks"] for d in doc_list))
            col4.metric("🖼️ Image chunks", sum(d["image_chunks"] for d in doc_list))

            st.markdown("")

            # Table
            import pandas as pd
            df = pd.DataFrame(doc_list)
            df = df.rename(columns={
                "doc": "Tài liệu",
                "ticker": "Mã CK",
                "report_date": "Ngày",
                "text_chunks": "Text",
                "table_chunks": "Table",
                "image_chunks": "Image",
            })
            df["Tổng"] = df["Text"] + df["Table"] + df["Image"]

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Mã CK": st.column_config.TextColumn(width="small"),
                    "Text": st.column_config.NumberColumn(width="small"),
                    "Table": st.column_config.NumberColumn(width="small"),
                    "Image": st.column_config.NumberColumn(width="small"),
                    "Tổng": st.column_config.NumberColumn(width="small"),
                },
            )
        else:
            st.info("Chưa có tài liệu nào. Upload file PDF ở trên để bắt đầu.")

    # ═══════════════════════════════════════════
    # Tab 3: Kiến trúc hệ thống
    # ═══════════════════════════════════════════
    with tab_arch:
        st.markdown("## 🏗️ Kiến trúc hệ thống RAG Đa phương thức")

        st.markdown("""
        ### Tổng quan
        Hệ thống RAG (Retrieval-Augmented Generation) phân tích báo cáo tài chính
        với khả năng xử lý đa phương thức — kết hợp **văn bản**, **bảng số liệu**,
        và **biểu đồ/hình ảnh** để trả lời câu hỏi chính xác.
        """)

        # Pipeline diagram
        st.markdown("### Pipeline xử lý")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            #### 1️⃣ Ingestion
            - **Trích xuất PDF** (Docling)
            - Tách text, table, image
            - Chunking (400 tokens, overlap 80)
            - Phát hiện biểu đồ tự động
            - Xuất metadata.json
            """)

        with col2:
            st.markdown("""
            #### 2️⃣ Embedding
            - **Gemini Embedding 2** (3072-dim)
            - **Dual pipeline:**
              - Text+Table → `rag_text` collection
              - Image → auto-caption + weighted blend (α=0.8) → `rag_image` collection
            - Lưu trữ ChromaDB
            """)

        with col3:
            st.markdown("""
            #### 3️⃣ Retrieval & Generation
            - **Dual Retrieval:**
              - Text: Dense + ticker boost
              - Image: Dense + ticker hard filter
            - **3-Agent Reasoning:**
              - Text + Image (song song) → Sum
            """)

        st.markdown("---")

        # Multi-Agent detail
        st.markdown("### 🤖 Multi-Agent Reasoning (3 Agents)")
        st.caption("⚡ Text Agent và Image Agent chạy SONG SONG — giảm ~40% latency")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""
            #### 📄 Stage 1a: Text Agent
            - Phân tích văn bản và bảng số liệu
            - **SELF-CORRECTION:** Phát hiện & sửa lỗi dữ kiện trong câu hỏi
            - **FUZZ ENTITY:** Suy luận công ty khi câu hỏi mơ hồ
            - Trích xuất key facts, metrics, reasoning

            #### 🖼️ Stage 1b: Image Agent
            - Phân tích biểu đồ bằng Vision LLM
            - OCR text, đọc trục, đọc số liệu
            - Nhận diện xu hướng (tăng/giảm/ổn định)
            - Trả về tín hiệu xấp xỉ khi không có số chính xác
            """)

        with c2:
            st.markdown("""
            #### ✏️ Stage 2: Sum Agent (Editor)
            - Nhận bản nháp ghép từ Text + Image Agent
            - **Chỉ chỉnh sửa**, không viết lại
            - Xử lý mâu thuẫn (giữ cả 2 nguồn, ghi chú rõ)
            - Loại trùng lặp giữa 2 agents
            - Giữ nguyên mọi dữ kiện & con số

            #### ⚡ Tối ưu hiệu năng
            - 2 agents Stage 1 chạy parallel (`ThreadPoolExecutor`)
            - Resize ảnh max 1280px → giảm ~50% token cost
            - Tổng: 3 LLM calls/câu hỏi
            """)

        st.markdown("---")

        # Key features
        st.markdown("### ✨ Đặc điểm nổi bật")

        f1, f2, f3 = st.columns(3)

        with f1:
            st.markdown("""
            **🔀 Dual-Pipeline Retrieval**
            - Text & Image tách biệt hoàn toàn
            - Ticker-aware: tự nhận diện mã CK
            - Score boost (text) & hard filter (image)
            """)

        with f2:
            st.markdown("""
            **🧬 Aggregated Image Embedding**
            - Caption + Image blend (α=0.8)
            - Auto-caption bằng Gemini Vision
            - L2-normalized vectors
            - Cải thiện +22.3% qua ablation
            """)

        with f3:
            st.markdown("""
            **🔬 Evaluation Framework**
            - 4-mode comparison
            - LLM-as-a-Judge (4 tiêu chí)
            - Ablation studies đầy đủ
            - Precision, Recall, MRR, Hit Rate
            """)

        st.markdown("---")
        
        # Tech stack
        from config.settings import CFG
        st.markdown("### 🔧 Công nghệ sử dụng")
        st.markdown(f"""
        - **Embedding:** `{CFG['embedding']['model']}`
        - **LLM:** `{CFG['agents']['llm_model']}`
        - **Vision:** `{CFG['agents']['vision_model']}`
        - **Vector DB:** `ChromaDB`
        - **Dimension:** `{CFG['embedding']['embed_dim']}`
        """)


if __name__ == "__main__":
    main()
