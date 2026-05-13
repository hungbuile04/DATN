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
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Agent cards */
    .agent-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        background: #fafbfc;
    }
    .agent-card.text     { border-left: 4px solid #3498db; }
    .agent-card.image    { border-left: 4px solid #2ecc71; }
    .agent-card.sum      { border-left: 4px solid #f39c12; }

    /* Chunk cards */
    .chunk-card {
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        background: white;
    }
    .chunk-score {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .score-high   { background: #d4edda; color: #155724; }
    .score-medium { background: #fff3cd; color: #856404; }
    .score-low    { background: #f8d7da; color: #721c24; }

    /* Confidence meter */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
        overflow: hidden;
        margin-top: 4px;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Pipeline flow */
    .pipeline-step {
        text-align: center;
        padding: 0.5rem;
    }
    .pipeline-arrow {
        text-align: center;
        font-size: 1.5rem;
        color: #6c757d;
        padding: 0.3rem;
    }

    /* Stats */
    .stat-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-box h3 { margin: 0; font-size: 1.6rem; color: #1e3a5f; }
    .stat-box p  { margin: 0.2rem 0 0 0; font-size: 0.85rem; color: #6c757d; }

    /* Chat messages */
    .stChatMessage { max-width: 100% !important; }
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

    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
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


# ─────────────────────────────────────────────
# Helper: render confidence
# ─────────────────────────────────────────────

def render_confidence(confidence: float, label: str = "Confidence"):
    """Hiển thị thanh confidence."""
    pct = int(confidence * 100)
    if confidence >= 0.7:
        color = "#28a745"
    elif confidence >= 0.4:
        color = "#ffc107"
    else:
        color = "#dc3545"

    st.markdown(f"""
    <div style="margin: 0.3rem 0;">
        <span style="font-size:0.85rem; color:#555;">{label}: <b>{pct}%</b></span>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
            min_value=1, max_value=15, value=5,
            help="Số đoạn văn bản/bảng được truy xuất"
        )
        image_top_k = st.slider(
            "Số lượng Image chunks (top-k)",
            min_value=0, max_value=10, value=3,
            help="Số biểu đồ/hình ảnh được truy xuất"
        )

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

        # Tech stack
        st.markdown("## 🔧 Công nghệ")
        st.markdown(f"""
        - **Embedding:** {CFG['embedding']['model']}
        - **LLM:** {CFG['agents']['llm_model']}
        - **Vision:** {CFG['agents']['vision_model']}
        - **Vector DB:** ChromaDB
        - **Dimension:** {CFG['embedding']['embed_dim']}
        """)

        st.markdown("---")
        st.markdown("## 💡 Câu hỏi mẫu")
        samples = [
            "Biên lợi nhuận gộp (GPM) của VNM thay đổi thế nào qua các quý gần đây?",
            "So sánh NIM của HDB và CTG trong năm 2025",
            "Doanh thu và lợi nhuận của HPG trong Q3/2025 là bao nhiêu?",
            "Xu hướng giá cổ phiếu MWG trong 6 tháng gần đây?",
            "Tỷ lệ nợ xấu của Vietcombank thay đổi ra sao?",
        ]
        for s in samples:
            if st.button(s, key=f"sample_{hash(s)}", use_container_width=True):
                st.session_state["sample_question"] = s

    return text_top_k, image_top_k, answer_mode


# ─────────────────────────────────────────────
# Render: Retrieval Results
# ─────────────────────────────────────────────

def render_retrieval_results(result: RetrievalResult):
    """Hiển thị kết quả retrieval."""
    st.markdown("### 🔍 Kết quả Retrieval")

    tab_text, tab_image = st.tabs([
        f"📄 Text/Table ({len(result.text_chunks)})",
        f"🖼️ Image ({len(result.image_chunks)})",
    ])

    with tab_text:
        if not result.text_chunks:
            st.info("Không tìm thấy text chunks phù hợp.")
        for i, chunk in enumerate(result.text_chunks):
            with st.expander(
                f"#{i+1} | {chunk.chunk_type.upper()} | {chunk.doc} — Trang {chunk.page} | Score: {chunk.score:.4f}",
                expanded=(i == 0),
            ):
                # Score badge
                st.markdown(f"""
                <span class="chunk-score {score_class(chunk.score)}">
                    Score: {chunk.score:.4f}
                </span>
                """, unsafe_allow_html=True)

                st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
                st.markdown(f"**Loại:** {chunk.chunk_type} | **Trang:** {chunk.page} | **Tài liệu:** {chunk.doc}")

                st.markdown("**Nội dung:**")
                st.markdown(chunk.content[:1500] if len(chunk.content) > 1500 else chunk.content)

    with tab_image:
        if not result.image_chunks:
            st.info("Không tìm thấy image chunks phù hợp.")
        for i, chunk in enumerate(result.image_chunks):
            with st.expander(
                f"#{i+1} | {chunk.doc} — Trang {chunk.page} | Score: {chunk.score:.4f}",
                expanded=(i == 0),
            ):
                col_img, col_info = st.columns([1, 1])

                with col_img:
                    if chunk.img_path and Path(chunk.img_path).exists():
                        st.image(chunk.img_path, use_container_width=True)
                    else:
                        st.warning("Không tìm thấy file ảnh.")

                with col_info:
                    st.markdown(f"""
                    <span class="chunk-score {score_class(chunk.score)}">
                        Score: {chunk.score:.4f}
                    </span>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
                    st.markdown(f"**Trang:** {chunk.page} | **Tài liệu:** {chunk.doc}")

                    if chunk.caption:
                        st.markdown("**Caption:**")
                        st.markdown(chunk.caption[:800])


# ─────────────────────────────────────────────
# Render: Agent Pipeline
# ─────────────────────────────────────────────

def render_agent_pipeline(final: dict):
    """Hiển thị quá trình reasoning của 3-agent pipeline."""
    st.markdown("### 🤖 Multi-Agent Reasoning Pipeline")

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
        with st.expander("📄 Stage 1a: Text Agent — Phân tích văn bản & bảng", expanded=False):
            if text_out.get("has_data"):
                render_confidence(text_out.get("confidence", 0), "Text Confidence")
                st.markdown("**Phân tích:**")
                st.markdown(text_out.get("analysis", "N/A"))
            else:
                st.warning("Text Agent không nhận được dữ liệu.")

    with col_image:
        with st.expander("🖼️ Stage 1b: Image Agent — Phân tích biểu đồ", expanded=False):
            if image_out.get("has_data"):
                render_confidence(image_out.get("confidence", 0), "Image Confidence")
                st.markdown("**Phân tích:**")
                st.markdown(image_out.get("analysis", "N/A"))
            else:
                st.warning("Image Agent không nhận được dữ liệu.")

    # Sum Agent reasoning
    if final.get("reasoning"):
        with st.expander("✏️ Stage 2: Sum Agent — Biên tập & tổng hợp", expanded=False):
            st.markdown(final["reasoning"])


# ─────────────────────────────────────────────
# Render: Final Answer
# ─────────────────────────────────────────────

def render_answer(final: dict):
    """Hiển thị câu trả lời cuối cùng."""
    st.markdown("### 💡 Câu trả lời")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0f7ff 0%, #e8f4f8 100%);
                border: 1px solid #b8daff; border-radius: 12px; padding: 1.2rem 1.5rem;
                font-size: 1.05rem; line-height: 1.7;">
        {final['answer']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_confidence(final.get("confidence", 0), "Tổng Confidence")
    with col2:
        st.markdown(f"**Model:** `{final.get('model', 'N/A')}`")
    with col3:
        st.markdown(f"**Có biểu đồ:** {'✅' if final.get('has_image') else '❌'}")
    with col4:
        st.markdown(f"**Số nguồn:** {len(final.get('sources', []))}")


# ─────────────────────────────────────────────
# Render: Sources
# ─────────────────────────────────────────────

def render_sources(final: dict):
    """Hiển thị danh sách nguồn tài liệu."""
    sources = final.get("sources", [])
    if not sources:
        return

    with st.expander(f"📝 Nguồn tài liệu ({len(sources)} chunks)", expanded=False):
        for i, src in enumerate(sources):
            doc = src.get("doc", "N/A")
            page = src.get("page", "?")
            chunk_id = src.get("chunk_id", "N/A")
            img = src.get("img_path", "")
            icon = "🖼️" if img else "📄"
            st.markdown(f"{icon} **{chunk_id}** — {doc}, trang {page}")


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Hệ thống RAG Phân tích Tài chính Đa phương thức</h1>
        <p>Multi-Agent RAG với Dual-Pipeline Retrieval • Đồ án tốt nghiệp</p>
    </div>
    """, unsafe_allow_html=True)

    # Init system
    retriever, orchestrator, generator, error = init_system()

    if error:
        st.error(f"Lỗi khởi tạo: {error}")
        st.stop()

    # Sidebar
    text_top_k, image_top_k, answer_mode = render_sidebar(retriever)

    # ── Tabs chính ──
    tab_chat, tab_arch = st.tabs(["💬 Hỏi đáp", "🏗️ Kiến trúc hệ thống"])

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

        # Xử lý câu hỏi mẫu
        sample_q = st.session_state.pop("sample_question", None)

        # Chat input
        user_input = st.chat_input("Đặt câu hỏi về tài chính, cổ phiếu...")

        question = sample_q or user_input

        if question:
            # Hiển thị câu hỏi
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Xử lý
            with st.chat_message("assistant"):
                # Step 1: Retrieval
                with st.status("🔍 Đang tìm kiếm tài liệu...", expanded=True) as status:
                    st.write("Encoding query và search trên ChromaDB...")
                    t0 = time.time()
                    try:
                        result = retriever.retrieve(
                            question,
                            text_top_k=text_top_k,
                            image_top_k=image_top_k,
                        )
                        retrieval_time = time.time() - t0
                        st.write(f"✅ Tìm thấy **{len(result.text_chunks)}** text chunks, **{len(result.image_chunks)}** image chunks ({retrieval_time:.1f}s)")
                        status.update(label=f"🔍 Retrieval hoàn tất ({retrieval_time:.1f}s)", state="complete")
                    except Exception as e:
                        st.error(f"Lỗi retrieval: {e}")
                        status.update(label="❌ Retrieval thất bại", state="error")
                        st.stop()

                # Step 2: Generation (phân nhánh theo chế độ)
                if answer_mode == "multi_agent":
                    # ── Multi-Agent: 3-agent pipeline ──
                    with st.status("🤖 Multi-Agent đang phân tích...", expanded=True) as status:
                        st.write("Stage 1: Text Agent + Image Agent — phân tích song song...")
                        st.write("Stage 2: Sum Agent — tổng hợp & biên tập...")
                        t0 = time.time()
                        try:
                            final = orchestrator.run(question, result)
                            reasoning_time = time.time() - t0
                            st.write(f"✅ Hoàn tất 3-agent reasoning ({reasoning_time:.1f}s)")
                            status.update(label=f"🤖 Multi-Agent hoàn tất ({reasoning_time:.1f}s)", state="complete")
                        except Exception as e:
                            st.error(f"Lỗi reasoning: {e}")
                            status.update(label="❌ Multi-Agent thất bại", state="error")
                            st.stop()
                else:
                    # ── Single LLM: 1 lần gọi duy nhất ──
                    gen_mode = "full_multimodal" if result.image_chunks else "text_table"
                    with st.status(f"💬 Single LLM đang sinh câu trả lời ({gen_mode})...", expanded=True) as status:
                        st.write(f"Gửi {len(result.text_chunks)} text + {len(result.image_chunks)} image chunks cho LLM...")
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
                                "confidence": 0.0,  # Single LLM không trả confidence
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
                            status.update(label=f"💬 Single LLM hoàn tất ({reasoning_time:.1f}s)", state="complete")
                        except Exception as e:
                            st.error(f"Lỗi generation: {e}")
                            status.update(label="❌ Single LLM thất bại", state="error")
                            st.stop()

                # Hiển thị câu trả lời
                answer_text = final.get("answer", "Không thể tạo câu trả lời.")
                st.markdown(answer_text)

                if answer_mode == "multi_agent":
                    render_confidence(final.get("confidence", 0), "Độ tin cậy")

                # Lưu vào history
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                st.session_state.results_history.append({
                    "question": question,
                    "result": result,
                    "final": final,
                    "answer_mode": answer_mode,
                    "retrieval_time": retrieval_time,
                    "reasoning_time": reasoning_time,
                })

            # Chi tiết bên dưới chat
            st.markdown("---")

            # Render chi tiết
            render_answer(final)
            if answer_mode == "multi_agent":
                render_agent_pipeline(final)
            render_retrieval_results(result)
            render_sources(final)

            # Timing
            mode_label = "Multi-Agent" if answer_mode == "multi_agent" else "Single LLM"
            total_time = retrieval_time + reasoning_time
            st.caption(f"⏱️ Tổng thời gian: {total_time:.1f}s (Retrieval: {retrieval_time:.1f}s | {mode_label}: {reasoning_time:.1f}s)")

    # ═══════════════════════════════════════════
    # Tab 2: Kiến trúc hệ thống
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


if __name__ == "__main__":
    main()
