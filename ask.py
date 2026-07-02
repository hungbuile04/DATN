import os
import argparse
import sys

from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")

if not google_key or not openrouter_key:
    print("Lỗi: Thiếu GOOGLE_API_KEY hoặc OPENROUTER_API_KEY trong file .env!")
    sys.exit(1)

from config.settings import CFG
from src.retrieval.retriever import DualRetriever
from src.agents.orchestrator import MultiAgentOrchestrator

def main():
    parser = argparse.ArgumentParser(description="Đặt câu hỏi trực tiếp cho hệ thống RAG (Multi-Agent).")
    parser.add_argument("--q", type=str, help="Câu hỏi của bạn (đặt trong dấu ngoặc kép)")
    parser.add_argument("--text_top_k", type=int, default=5, help="Số lượng text chunks cần lấy")
    parser.add_argument("--image_top_k", type=int, default=3, help="Số lượng image/table chunks cần lấy")
    args = parser.parse_args()

    print("⏳ Đang khởi tạo hệ thống Retrieval & Các Đặc vụ (Agents)...")
    
    # 1. Khởi tạo Retriever
    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )

    # 2. Khởi tạo Mô hình Multi-Agent
    orchestrator = MultiAgentOrchestrator(
        api_key=openrouter_key,
        text_model=CFG["agents"]["llm_model"],
        vision_model=CFG["agents"]["vision_model"],
    )

    # Chế độ tương tác liên tục (nếu không truyền tham số câu hỏi)
    if not args.q:
        print("\n" + "="*50)
        print("🤖 CHAT VỚI RAG HỆ THỐNG PHÂN TÍCH TÀI CHÍNH")
        print("Gõ 'exit' hoặc 'quit' để thoát.")
        print("="*50)
        
        while True:
            question = input("\nBạn hỏi 💬: ")
            if question.lower() in ['exit', 'quit']:
                break
            if not question.strip():
                continue
            process_question(question, retriever, orchestrator, args)
    else:
        # Chạy 1 lần nếu truyền cờ --q
        process_question(args.q, retriever, orchestrator, args)


def process_question(question, retriever, orchestrator, args):
    """Xử lý quá trình Retrieval -> Phân tích -> Sinh câu trả lời"""
    print(f"\n🔍 Đang tìm kiếm tài liệu...", end=" ", flush=True)
    try:
        # Lấy dữ liệu (Retrieval Stage)
        result = retriever.retrieve(question, text_top_k=args.text_top_k, image_top_k=args.image_top_k)
        print(f"✅ (Tìm thấy {len(result.text_chunks)} đoạn text, {len(result.image_chunks)} ảnh/biểu đồ)")
        
        print(f"🧠 Các đặc vụ đang phân tích (Reasoning)...")
        # Phân tích ra đáp án (Reasoning & Generation Stage)
        final = orchestrator.run(question, result)
        
        print("\n" + "─"*50)
        print(f"💡 CÂU TRẢ LỜI:\n{final['answer']}")
        print("─"*50)
        print(f"🤖 Lực lượng tham gia: {final['model']} | Tự tin (Confidence): {final['confidence']:.2f}")
        print(f"📝 Nguồn tài liệu đã dùng để trả lời: {len(final['sources'])} chunks")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra trong quá trình xử lý: {e}")

if __name__ == "__main__":
    main()
