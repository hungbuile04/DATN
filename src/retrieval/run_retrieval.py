# src/retrieval/run_retrieval.py

"""
Test script cho dual-pipeline retrieval.

Usage:
    python src/retrieval/run_retrieval.py --query "biến động giá cổ phiếu VNM"
    python src/retrieval/run_retrieval.py --query "NIM HDB quý 3" --text_top_k 3 --image_top_k 2
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from config.settings import CFG
from query_router import classify_query
from retriever import DualRetriever


def run(query: str, text_top_k: int, image_top_k: int):
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    chroma_path = str(CFG["paths"]["vector_db"])

    query_type = classify_query(query)

    # Query type chỉ dùng để TĂNG image_top_k khi visual, KHÔNG cắt về 0.
    # Lý do: câu hỏi "factual" vẫn có thể có biểu đồ liên quan
    #   (vd: "giá than nhập khẩu" → biểu đồ giá than HT1)
    # Để retriever tự filter bằng relevance score thay vì hard cut.
    if query_type == "visual":
        image_top_k = max(image_top_k, 3)

    print(f"\n🔍 Query       : {query}")
    print(f"🏷  Query type  : {query_type}")
    print(f"📦 Text top-k  : {text_top_k}")
    print(f"📷 Image top-k : {image_top_k}")
    print(f"{'─' * 60}")

    retriever = DualRetriever(
        chroma_path=chroma_path,
        api_key=api_key,
        cfg=CFG,
    )

    result = retriever.retrieve(query, text_top_k=text_top_k, image_top_k=image_top_k)

    # --- Text results ---
    print(f"\n{'─' * 60}")
    print(f"📝 TEXT PIPELINE ({len(result.text_chunks)} results)")
    print(f"{'─' * 60}")
    if not result.text_chunks:
        print("  (trống)")
    for i, chunk in enumerate(result.text_chunks, 1):
        print(f"\n  [{i}] {chunk.chunk_id}")
        print(f"      type  : {chunk.chunk_type}  |  score: {chunk.score:.4f}")
        print(f"      doc   : {chunk.doc}  |  page: {chunk.page}")
        preview = chunk.content[:200].replace("\n", " ")
        print(f"      text  : {preview}...")

    # --- Image results ---
    print(f"\n{'─' * 60}")
    print(f"📷 IMAGE PIPELINE ({len(result.image_chunks)} results)")
    print(f"{'─' * 60}")
    if not result.image_chunks:
        print("  (trống)")
    for i, chunk in enumerate(result.image_chunks, 1):
        print(f"\n  [{i}] {chunk.chunk_id}")
        print(f"      score   : {chunk.score:.4f}")
        print(f"      doc     : {chunk.doc}  |  page: {chunk.page}")
        print(f"      caption : {chunk.caption[:150]}...")
        print(f"      img     : {chunk.img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dual-pipeline RAG retrieval")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--text_top_k", type=int, default=5)
    parser.add_argument("--image_top_k", type=int, default=3)
    args = parser.parse_args()
    run(query=args.query, text_top_k=args.text_top_k, image_top_k=args.image_top_k)
