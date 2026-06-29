# src/embedding/run_embedding.py

"""
Chạy dual-pipeline embedding:
    Pipeline 1: text + table → rag_text collection
    Pipeline 2: image → auto-caption → rag_image collection

Sử dụng:
    cd source_code_2
    python src/embedding/run_embedding.py

Yêu cầu:
    .env chứa GOOGLE_API_KEY=AIza...
    metadata.json đã được tạo bởi run_extraction.py
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from config.settings import CFG
from src.embedding.embedder import embed_and_store


def run() -> None:
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "Thiếu GOOGLE_API_KEY.\n"
            "Tạo file .env ở root project với nội dung:\n"
            "  GOOGLE_API_KEY=AIza..."
        )

    metadata_path = CFG["paths"]["metadata"]
    chroma_path   = str(CFG["paths"]["vector_db"])

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {metadata_path}\n"
            "Chạy run_extraction.py trước."
        )

    embed_model     = CFG["embedding"].get("model", "gemini-embedding-2-preview")
    text_col_name   = CFG["embedding"].get("text_collection", "rag_text")
    image_col_name  = CFG["embedding"].get("image_collection", "rag_image")

    print(f"Metadata       : {metadata_path}")
    print(f"Vector DB      : {chroma_path}")
    print(f"Model          : {embed_model}")
    print(f"Text col       : {text_col_name}")
    print(f"Image col      : {image_col_name}")

    stats = embed_and_store(
        metadata_path=str(metadata_path),
        chroma_path=chroma_path,
        api_key=api_key,
        cfg=CFG,
    )

    print(f"\nSummary:")
    print(f"  Text   embedded : {stats['n_text']}")
    print(f"  Table  embedded : {stats['n_table']}")
    print(f"  Image  embedded : {stats['n_image']}")
    print(f"  rag_text  total : {stats['text_collection_total']}")
    print(f"  rag_image total : {stats['image_collection_total']}")


if __name__ == "__main__":
    run()
