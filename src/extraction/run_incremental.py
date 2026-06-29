# src/extraction/run_incremental.py

"""
Xử lý TĂNG DẦN: chỉ extract + embed tài liệu MỚI (chưa có trong metadata.json).

Usage:
    cd source_code_2
    python src/extraction/run_incremental.py

    # Chỉ extract (không embed)
    python src/extraction/run_incremental.py --extract-only

    # Chỉ embed (đã extract rồi)
    python src/extraction/run_incremental.py --embed-only
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG
from src.extraction.extractor import extract_pdf


def get_existing_docs(metadata_path: Path) -> set[str]:
    """Đọc metadata.json → lấy danh sách doc đã xử lý."""
    if not metadata_path.exists():
        return set()
    with open(metadata_path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["doc"] for e in entries}


def find_new_pdfs(pdf_dir: Path, existing_docs: set[str]) -> list[Path]:
    """Tìm PDF chưa có trong metadata."""
    new_files = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc_name = pdf_path.stem  # tên file không có .pdf
        if doc_name not in existing_docs:
            new_files.append(pdf_path)
    return new_files


def run_extract(new_pdfs: list[Path]) -> list[dict]:
    """Extract PDF mới → append vào metadata.json."""
    out_dir = CFG["paths"]["pdf_processed"]
    metadata_path = CFG["paths"]["metadata"]
    dpi = CFG["extraction"].get("dpi", 150)

    # Đọc metadata cũ
    existing = []
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            existing = json.load(f)

    new_chunks = []
    for pdf_path in new_pdfs:
        print(f"\n📄 Extracting: {pdf_path.name}")
        chunks = extract_pdf(str(pdf_path), str(out_dir), images_dpi=dpi)
        new_chunks.extend(chunks)
        print(f"   → {len(chunks)} chunks")

    # Append vào metadata.json (KHÔNG ghi đè)
    combined = existing + new_chunks
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Đã thêm {len(new_chunks)} chunks mới vào metadata.json")
    print(f"  Tổng: {len(combined)} chunks ({len(existing)} cũ + {len(new_chunks)} mới)")

    return new_chunks


def run_embed():
    """Embed tất cả chunk chưa có trong vector DB (tự skip đã embed)."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    from src.embedding.embedder import embed_and_store

    metadata_path = CFG["paths"]["metadata"]
    chroma_path = str(CFG["paths"]["vector_db"])

    print(f"\n🔗 Embedding vào vector DB...")
    print(f"   Metadata: {metadata_path}")
    print(f"   ChromaDB: {chroma_path}")

    stats = embed_and_store(
        metadata_path=str(metadata_path),
        chroma_path=chroma_path,
        api_key=api_key,
        cfg=CFG,
    )

    print(f"\n✓ Embedding hoàn tất:")
    print(f"  Text   embedded : {stats['n_text']}")
    print(f"  Table  embedded : {stats['n_table']}")
    print(f"  Image  embedded : {stats['n_image']}")
    print(f"  rag_text  total : {stats['text_collection_total']}")
    print(f"  rag_image total : {stats['image_collection_total']}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Xử lý tăng dần tài liệu mới")
    p.add_argument("--extract-only", action="store_true",
                    help="Chỉ extract, không embed")
    p.add_argument("--embed-only", action="store_true",
                    help="Chỉ embed (metadata đã có)")
    args = p.parse_args()

    pdf_dir = CFG["paths"]["pdf_raw"]
    metadata_path = CFG["paths"]["metadata"]

    if args.embed_only:
        run_embed()
        return

    # Tìm PDF mới
    existing_docs = get_existing_docs(metadata_path)
    new_pdfs = find_new_pdfs(pdf_dir, existing_docs)

    if not new_pdfs:
        print("✅ Không có tài liệu mới cần xử lý.")
        print(f"   Đã có {len(existing_docs)} tài liệu trong metadata.json")
        return

    print(f"📋 Tìm thấy {len(new_pdfs)} tài liệu MỚI:")
    for f in new_pdfs:
        print(f"   + {f.name}")
    print(f"   (Đã có {len(existing_docs)} tài liệu cũ — KHÔNG xử lý lại)")

    # Extract
    new_chunks = run_extract(new_pdfs)

    # Embed (trừ khi --extract-only)
    if not args.extract_only and new_chunks:
        run_embed()

    print(f"\n🎉 Hoàn tất! Hệ thống sẵn sàng hỏi đáp với tài liệu mới.")


if __name__ == "__main__":
    main()
