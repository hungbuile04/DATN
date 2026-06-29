"""
Chỉ extract + embed cho các PDF mới (VIC, PLX).
KHÔNG chạy lại toàn bộ.

Sử dụng:
    cd source_code_2
    python embed_new_docs.py
"""

import sys, os, json, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent))

from config.settings import CFG
from src.extraction.extractor import extract_pdf
from src.embedding.embedder import (
    GeminiEmbedder, load_or_create_collection, _build_meta
)

# ── Cấu hình ──
PDF_FILES = [
    "pdfs/raw/VIC_2026_05_15_SSIResearch.pdf",
    "pdfs/raw/PLX_2026_05_15_SSIResearch(1).pdf",
]
METADATA_PATH = Path(CFG["paths"]["metadata"])
CHROMA_PATH   = str(CFG["paths"]["vector_db"])


def main():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    embed_cfg      = CFG["embedding"]
    batch_size     = embed_cfg.get("batch_size", 10)
    sleep_sec      = embed_cfg.get("sleep_between_batches", 1.5)
    min_token_text = embed_cfg.get("min_token_count", 50)
    text_col_name  = embed_cfg.get("text_collection", "rag_text")
    image_col_name = embed_cfg.get("image_collection", "rag_image")
    model_name     = embed_cfg.get("model", "gemini-embedding-2-preview")

    embedder         = GeminiEmbedder(api_key=api_key, model=model_name, sleep_seconds=sleep_sec)
    text_collection  = load_or_create_collection(CHROMA_PATH, text_col_name)
    image_collection = load_or_create_collection(CHROMA_PATH, image_col_name)

    # Load metadata hiện có
    if METADATA_PATH.exists():
        all_meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    else:
        all_meta = []

    existing_ids = {e["chunk_id"] for e in all_meta}

    total_text = 0
    total_image = 0

    for pdf_rel in PDF_FILES:
        pdf_path = Path(pdf_rel)
        if not pdf_path.exists():
            print(f"⚠ Không tìm thấy: {pdf_path}")
            continue

        print(f"\n{'='*60}")
        print(f"📄 Processing: {pdf_path.name}")
        print(f"{'='*60}")

        # ── Bước 1: Extract ──
        chunks = extract_pdf(
            pdf_path=str(pdf_path),
            out_dir=str(CFG["paths"]["pdf_processed"]),
            images_dpi=CFG["extraction"].get("dpi", 216),
        )

        if not chunks:
            print(f"  ❌ Không extract được chunk nào.")
            continue

        # Lọc chunks mới (chưa có trong metadata)
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        print(f"  → Extract: {len(chunks)} chunks, mới: {len(new_chunks)}")

        if not new_chunks:
            print(f"  ⏩ Tất cả chunks đã tồn tại, bỏ qua.")
            continue

        # Thêm vào metadata
        all_meta.extend(new_chunks)
        for c in new_chunks:
            existing_ids.add(c["chunk_id"])

        # ── Bước 2: Embed text + table ──
        text_chunks = [c for c in new_chunks
                       if c.get("chunk_type") in ("text", "table")
                       and c.get("token_count", 0) >= min_token_text]

        if text_chunks:
            print(f"  📝 Embedding {len(text_chunks)} text/table chunks...")
            contents = [Path(c["text_path"]).read_text(encoding="utf-8") for c in text_chunks]

            all_vectors = []
            for i in range(0, len(contents), batch_size):
                batch = contents[i: i + batch_size]
                vectors = embedder.embed_texts_batch(batch)
                all_vectors.extend(vectors)
                if i + batch_size < len(contents):
                    time.sleep(sleep_sec)

            text_collection.upsert(
                ids=[c["chunk_id"] for c in text_chunks],
                embeddings=all_vectors,
                documents=contents,
                metadatas=[_build_meta(c) for c in text_chunks],
            )
            total_text += len(text_chunks)
            print(f"  ✓ Text/Table: {len(text_chunks)} vectors → {text_col_name}")

        # ── Bước 3: Embed images ──
        img_chunks = [c for c in new_chunks
                      if c.get("chunk_type") == "image"
                      and c.get("img_path")
                      and Path(c["img_path"]).exists()]

        if img_chunks:
            print(f"  🖼️ Embedding {len(img_chunks)} image chunks...")
            for c in img_chunks:
                try:
                    # Step 1: Auto-caption bằng Gemini Vision
                    rich_caption = embedder.caption_image(c["img_path"])
                    time.sleep(0.5)

                    # Step 2: Weighted blend embed (α=0.8 caption + 0.2 image)
                    blended = embedder.embed_image(c["img_path"], caption=rich_caption)

                    c["caption"] = rich_caption

                    image_collection.upsert(
                        ids=[c["chunk_id"]],
                        embeddings=[blended],
                        documents=[rich_caption],
                        metadatas=[_build_meta(c)],
                    )
                    total_image += 1
                    print(f"    ✓ {c['chunk_id']}")
                except Exception as e:
                    print(f"    ✗ {c.get('chunk_id', '?')}: {e}")
                time.sleep(sleep_sec)

    # ── Lưu metadata cập nhật ──
    METADATA_PATH.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"✅ HOÀN TẤT")
    print(f"  Text/Table mới : {total_text}")
    print(f"  Image mới      : {total_image}")
    print(f"  rag_text tổng  : {text_collection.count()}")
    print(f"  rag_image tổng : {image_collection.count()}")
    print(f"  metadata.json  : {len(all_meta)} entries")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
