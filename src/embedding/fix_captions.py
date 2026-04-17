# src/embedding/fix_captions.py

"""
Re-caption + re-embed tất cả ảnh bị caption fallback generic.

Vấn đề: Lúc chạy run_embedding.py, Gemini Vision bị 429/503
→ tất cả 98 ảnh rơi vào fallback "Biểu đồ tài chính (chunk_id)"
→ BM25 + Dense score cực thấp → image retrieval thất bại hoàn toàn

Fix: Re-caption từng ảnh bằng Gemini Vision Flash,
     rồi re-embed (aggregated: caption + image bytes)
     và update ChromaDB.

Usage:
    conda run -n base python src/embedding/fix_captions.py
    conda run -n base python src/embedding/fix_captions.py --dry-run   # chỉ xem, không sửa
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG


def run(dry_run: bool = False):
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY")

    import chromadb
    from src.embedding.embedder import GeminiEmbedder

    model_name = CFG["embedding"].get("model", "gemini-embedding-2-preview")
    image_col_name = CFG["embedding"].get("image_collection", "rag_image")
    sleep_sec = CFG["embedding"].get("sleep_between_batches", 1.5)

    embedder = GeminiEmbedder(api_key=api_key, model=model_name, sleep_seconds=sleep_sec)

    chroma_path = str(CFG["paths"]["vector_db"])
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection(image_col_name)

    data = col.get(include=["documents", "metadatas"])

    # Tìm ảnh cần fix
    to_fix = []
    for i, chunk_id in enumerate(data["ids"]):
        caption = data["documents"][i] or ""
        meta = data["metadatas"][i]
        img_path = meta.get("img_path", "")

        # if caption.startswith("Biểu đồ tài chính (") and img_path and Path(img_path).exists():
        if img_path and Path(img_path).exists():
            to_fix.append({
                "index": i,
                "chunk_id": chunk_id,
                "img_path": img_path,
                "old_caption": caption,
                "meta": meta,
            })

    print(f"Total images: {len(data['ids'])}")
    print(f"Need re-caption: {len(to_fix)}")

    if dry_run:
        print("\n[DRY RUN] Would fix:")
        for item in to_fix[:10]:
            print(f"  {item['chunk_id']}  →  {item['img_path']}")
        if len(to_fix) > 10:
            print(f"  ... and {len(to_fix) - 10} more")
        return

    # Re-caption + re-embed
    fixed = 0
    failed = 0

    for j, item in enumerate(to_fix):
        chunk_id = item["chunk_id"]
        img_path = item["img_path"]

        print(f"\n[{j+1}/{len(to_fix)}] {chunk_id}")

        # Step 1: Re-caption
        try:
            new_caption = embedder.caption_image(img_path)
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"  ✗ Caption failed: {e}")
            failed += 1
            time.sleep(5)
            continue

        # Kiểm tra caption có thực sự mới không
        if new_caption.startswith("Biểu đồ tài chính ("):
            print(f"  ✗ Still fallback, skipping")
            failed += 1
            continue

        print(f"  Caption: {new_caption[:80]}...")

        # Step 2: Re-embed (aggregated)
        try:
            new_vector = embedder.embed_image(img_path, caption=new_caption)
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"  ✗ Embed failed: {e}")
            failed += 1
            time.sleep(5)
            continue

        # Step 3: Update ChromaDB
        updated_meta = dict(item["meta"])
        updated_meta["caption"] = new_caption

        col.update(
            ids=[chunk_id],
            embeddings=[new_vector],
            documents=[new_caption],
            metadatas=[updated_meta],
        )

        fixed += 1
        print(f"  ✓ Updated")

    # Also update metadata.json
    metadata_path = CFG["paths"]["metadata"]
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            all_meta = json.load(f)

        updated_count = 0
        # Build lookup from ChromaDB
        updated_data = col.get(include=["documents", "metadatas"])
        caption_map = {}
        for i, cid in enumerate(updated_data["ids"]):
            caption_map[cid] = updated_data["documents"][i]

        for entry in all_meta:
            cid = entry.get("chunk_id", "")
            if cid in caption_map and not caption_map[cid].startswith("Biểu đồ tài chính ("):
                entry["caption"] = caption_map[cid]
                updated_count += 1

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_meta, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Updated {updated_count} entries in metadata.json")

    # Also rebuild BM25 cache (invalidate)
    bm25_cache = Path(CFG["paths"]["bm25_cache"]) / "bm25_image.pkl"
    if bm25_cache.exists():
        bm25_cache.unlink()
        print(f"✓ Deleted BM25 image cache (will rebuild on next retrieval)")

    print(f"\n{'='*50}")
    print(f"DONE: fixed={fixed}, failed={failed}, total={len(to_fix)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix fallback captions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
