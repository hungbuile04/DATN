# src/embedding/reembed_images.py

"""
Re-embed toàn bộ ảnh trong ChromaDB với weighted blend (α=0.8).

Dùng khi:
  - Đã sửa embed_image() sang weighted blend nhưng vector cũ vẫn dùng Gemini aggregated
  - Muốn áp dụng alpha khác mà không embed lại từ đầu

Caption đã có sẵn trong ChromaDB → KHÔNG re-caption, chỉ re-embed.
Tốn 2 API call/ảnh (caption_vec + image_vec riêng).

Usage:
    conda run -n base python src/embedding/reembed_images.py
    conda run -n base python src/embedding/reembed_images.py --alpha 0.8
    conda run -n base python src/embedding/reembed_images.py --dry-run
    conda run -n base python src/embedding/reembed_images.py --ticker VNM
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG


def run(alpha: float = 0.8, dry_run: bool = False, ticker_filter: str = ""):
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    import chromadb
    from src.embedding.embedder import GeminiEmbedder

    model_name   = CFG["embedding"].get("model", "gemini-embedding-2-preview")
    image_col_name = CFG["embedding"].get("image_collection", "rag_image")
    sleep_sec    = CFG["embedding"].get("sleep_between_batches", 1.5)

    embedder = GeminiEmbedder(api_key=api_key, model=model_name, sleep_seconds=sleep_sec)

    chroma_path = str(CFG["paths"]["vector_db"])
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection(image_col_name)

    data = col.get(include=["documents", "metadatas"])
    total = len(data["ids"])
    print(f"Total images in ChromaDB : {total}")
    print(f"Alpha (caption weight)   : {alpha}")
    if ticker_filter:
        print(f"Ticker filter            : {ticker_filter}")

    # ── Lọc ảnh cần re-embed ──
    to_process = []
    skipped_no_path = 0
    skipped_fallback = 0
    skipped_ticker = 0

    for i, chunk_id in enumerate(data["ids"]):
        caption  = data["documents"][i] or ""
        meta     = data["metadatas"][i]
        img_path = meta.get("img_path", "")
        ticker   = meta.get("ticker", "")

        # Bỏ qua nếu không có file ảnh
        if not img_path or not Path(img_path).exists():
            skipped_no_path += 1
            continue

        # Bỏ qua nếu caption vẫn là fallback (dùng fix_captions.py trước)
        if caption.startswith("Biểu đồ tài chính ("):
            skipped_fallback += 1
            continue

        # Lọc theo ticker nếu có
        if ticker_filter and ticker != ticker_filter:
            skipped_ticker += 1
            continue

        to_process.append({
            "chunk_id": chunk_id,
            "caption":  caption,
            "img_path": img_path,
            "meta":     meta,
        })

    print(f"\nSkipped (no image file)  : {skipped_no_path}")
    print(f"Skipped (fallback caption): {skipped_fallback}  ← chạy fix_captions.py trước")
    print(f"Skipped (ticker filter)  : {skipped_ticker}")
    print(f"To re-embed              : {len(to_process)}")

    if not to_process:
        print("\nKhông có ảnh nào cần re-embed.")
        return

    if dry_run:
        print("\n[DRY RUN] Would re-embed:")
        for item in to_process[:10]:
            print(f"  {item['chunk_id']}")
            print(f"    caption: {item['caption'][:80]}...")
        if len(to_process) > 10:
            print(f"  ... và {len(to_process) - 10} ảnh nữa")
        print(f"\nEstimated API calls: {len(to_process) * 2} (2 calls/image)")
        return

    # ── Re-embed ──
    done = 0
    failed = 0

    for j, item in enumerate(to_process):
        chunk_id = item["chunk_id"]
        caption  = item["caption"]
        img_path = item["img_path"]

        print(f"\n[{j+1}/{len(to_process)}] {chunk_id}")
        print(f"  caption: {caption[:70]}...")

        # ── Call 1: Caption vector ──
        try:
            caption_vec = embedder.embed_text(caption)
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"  ✗ Caption embed failed: {e}")
            failed += 1
            time.sleep(5)
            continue

        # ── Call 2: Image-only vector ──
        try:
            from google.genai import types as genai_types
            from src.embedding.embedder import _get_mime

            img_bytes = open(img_path, "rb").read()
            mime      = _get_mime(Path(img_path))
            parts     = [genai_types.Part.from_bytes(data=img_bytes, mime_type=mime)]

            result = embedder._retry(
                embedder.client.models.embed_content,
                model=embedder.model,
                contents=[genai_types.Content(parts=parts)],
            )
            image_vec = list(result.embeddings[0].values)
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"  ✗ Image embed failed: {e}")
            failed += 1
            time.sleep(5)
            continue

        # ── Blend: α × caption + (1-α) × image → L2-normalize ──
        blended = [alpha * c + (1 - alpha) * v for c, v in zip(caption_vec, image_vec)]
        norm    = sum(x * x for x in blended) ** 0.5
        new_vec = [x / norm for x in blended] if norm > 0 else blended

        # ── Update ChromaDB ──
        col.update(
            ids=[chunk_id],
            embeddings=[new_vec],
        )

        done += 1
        print(f"  ✓ Re-embedded (alpha={alpha})")

    # ── Invalidate BM25 image cache ──
    bm25_cache = Path(CFG["paths"]["bm25_cache"]) / "bm25_image.pkl"
    if bm25_cache.exists():
        bm25_cache.unlink()
        print(f"\n✓ Deleted BM25 image cache (sẽ rebuild lần sau)")

    print(f"\n{'='*50}")
    print(f"DONE: re-embedded={done}, failed={failed}, skipped_fallback={skipped_fallback}")
    if skipped_fallback:
        print(f"⚠  Còn {skipped_fallback} ảnh fallback caption → chạy fix_captions.py để fix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed all images with weighted blend")
    parser.add_argument("--alpha",   type=float, default=0.8,
                        help="Caption weight (0.0=image_only, 1.0=caption_only, default=0.8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ xem danh sách, không thực sự re-embed")
    parser.add_argument("--ticker",  type=str, default="",
                        help="Chỉ re-embed ảnh của ticker cụ thể (VD: VNM)")
    args = parser.parse_args()

    run(alpha=args.alpha, dry_run=args.dry_run, ticker_filter=args.ticker)