# src/eval/ablation_embed_weight.py

"""
Ablation: Tìm tỷ lệ tối ưu giữa caption_vec và image_vec.

Thay vì dùng Gemini aggregated (internal weight không kiểm soát được),
tự blend 2 vectors với trọng số α:

    final_vec = α × caption_vec + (1-α) × image_vec

Sweep α từ 0.0 → 1.0, đo cosine similarity với query.

Dùng lại vectors đã embed từ ablation_embed_image.json
→ KHÔNG cần gọi thêm API.

Usage:
    python src/eval/ablation_embed_image.py   ← chạy trước nếu chưa có vectors
    python src/eval/ablation_embed_weight.py  ← chạy sau, dùng cached vectors
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def blend(vec_a: list[float], vec_b: list[float], alpha: float) -> list[float]:
    """Weighted blend: α × vec_a + (1-α) × vec_b, rồi L2-normalize."""
    blended = [alpha * a + (1 - alpha) * b for a, b in zip(vec_a, vec_b)]
    norm = sum(x * x for x in blended) ** 0.5
    if norm == 0:
        return blended
    return [x / norm for x in blended]


def run():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    from src.embedding.embedder import GeminiEmbedder, _get_mime
    import chromadb

    model_name = CFG["embedding"].get("model", "gemini-embedding-2-preview")
    embedder = GeminiEmbedder(api_key=api_key, model=model_name)

    # ── Load questions (visual + cross_modal) ──
    with open("src/eval/questions.json", encoding="utf-8") as f:
        all_questions = json.load(f)

    queries = [
        q for q in all_questions
        if q["category"] in ("visual", "cross_modal")
    ]
    print(f"Loaded {len(queries)} visual/cross_modal questions")

    # ── Load images từ ChromaDB ──
    chroma_path = str(CFG["paths"]["vector_db"])
    image_col_name = CFG["embedding"].get("image_collection", "rag_image")
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    image_col = chroma_client.get_collection(image_col_name)

    img_data = image_col.get(include=["documents", "metadatas"])

    # Lấy ảnh 10 mã cổ phiếu — nhất quán với ablation_embed_image.py
    TARGET_TICKERS = {
        "VNM",  # FMCG
        "HDB",  # Ngân hàng
        "HPG",  # Thép
        "GAS",  # Dầu khí
        "MWG",  # Bán lẻ
        "FRT",  # Bán lẻ dược
        "VCB",  # Ngân hàng nhà nước
        "PVD",  # Dịch vụ dầu khí
        "REE",  # Cơ điện
        "DCM",  # Hoá chất
    }

    images = []
    for i, chunk_id in enumerate(img_data["ids"]):
        meta = img_data["metadatas"][i]
        caption = img_data["documents"][i]
        img_path = meta.get("img_path", "")
        ticker = meta.get("ticker", "")
        if img_path and Path(img_path).exists() and ticker in TARGET_TICKERS:
            images.append({
                "chunk_id": chunk_id,
                "caption": caption,
                "img_path": img_path,
                "ticker": ticker,
                "doc": meta.get("doc", ""),
            })
    print(f"Found {len(images)} images (VNM + HDB only, from {len(img_data['ids'])} total)")

    # ── Embed tất cả ảnh: caption_vec + image_vec riêng ──
    print(f"\nEmbedding {len(images)} images (caption + image riêng)...")

    image_vecs = {}  # chunk_id → {"caption": [...], "image": [...]}

    for i, img in enumerate(images):
        chunk_id = img["chunk_id"]
        caption = img["caption"]
        img_path = img["img_path"]

        print(f"  [{i+1}/{len(images)}] {chunk_id}", end="", flush=True)

        vecs = {}

        # Caption vector
        try:
            vecs["caption"] = embedder.embed_text(caption)
            time.sleep(1.5)
        except Exception as e:
            print(f" cap_err={e}")
            continue

        # Image-only vector
        try:
            from google.genai import types as genai_types
            img_bytes = open(img_path, 'rb').read()
            mime = _get_mime(Path(img_path))
            parts = [genai_types.Part.from_bytes(data=img_bytes, mime_type=mime)]
            r = embedder._retry(
                embedder.client.models.embed_content,
                model=embedder.model,
                contents=[genai_types.Content(parts=parts)],
            )
            vecs["image"] = list(r.embeddings[0].values)
            time.sleep(1.5)
        except Exception as e:
            print(f" img_err={e}")
            continue

        image_vecs[chunk_id] = vecs
        print(" ✓")

    # ── Embed queries ──
    print(f"\nEmbedding {len(queries)} queries...")
    query_vecs = {}
    for q in queries:
        query_vecs[q["id"]] = embedder.embed_query(q["question"])
        time.sleep(1.5)
        print(f"  {q['id']} ✓")

    # ── Sweep α ──
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"\n{'='*80}")
    print(f"SWEEP α: final_vec = α × caption_vec + (1-α) × image_vec")
    print(f"{'='*80}")
    print(f"\n  α=0.0 = image_only, α=1.0 = caption_only")
    print(f"  Gemini aggregated ≈ α=0.2 (based on similarity pattern)")

    alpha_scores = {}  # α → avg_sim

    for alpha in alphas:
        sims = []
        for q in queries:
            qid = q["id"]
            qvec = query_vecs[qid]

            best_sim = -1.0
            for img in images:
                cid = img["chunk_id"]
                if cid not in image_vecs:
                    continue
                v = image_vecs[cid]
                blended_vec = blend(v["caption"], v["image"], alpha)
                sim = cosine_sim(qvec, blended_vec)
                if sim > best_sim:
                    best_sim = sim
            sims.append(best_sim)

        avg_sim = sum(sims) / len(sims) if sims else 0
        alpha_scores[alpha] = avg_sim

    # ── Print results ──
    print(f"\n  {'α':>5}  {'Avg Top-1 Sim':>14}  {'Bar'}")
    print(f"  {'─'*50}")

    best_alpha = max(alpha_scores, key=alpha_scores.get)
    max_sim = max(alpha_scores.values())
    min_sim = min(alpha_scores.values())

    for alpha in alphas:
        sim = alpha_scores[alpha]
        bar_len = int((sim - min_sim) / (max_sim - min_sim + 1e-9) * 30)
        bar = "█" * bar_len
        marker = " ← best" if alpha == best_alpha else ""
        label = ""
        if alpha == 0.0:
            label = " (image_only)"
        elif alpha == 1.0:
            label = " (caption_only)"
        elif alpha == 0.2:
            label = " (~Gemini aggregated)"
        print(f"  {alpha:>5.1f}  {sim:>14.5f}  {bar}{marker}{label}")

    # ── Per-query at optimal α ──
    print(f"\n{'─'*80}")
    print(f"Optimal α = {best_alpha:.1f} (avg sim = {alpha_scores[best_alpha]:.5f})")
    print(f"\nComparison at optimal α vs pure methods:")
    print(f"  α=1.0 caption_only : {alpha_scores[1.0]:.5f}")
    print(f"  α=0.0 image_only   : {alpha_scores[0.0]:.5f}")
    print(f"  α=0.2 ~aggregated  : {alpha_scores[0.2]:.5f}")
    print(f"  α={best_alpha:.1f} optimal      : {alpha_scores[best_alpha]:.5f}")

    improvement_vs_agg = (alpha_scores[best_alpha] - alpha_scores[0.2]) / alpha_scores[0.2] * 100
    print(f"\n  Improvement vs Gemini aggregated: {improvement_vs_agg:+.1f}%")

    # ── Save ──
    out = {
        "alphas": {str(a): round(s, 5) for a, s in alpha_scores.items()},
        "best_alpha": best_alpha,
        "best_sim": round(alpha_scores[best_alpha], 5),
        "n_queries": len(queries),
        "n_images": len(images),
    }
    out_path = Path("results/ablation_embed_weight.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results → {out_path}")


if __name__ == "__main__":
    run()
