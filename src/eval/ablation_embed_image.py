# src/eval/ablation_embed_image.py

"""
Ablation: So sánh 3 cách embed ảnh cho image retrieval.

Mục tiêu: chứng minh aggregated embedding (caption + image bytes)
tốt hơn caption-only hoặc image-only.

3 phương pháp:
  1. caption_only  — chỉ embed text caption (bỏ image bytes)
  2. image_only    — chỉ embed image bytes (bỏ caption)
  3. aggregated    — caption + image bytes cùng lúc (hiện tại)

Đo 2 metric:
  - Cosine similarity: query ↔ image vector (cao = relevant hơn)
  - Retrieval Rank:    ảnh đúng xếp hạng bao nhiêu trong top-k

Dùng các câu hỏi visual + cross_modal từ questions.json.
Caption lấy từ ChromaDB (đã lưu sẵn), không re-generate.

Usage:
    python src/eval/ablation_embed_image.py
    python src/eval/ablation_embed_image.py --questions src/eval/questions.json
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


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def run(questions_path: str):
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    from src.embedding.embedder import GeminiEmbedder, _get_mime
    import chromadb

    model_name = CFG["embedding"].get("model", "gemini-embedding-2-preview")
    embedder = GeminiEmbedder(api_key=api_key, model=model_name)

    # ── Load questions (chỉ visual + cross_modal) ──
    with open(questions_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    queries = [
        q for q in all_questions
        if q["category"] in ("visual", "cross_modal")
    ]
    print(f"Loaded {len(queries)} visual/cross_modal questions from {questions_path}")

    # ── Load image data từ ChromaDB (caption đã có sẵn) ──
    chroma_path = str(CFG["paths"]["vector_db"])
    image_col_name = CFG["embedding"].get("image_collection", "rag_image")
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    image_col = chroma_client.get_collection(image_col_name)

    img_data = image_col.get(include=["documents", "metadatas"])
    print(f"Found {len(img_data['ids'])} images in ChromaDB")

    # Build image info list — 10 mã đa dạng ngành
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
        caption = img_data["documents"][i]  # caption đã lưu
        img_path = meta.get("img_path", "")
        ticker = meta.get("ticker", "")
        if img_path and Path(img_path).exists() and ticker in TARGET_TICKERS:
            images.append({
                "chunk_id": chunk_id,
                "caption": caption,
                "img_path": img_path,
                "ticker": ticker,
                "doc": meta.get("doc", ""),
                "page": meta.get("page", 0),
            })
    print(f"Valid images ({len(TARGET_TICKERS)} tickers): {len(images)} / {len(img_data['ids'])} total")

    # ── Embed tất cả ảnh theo 3 cách (1 lần duy nhất) ──
    print(f"\nEmbedding {len(images)} images × 3 methods...")

    image_vectors = {}  # chunk_id → {caption_only, image_only, aggregated}

    for i, img in enumerate(images):
        chunk_id = img["chunk_id"]
        caption = img["caption"]
        img_path = img["img_path"]

        print(f"  [{i+1}/{len(images)}] {chunk_id}", end="", flush=True)

        vecs = {}

        # Method 1: Caption only
        try:
            vecs["caption_only"] = embedder.embed_text(caption)
            time.sleep(1.5)
        except Exception as e:
            print(f" caption_err={e}")
            vecs["caption_only"] = None

        # Method 2: Image only
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
            vecs["image_only"] = list(r.embeddings[0].values)
            time.sleep(1.5)
        except Exception as e:
            print(f" img_err={e}")
            vecs["image_only"] = None

        # Method 3: Aggregated (current approach)
        try:
            vecs["aggregated"] = embedder.embed_image(img_path, caption=caption)
            time.sleep(1.5)
        except Exception as e:
            print(f" agg_err={e}")
            vecs["aggregated"] = None

        image_vectors[chunk_id] = vecs
        print(" ✓")

    # ── Embed queries & tính similarity + rank ──
    print(f"\nScoring {len(queries)} queries...")

    METHODS = ["caption_only", "image_only", "aggregated"]
    results = []

    for qi, q in enumerate(queries):
        question = q["question"]
        qid = q["id"]
        expected_doc = q.get("doc", "")  # nếu có, dùng để tính precision

        print(f"\n  [{qid}] {question[:60]}...")

        query_vec = embedder.embed_query(question)
        time.sleep(1.5)

        # Tính similarity với mỗi ảnh, mỗi method
        for method in METHODS:
            scored = []
            for img in images:
                vec = image_vectors[img["chunk_id"]].get(method)
                if vec is None:
                    continue
                sim = cosine_sim(query_vec, vec)
                scored.append({
                    "chunk_id": img["chunk_id"],
                    "ticker": img["ticker"],
                    "doc": img["doc"],
                    "sim": sim,
                })

            # Rank by similarity (cao → thấp)
            scored.sort(key=lambda x: x["sim"], reverse=True)

            # Top-3
            top3 = scored[:3]
            top3_ids = [s["chunk_id"] for s in top3]
            top3_sims = [round(s["sim"], 5) for s in top3]
            top3_docs = [s["doc"] for s in top3]

            # Nếu biết expected doc, tính rank
            rank_of_correct = None
            if expected_doc:
                for rank, s in enumerate(scored, 1):
                    if expected_doc in s["doc"]:
                        rank_of_correct = rank
                        break

            results.append({
                "qid": qid,
                "question": question[:80],
                "method": method,
                "top1_id": top3_ids[0] if top3 else "",
                "top1_sim": top3_sims[0] if top3 else 0,
                "top1_doc": top3_docs[0] if top3 else "",
                "top3_ids": top3_ids,
                "top3_sims": top3_sims,
                "top3_docs": top3_docs,
                "rank_correct_doc": rank_of_correct,
                "expected_doc": expected_doc,
            })

        # In so sánh nhanh
        for method in METHODS:
            r = [x for x in results if x["qid"] == qid and x["method"] == method][0]
            rank_str = f"rank={r['rank_correct_doc']}" if r["rank_correct_doc"] else "n/a"
            print(f"    {method:<15} top1={r['top1_sim']:.4f} ({r['top1_doc'][:30]})  {rank_str}")

    # ── Aggregate results ──
    print(f"\n\n{'='*80}")
    print("ABLATION: Image Embedding Methods")
    print(f"{'='*80}")

    for method in METHODS:
        m_results = [r for r in results if r["method"] == method]
        avg_sim = sum(r["top1_sim"] for r in m_results) / len(m_results)
        ranks = [r["rank_correct_doc"] for r in m_results if r["rank_correct_doc"] is not None]
        avg_rank = sum(ranks) / len(ranks) if ranks else float("inf")
        top1_hit = sum(1 for r in m_results if r["rank_correct_doc"] == 1) if ranks else 0
        top3_hit = sum(1 for r in m_results if r["rank_correct_doc"] and r["rank_correct_doc"] <= 3) if ranks else 0

        tag = " ← current" if method == "aggregated" else ""
        print(f"\n  {method:<15}{tag}")
        print(f"    Avg top-1 similarity : {avg_sim:.5f}")
        if ranks:
            print(f"    Avg rank correct doc : {avg_rank:.1f}")
            print(f"    Top-1 hit rate       : {top1_hit}/{len(ranks)} ({top1_hit/len(ranks)*100:.0f}%)")
            print(f"    Top-3 hit rate       : {top3_hit}/{len(ranks)} ({top3_hit/len(ranks)*100:.0f}%)")

    # ── Per-query comparison ──
    print(f"\n{'─'*80}")
    print(f"Per-query comparison (top-1 similarity):")
    print(f"  {'QID':<6} {'Caption':>8} {'Image':>8} {'Aggreg':>8}  {'Best':<12} Question")
    print(f"  {'─'*78}")

    qids_seen = []
    for q in queries:
        qid = q["id"]
        if qid in qids_seen:
            continue
        qids_seen.append(qid)
        sims = {}
        for method in METHODS:
            r = [x for x in results if x["qid"] == qid and x["method"] == method]
            sims[method] = r[0]["top1_sim"] if r else 0

        best = max(sims, key=sims.get)
        print(f"  {qid:<6} {sims['caption_only']:>8.4f} {sims['image_only']:>8.4f} {sims['aggregated']:>8.4f}  {best:<12} {q['question'][:40]}")

    # Win rate
    print(f"\n{'─'*80}")
    agg_wins_caption = 0
    agg_wins_image = 0
    total_qs = 0
    for qid in qids_seen:
        sc = [x for x in results if x["qid"] == qid and x["method"] == "caption_only"]
        si = [x for x in results if x["qid"] == qid and x["method"] == "image_only"]
        sa = [x for x in results if x["qid"] == qid and x["method"] == "aggregated"]
        if sc and sa:
            total_qs += 1
            if sa[0]["top1_sim"] >= sc[0]["top1_sim"]:
                agg_wins_caption += 1
            if si and sa[0]["top1_sim"] >= si[0]["top1_sim"]:
                agg_wins_image += 1

    print(f"Win rate (aggregated ≥ other):")
    print(f"  vs caption_only : {agg_wins_caption}/{total_qs} ({agg_wins_caption/total_qs*100:.0f}%)" if total_qs else "  n/a")
    print(f"  vs image_only   : {agg_wins_image}/{total_qs} ({agg_wins_image/total_qs*100:.0f}%)" if total_qs else "  n/a")

    # ── Save ──
    out_path = Path("results/ablation_embed_image.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Detailed results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation: Image Embedding Methods")
    parser.add_argument("--questions", default="src/eval/questions.json")
    args = parser.parse_args()
    run(args.questions)
