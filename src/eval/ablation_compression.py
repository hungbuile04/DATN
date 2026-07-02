# src/eval/ablation_compression.py

"""
Ablation Study: Nén Vector Embedding (Matryoshka Truncation + Quantization).

Gemini Embedding 2 hỗ trợ Matryoshka Embedding — có thể cắt N chiều đầu tiên
từ vector 3072-dim mà vẫn giữ chất lượng tốt.

Thí nghiệm so sánh:
  1. float32 × 3072-dim (baseline — hiện tại)
  2. float32 × 1536-dim (Matryoshka truncation 50%)
  3. float32 × 768-dim  (Matryoshka truncation 25%)
  4. float32 × 256-dim  (Matryoshka truncation 8%)
  5. float16 × 3072-dim (quantization — giữ nguyên dim, giảm precision)

Metrics: Precision@k, Hit Rate@k, MRR, Storage (bytes/vector)

Cách hoạt động (KHÔNG cần re-embed, KHÔNG tốn API):
  - Lấy toàn bộ embeddings từ ChromaDB (đã lưu sẵn)
  - Truncate hoặc quantize offline
  - Tính cosine similarity thủ công
  - Đánh giá retrieval metrics

Usage:
    python -m src.eval.ablation_compression
    python -m src.eval.ablation_compression --questions src/eval/questions.json
"""

import argparse
import json

import os
import sys
import time

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CFG
from src.retrieval.retriever import DualRetriever, RetrievedChunk

# Import metrics
from src.eval.eval_retrieval import (
    precision_at_k, hit_rate_at_k, mrr,
)


# ══════════════════════════════════════════════
# Compression Strategies
# ══════════════════════════════════════════════

@dataclass
class CompressionConfig:
    name: str
    dim: int            # output dimension (truncation)
    dtype: str          # "float32" hoặc "float16"
    bytes_per_vector: int
    description: str


STRATEGIES = [
    CompressionConfig("float32_3072", 3072, "float32", 3072 * 4, "Baseline (hiện tại)"),
    CompressionConfig("float32_1536", 1536, "float32", 1536 * 4, "Matryoshka 50% dim"),
    CompressionConfig("float32_768",  768,  "float32", 768 * 4,  "Matryoshka 25% dim"),
    CompressionConfig("float32_256",  256,  "float32", 256 * 4,  "Matryoshka 8% dim"),
    CompressionConfig("float16_3072", 3072, "float16", 3072 * 2, "Quantize float16"),
]


# ══════════════════════════════════════════════
# Vector Operations
# ══════════════════════════════════════════════

def truncate_vec(vec: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation: lấy N chiều đầu tiên."""
    return vec[:dim]


def quantize_float16(vec: np.ndarray) -> np.ndarray:
    """Quantize float32 → float16 → float32 (simulate precision loss)."""
    return vec.astype(np.float16).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity giữa 2 vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compress_vector(vec: np.ndarray, config: CompressionConfig) -> np.ndarray:
    """Áp dụng compression strategy lên 1 vector."""
    # 1. Truncate dimension
    compressed = truncate_vec(vec, config.dim)
    # 2. Quantize nếu cần
    if config.dtype == "float16":
        compressed = quantize_float16(compressed)
    # 3. L2-normalize lại sau khi truncate
    norm = np.linalg.norm(compressed)
    if norm > 0:
        compressed = compressed / norm
    return compressed


# ══════════════════════════════════════════════
# Load embeddings từ ChromaDB
# ══════════════════════════════════════════════

def load_collection_data(retriever: DualRetriever, collection_name: str):
    """
    Load toàn bộ embeddings + metadata từ 1 ChromaDB collection.
    Returns: dict { chunk_id: { "embedding": np.array, "metadata": dict, "document": str } }
    """
    col = retriever.text_col if collection_name == "text" else retriever.image_col
    data = col.get(include=["embeddings", "metadatas", "documents"])

    result = {}
    for i, chunk_id in enumerate(data["ids"]):
        result[chunk_id] = {
            "embedding": np.array(data["embeddings"][i], dtype=np.float32),
            "metadata": data["metadatas"][i],
            "document": data["documents"][i] if data["documents"] else "",
        }
    return result


# ══════════════════════════════════════════════
# Retrieval với compressed vectors
# ══════════════════════════════════════════════

def retrieve_compressed(
    query_vec: np.ndarray,
    corpus: dict,
    config: CompressionConfig,
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k từ corpus dùng compressed vectors.
    Tính cosine similarity thủ công (không qua ChromaDB).
    """
    # Compress query
    q_compressed = compress_vector(query_vec, config)

    # Score tất cả chunks
    scores = []
    for chunk_id, chunk_data in corpus.items():
        doc_compressed = compress_vector(chunk_data["embedding"], config)
        sim = cosine_similarity(q_compressed, doc_compressed)
        scores.append((chunk_id, sim, chunk_data))

    # Sort giảm dần
    scores.sort(key=lambda x: x[1], reverse=True)

    # Build RetrievedChunks
    results = []
    for chunk_id, score, chunk_data in scores[:top_k]:
        meta = chunk_data["metadata"]
        results.append(RetrievedChunk(
            chunk_id=chunk_id,
            content=chunk_data["document"],
            chunk_type=meta.get("chunk_type", "text"),
            score=score,
            page=int(meta.get("page", 0)),
            doc=meta.get("doc", ""),
            img_path=meta.get("img_path", ""),
            has_chart=bool(meta.get("img_path")),
            caption=meta.get("caption", ""),
        ))
    return results


# ══════════════════════════════════════════════
# Main Evaluation
# ══════════════════════════════════════════════

def run_ablation(retriever: DualRetriever, questions: list[dict],
                 text_top_k: int = 6, image_top_k: int = 4):
    """Chạy ablation study cho tất cả compression strategies."""

    print("\n" + "="*70)
    print("ABLATION STUDY: Vector Compression")
    print("="*70)

    # Load toàn bộ embeddings 1 lần
    print("\n📦 Loading embeddings from ChromaDB...")
    text_corpus = load_collection_data(retriever, "text")
    image_corpus = load_collection_data(retriever, "image")
    print(f"  Text corpus:  {len(text_corpus)} chunks")
    print(f"  Image corpus: {len(image_corpus)} chunks")

    # Embed tất cả queries 1 lần (full 3072-dim)
    print(f"\n🔤 Embedding {len(questions)} queries...")
    query_vecs = {}
    for i, q in enumerate(questions):
        qid = q["id"]
        if qid not in query_vecs:
            try:
                vec = retriever._embed_query(q["question"])
                query_vecs[qid] = np.array(vec, dtype=np.float32)
            except Exception as e:
                print(f"  ⚠ Embed error for {qid}: {e}")
                time.sleep(10)
                continue
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(questions)} queries embedded")
    print(f"  ✓ {len(query_vecs)} queries embedded")

    # Đánh giá từng strategy
    total_k = text_top_k + image_top_k
    all_results = {}

    for config in STRATEGIES:
        print(f"\n{'─'*60}")
        print(f"📐 Strategy: {config.name} — {config.description}")
        print(f"   Dim={config.dim}, dtype={config.dtype}, "
              f"bytes/vec={config.bytes_per_vector:,}")
        print(f"{'─'*60}")

        metrics = {"p@k": [], "hit@k": [], "mrr": []}

        for i, q in enumerate(questions):
            qid = q["id"]
            doc = q.get("doc", "")
            requires = q.get("requires", [])

            if qid not in query_vecs or not doc:
                continue

            query_vec = query_vecs[qid]

            # Text retrieval
            text_chunks = retrieve_compressed(
                query_vec, text_corpus, config, text_top_k
            )
            # Image retrieval
            image_chunks = retrieve_compressed(
                query_vec, image_corpus, config, image_top_k
            )

            all_chunks = text_chunks + image_chunks

            # Metrics
            metrics["p@k"].append(precision_at_k(all_chunks, doc, requires, total_k))
            metrics["hit@k"].append(hit_rate_at_k(all_chunks, doc, requires, total_k))
            metrics["mrr"].append(mrr(all_chunks, doc, requires))

            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(questions)} questions evaluated")

        # Aggregate
        n = len(metrics["p@k"])
        if n == 0:
            continue

        avg_p = sum(metrics["p@k"]) / n
        avg_h = sum(metrics["hit@k"]) / n
        avg_m = sum(metrics["mrr"]) / n

        # Storage estimation
        total_vecs = len(text_corpus) + len(image_corpus)
        storage_mb = (total_vecs * config.bytes_per_vector) / (1024 * 1024)
        compression_ratio = STRATEGIES[0].bytes_per_vector / config.bytes_per_vector

        all_results[config.name] = {
            "dim": config.dim,
            "dtype": config.dtype,
            "precision_at_k": round(avg_p, 4),
            "hit_rate_at_k": round(avg_h, 4),
            "mrr": round(avg_m, 4),
            "n": n,
            "bytes_per_vector": config.bytes_per_vector,
            "storage_mb": round(storage_mb, 2),
            "compression_ratio": f"{compression_ratio:.1f}x",
            "description": config.description,
        }

        print(f"\n  ✓ {config.name}: P@k={avg_p:.4f} Hit@k={avg_h:.4f} "
              f"MRR={avg_m:.4f} | Storage={storage_mb:.2f}MB ({compression_ratio:.1f}x)")

    # ── Bảng tổng hợp ──
    print(f"\n\n{'═'*80}")
    print(f"BẢNG TỔNG HỢP: Vector Compression Ablation")
    print(f"{'═'*80}")
    print(f"  {'Strategy':<20} {'Dim':>5} {'DType':>8} {'P@k':>8} {'Hit@k':>8} "
          f"{'MRR':>8} {'Storage':>10} {'Nén':>6}")
    print(f"  {'─'*75}")

    for name, r in all_results.items():
        marker = " ◀ baseline" if r["dim"] == 3072 and r["dtype"] == "float32" else ""
        print(f"  {name:<20} {r['dim']:>5} {r['dtype']:>8} {r['precision_at_k']:>8.4f} "
              f"{r['hit_rate_at_k']:>8.4f} {r['mrr']:>8.4f} "
              f"{r['storage_mb']:>8.2f}MB {r['compression_ratio']:>6}{marker}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study: Vector Compression (Matryoshka + Quantization)"
    )
    parser.add_argument("--questions", default="src/eval/questions.json")
    parser.add_argument("--output", default="results/ablation_compression.json")
    parser.add_argument("--text_top_k", type=int, default=6)
    parser.add_argument("--image_top_k", type=int, default=4)
    args = parser.parse_args()

    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_key:
        raise EnvironmentError("Thiếu GOOGLE_API_KEY trong .env")

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {args.questions}")

    retriever = DualRetriever(
        chroma_path=str(CFG["paths"]["vector_db"]),
        api_key=google_key,
        cfg=CFG,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "vector_compression_ablation",
        "config": {
            "text_top_k": args.text_top_k,
            "image_top_k": args.image_top_k,
            "n_questions": len(questions),
            "n_text_chunks": retriever.text_col.count(),
            "n_image_chunks": retriever.image_col.count(),
        },
        "strategies": run_ablation(
            retriever, questions, args.text_top_k, args.image_top_k
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved → {out}")


if __name__ == "__main__":
    main()
