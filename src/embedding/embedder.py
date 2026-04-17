# src/embedding/embedder.py

"""
Dual-Pipeline Embedding cho Multimodal RAG (MDocAgent-style).

2 pipelines hoàn toàn tách biệt:
    Text Pipeline:  text + table chunks → Gemini text embedding → rag_text collection
    Image Pipeline: image chunks → auto-caption + aggregated embedding → rag_image collection

Mỗi pipeline có ChromaDB collection riêng. KHÔNG chia sẻ vector space.
"""

import json
import time
import sys
from pathlib import Path

import chromadb
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import CFG

# Model dùng để auto-caption charts trước khi embed
# gemini-2.0-flash đã deprecated → dùng gemini-2.5-flash-lite
VISION_MODEL = "gemini-2.5-flash"

CAPTION_PROMPT = (
    "Bạn là trợ lý phân tích báo cáo tài chính.\n"
    "Mô tả biểu đồ/hình ảnh này bằng tiếng Việt (3-5 câu).\n"
    "Nêu rõ:\n"
    "(1) Tiêu đề biểu đồ và loại biểu đồ (đường/cột/tròn)\n"
    "(2) Các chỉ số trên trục (tên trục, đơn vị, khoảng giá trị)\n"
    "(3) Các đường/cột chính và legend (tên, màu sắc nếu có)\n"
    "(4) Xu hướng chính, điểm cực đại/cực tiểu, giá trị cụ thể đọc được\n"
    "(5) Mã cổ phiếu nếu có. Nêu cả tiếng Việt lẫn tiếng Anh cho thuật ngữ "
    "(VD: biên lợi nhuận gộp/GPM, giá mục tiêu/Target Price, khuyến nghị/recommendation)."
)


# ======================================================
# Gemini Embedder
# ======================================================

class GeminiEmbedder:
    """
    Embedding client dùng Gemini Embedding 2.
    Hỗ trợ cả text embedding và multimodal (caption + image) embedding.
    Tự động retry khi gặp lỗi 503/429 (API quá tải).
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-2-preview",
                 sleep_seconds: float = 1.5, max_retries: int = 5):
        from google import genai
        from google.genai import types
        self._genai = genai
        self._types = types
        self.model = model
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key)

    def _retry(self, fn, *args, **kwargs):
        """Retry wrapper: exponential backoff cho 503/429."""
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                err_str = str(e)
                retryable = any(code in err_str for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"])
                if not retryable or attempt == self.max_retries - 1:
                    raise
                wait = min(2 ** attempt * 2, 60)  # 2s, 4s, 8s, 16s, 32s
                print(f"    ⚠ API error (attempt {attempt+1}/{self.max_retries}), retry in {wait}s...")
                time.sleep(wait)

    # ── Text methods ──────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Embed một đoạn text."""
        result = self._retry(
            self.client.models.embed_content, model=self.model, contents=text,
        )
        return list(result.embeddings[0].values)

    def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch texts trong 1 API call."""
        if not texts:
            return []
        result = self._retry(
            self.client.models.embed_content, model=self.model, contents=texts,
        )
        return [list(e.values) for e in result.embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed query string. Dùng cho cả text và image retrieval."""
        result = self._retry(
            self.client.models.embed_content, model=self.model, contents=query,
        )
        return list(result.embeddings[0].values)

    # ── Image methods ─────────────────────────────────

    def caption_image(self, image_path: str) -> str:
        """
        Tự động sinh caption cho chart/figure bằng Gemini Vision Flash.

        Trả về mô tả tiếng Việt 2-3 câu, ví dụ:
          "Biểu đồ đường thể hiện giá cổ phiếu VNM từ tháng 12/2025 đến 2/2026.
           Giá dao động trong khoảng 55.000-72.000 đồng, với xu hướng tăng nhẹ."
        """
        img_path = Path(image_path)
        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        mime_type = _get_mime(img_path)

        def _call():
            return self.client.models.generate_content(
                model=VISION_MODEL,
                contents=[
                    self._types.Content(
                        parts=[
                            self._types.Part(text=CAPTION_PROMPT),
                            self._types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type,
                            ),
                        ]
                    )
                ],
            )

        try:
            response = self._retry(_call)
            return response.text.strip()
        except Exception:
            return f"Biểu đồ tài chính ({img_path.stem})"

    # def embed_image(self, image_path: str, caption: str = "") -> list[float]:
    #     """
    #     Embed ảnh bằng aggregated embedding: caption + image bytes → 1 vector.

    #     Khi gửi (text_part, image_part) trong cùng 1 Content entry,
    #     model trả về 1 embedding kết hợp cả ngữ nghĩa text lẫn visual.
    #     """
    #     img_path = Path(image_path)
    #     if not img_path.exists():
    #         raise FileNotFoundError(f"Ảnh không tồn tại: {image_path}")

    #     with open(img_path, 'rb') as f:
    #         image_bytes = f.read()

    #     mime_type = _get_mime(img_path)

    #     parts = []
    #     if caption:
    #         parts.append(self._types.Part(text=caption))
    #     parts.append(
    #         self._types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    #     )

    #     result = self._retry(
    #         self.client.models.embed_content,
    #         model=self.model,
    #         contents=[self._types.Content(parts=parts)],
    #     )
    #     return list(result.embeddings[0].values)

    def embed_image(self, image_path: str, caption: str = "", alpha: float = 0.8) -> list[float]:
        """
        Embed ảnh bằng weighted blend: α × caption_vec + (1-α) × image_vec.
        
        Thay vì gửi cùng nhau cho Gemini tự blend (α≈0.2 không kiểm soát được),
        tự blend với α=0.8 — tốt hơn 22.3% theo ablation.
        
        Args:
            image_path: Đường dẫn tới file ảnh
            caption:    Caption mô tả ảnh (nếu có)
            alpha:      Trọng số caption (0.0=image_only, 1.0=caption_only, default=0.8)
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Ảnh không tồn tại: {image_path}")

        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        mime_type = _get_mime(img_path)

        # ── Call 1: Caption vector (text only) ──
        if caption:
            caption_vec = self.embed_text(caption)
            time.sleep(self.sleep_seconds)
        else:
            caption_vec = None

        # ── Call 2: Image vector (image bytes only, không có caption) ──
        parts = [self._types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        result = self._retry(
            self.client.models.embed_content,
            model=self.model,
            contents=[self._types.Content(parts=parts)],
        )
        image_vec = list(result.embeddings[0].values)

        # ── Blend: α × caption + (1-α) × image, rồi L2-normalize ──
        if caption_vec is None:
            return image_vec  # fallback nếu không có caption

        blended = [alpha * c + (1 - alpha) * v for c, v in zip(caption_vec, image_vec)]
        norm = sum(x * x for x in blended) ** 0.5
        if norm == 0:
            return blended
        return [x / norm for x in blended]


# ======================================================
# ChromaDB helpers
# ======================================================

def load_or_create_collection(
    chroma_path: str,
    collection_name: str,
) -> chromadb.Collection:
    """Khởi tạo hoặc load ChromaDB persistent collection."""
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ======================================================
# Main: Dual-Pipeline Embed & Store
# ======================================================

def embed_and_store(
    metadata_path: str,
    chroma_path: str,
    api_key: str,
    cfg: dict,
) -> dict:
    """
    Đọc metadata.json → embed vào 2 collections riêng biệt.

    Pipeline 1 (Text): text + table → rag_text collection
    Pipeline 2 (Image): image → auto-caption → aggregated embed → rag_image collection

    Returns: dict thống kê
    """
    embed_cfg       = cfg["embedding"]
    batch_size      = embed_cfg.get("batch_size", 10)
    sleep_sec       = embed_cfg.get("sleep_between_batches", 1.5)
    min_token_text  = embed_cfg.get("min_token_count", 50)
    text_col_name   = embed_cfg.get("text_collection", "rag_text")
    image_col_name  = embed_cfg.get("image_collection", "rag_image")
    model_name      = embed_cfg.get("model", "gemini-embedding-2-preview")

    embedder = GeminiEmbedder(api_key=api_key, model=model_name, sleep_seconds=sleep_sec)
    text_collection  = load_or_create_collection(chroma_path, text_col_name)
    image_collection = load_or_create_collection(chroma_path, image_col_name)

    print(f"\nDual-Pipeline Embedding (MDocAgent-style)")
    print(f"  Model       : {model_name}")
    print(f"  Text col    : {text_col_name}")
    print(f"  Image col   : {image_col_name}")

    with open(metadata_path, encoding="utf-8") as f:
        all_entries: list = json.load(f)

    # --- Phân loại chunks ---
    text_entries:  list = []
    table_entries: list = []
    image_entries: list = []

    for entry in all_entries:
        chunk_type = entry.get("chunk_type", "")
        if chunk_type == "image":
            if entry.get("img_path") and Path(entry["img_path"]).exists():
                image_entries.append(entry)
        elif chunk_type == "table":
            table_entries.append(entry)
        elif chunk_type == "text":
            if entry.get("token_count", 0) >= min_token_text:
                text_entries.append(entry)

    n_skip = sum(1 for e in all_entries
                 if e.get("chunk_type") == "text"
                 and e.get("token_count", 0) < min_token_text)

    print(f"\nChunks:")
    print(f"  Text   : {len(text_entries):3d}  (skip {n_skip} quá ngắn)")
    print(f"  Table  : {len(table_entries):3d}")
    print(f"  Image  : {len(image_entries):3d}")

    # ═══════════════════════════════════════════════════
    # PIPELINE 1: Text + Table → rag_text
    # ═══════════════════════════════════════════════════
    all_text_entries = text_entries + table_entries

    if all_text_entries:
        print(f"\n{'='*50}")
        print(f"PIPELINE 1: Text + Table → {text_col_name}")
        print(f"{'='*50}")
        print(f"  Embedding {len(all_text_entries)} chunks...")

        contents: list[str] = [
            Path(e["text_path"]).read_text(encoding="utf-8")
            for e in tqdm(all_text_entries, desc="  Read")
        ]

        all_vectors: list[list[float]] = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i: i + batch_size]
            vectors = embedder.embed_texts_batch(batch)
            all_vectors.extend(vectors)
            if i + batch_size < len(contents):
                time.sleep(sleep_sec)

        text_collection.upsert(
            ids=[e["chunk_id"] for e in all_text_entries],
            embeddings=all_vectors,
            documents=contents,
            metadatas=[_build_meta(e) for e in all_text_entries],
        )
        print(f"  ✓ Upserted {len(all_text_entries)} vectors → {text_col_name}")

    # ═══════════════════════════════════════════════════
    # PIPELINE 2: Image → auto-caption → rag_image
    # ═══════════════════════════════════════════════════
    if image_entries:
        print(f"\n{'='*50}")
        print(f"PIPELINE 2: Image → {image_col_name}")
        print(f"{'='*50}")

        # ── Resume: skip ảnh đã embed trong ChromaDB ──
        existing_ids = set(image_collection.get(include=[])["ids"])
        todo_entries = [e for e in image_entries if e["chunk_id"] not in existing_ids]
        n_skip_img = len(image_entries) - len(todo_entries)
        if n_skip_img > 0:
            print(f"  ⏩ Skip {n_skip_img} images đã embed, còn {len(todo_entries)} cần xử lý")
        else:
            print(f"  Auto-caption + aggregated embed ({len(todo_entries)} images)...")

        img_ids, img_vectors, img_docs, img_metas = [], [], [], []

        for entry in tqdm(todo_entries, desc="  Caption+Embed"):
            try:
                # Step 1: Auto-caption bằng Gemini Vision
                rich_caption = embedder.caption_image(entry["img_path"])
                time.sleep(0.5)

                # Step 2: Aggregated embedding (caption + image → 1 vector)
                vector = embedder.embed_image(entry["img_path"], caption=rich_caption)

            except Exception as e:
                print(f"\n  Error [{entry['chunk_id']}]: {e}")
                continue

            entry["caption"] = rich_caption

            img_ids.append(entry["chunk_id"])
            img_vectors.append(vector)
            img_docs.append(rich_caption)
            img_metas.append(_build_meta(entry))

            time.sleep(sleep_sec)

        if img_ids:
            image_collection.upsert(
                ids=img_ids,
                embeddings=img_vectors,
                documents=img_docs,
                metadatas=img_metas,
            )
            print(f"  ✓ Upserted {len(img_ids)} vectors → {image_col_name}")

    # --- Summary ---
    n_text  = text_collection.count()
    n_image = image_collection.count()
    print(f"\nDone.")
    print(f"  {text_col_name:15s}: {n_text} entries")
    print(f"  {image_col_name:15s}: {n_image} entries")

    return {
        "n_text":  len(text_entries),
        "n_table": len(table_entries),
        "n_image": len(image_entries),
        "text_collection_total":  n_text,
        "image_collection_total": n_image,
    }


# ======================================================
# Helpers
# ======================================================

def _get_mime(path: Path) -> str:
    """Xác định MIME type từ extension."""
    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}
    return mime_map.get(path.suffix.lower(), 'image/png')


def _build_meta(entry: dict) -> dict:
    """Build ChromaDB metadata dict (chỉ str/int/float)."""
    return {
        "chunk_type":  str(entry.get("chunk_type", "")),
        "doc":         str(entry.get("doc", "")),
        "page":        int(entry.get("page", 0)),
        "ticker":      str(entry.get("ticker", "")),
        "report_date": str(entry.get("report_date", "")),
        "source":      str(entry.get("source", "")),
        "page_type":   str(entry.get("page_type", "")),
        "table_name":  str(entry.get("table_name", "")),
        "img_path":    str(entry.get("img_path") or ""),
        "caption":     str(entry.get("caption") or ""),
        "token_count": int(entry.get("token_count", 0)),
    }
