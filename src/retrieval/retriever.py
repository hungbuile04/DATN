# src/retrieval/retriever.py

"""
DualRetriever — Dual-Pipeline Retrieval (MDocAgent-style).

2 pipelines hoàn toàn tách biệt:
    Text Pipeline:  Dense (Gemini Embedding 2) → ticker boost → top-k
    Image Pipeline: Dense → ticker hard filter → top-k

KHÔNG merge scores giữa 2 pipelines.
Query encode 1 lần → search trên 2 collections riêng.

Ticker-aware: tự động detect mã cổ phiếu trong query,
    ưu tiên kết quả đúng ticker (image: hard filter, text: score boost).

Note: BM25+RRF đã bị loại bỏ sau thí nghiệm eval_retrieval cho thấy
    Dense-only (Gemini Embedding 2, 3072-dim) vượt trội Hybrid RRF
    về cả Precision@k (+10%), Hit Rate@k (+20%), MRR (+22%).
    BM25 tokenizer (str.split) yếu cho tiếng Việt → thêm nhiễu.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import CFG


# ──────────────────────────────────────────────
# Output dataclasses
# ──────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:   str
    content:    str      # text/table content hoặc rich caption (image)
    chunk_type: str      # "text" | "table" | "image"
    score:      float    # RRF score trong pipeline tương ứng
    page:       int
    doc:        str
    img_path:   str      # đường dẫn ảnh (chỉ image chunks)
    has_chart:  bool     # True nếu chunk_type == "image"
    caption:    str      # rich caption từ Gemini Vision


@dataclass
class RetrievalResult:
    """Kết quả retrieval từ 2 pipelines riêng biệt."""
    text_chunks:  list[RetrievedChunk] = field(default_factory=list)
    image_chunks: list[RetrievedChunk] = field(default_factory=list)


# ──────────────────────────────────────────────
# DualRetriever
# ──────────────────────────────────────────────

class DualRetriever:
    """
    Dual-pipeline retriever: text và image riêng biệt.

    Text pipeline:
        Dense (cosine similarity) → ticker boost x1.5 → [Reranker] → top-k

    Image pipeline:
        Dense (cosine similarity) → ticker hard filter → [Reranker] → top-k
        Fallback: nếu filter ra ít kết quả, bổ sung từ search không filter

    Query encode 1 lần bằng Gemini Embedding 2, search trên 2 collections.
    KHÔNG merge scores giữa 2 pipelines.

    Reranker (optional):
        Gemini Flash cross-encoder scoring — chèn giữa dense search và top-k.
        Bật/tắt qua use_reranker flag.
    """

    def __init__(self, chroma_path: str, api_key: str, cfg: dict,
                 openrouter_key: str = "", use_reranker: bool = False):
        embed_cfg = cfg["embedding"]

        model_name      = embed_cfg.get("model", "gemini-embedding-2-preview")
        text_col_name   = embed_cfg.get("text_collection", "rag_text")
        image_col_name  = embed_cfg.get("image_collection", "rag_image")

        # Gemini client cho query embedding
        from google import genai
        self._genai_client = genai.Client(api_key=api_key)
        self._embed_model = model_name

        # ChromaDB collections
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.text_col  = chroma_client.get_collection(text_col_name)
        self.image_col = chroma_client.get_collection(image_col_name)

        # Reranker (optional)
        self.use_reranker = use_reranker
        self._reranker = None
        if use_reranker and openrouter_key:
            from src.retrieval.reranker import GeminiReranker
            reranker_cfg = cfg.get("retrieval", {})
            self._reranker = GeminiReranker(
                api_key=openrouter_key,
                model=reranker_cfg.get("reranker_model", "google/gemini-2.5-flash"),
            )
            print("  ✓ Reranker enabled (Gemini Flash)")

        # Known tickers — populated during _load_corpus
        self._known_tickers: set[str] = set()
        # Query embedding cache: avoid re-embedding identical queries
        self._query_cache: dict[str, list[float]] = {}

        self._load_corpus()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    # Ngưỡng tối thiểu cho image RRF score.
    # Giảm từ 0.02 → 0.015 sau khi fix captions: caption tốt hơn nhưng
    # discriminative hơn → score tuyệt đối giảm nhẹ. 0.02 lọc mất ảnh đúng.
    IMAGE_MIN_SCORE = 0.015

    def retrieve(
        self,
        query: str,
        text_top_k: int = 6,
        image_top_k: int = 4,
        unified_top_k: int = 0,
    ) -> RetrievalResult:
        """
        Truy vấn dual pipeline và trả về kết quả riêng biệt.

        Hai chế độ:
            - Cố định (mặc định): text_top_k=6, image_top_k=4
            - Thích ứng: unified_top_k > 0 → query cả 2 collection,
              gộp theo score, lấy top-k chung → tự phân bổ tỷ lệ
              text/image theo mức độ liên quan của câu hỏi.

        Ticker detection 2 tầng:
            1. Detect trực tiếp từ query ("VNM", "Vinamilk")
            2. Fallback: suy luận từ text results (majority vote)

        Args:
            query:          câu hỏi người dùng
            text_top_k:     top-k cho text pipeline (chế độ cố định)
            image_top_k:    top-k cho image pipeline (chế độ cố định)
            unified_top_k:  nếu > 0, dùng chế độ thích ứng

        Returns:
            RetrievalResult với text_chunks và image_chunks riêng biệt
        """
        # Encode query 1 lần duy nhất
        query_vec = self._embed_query(query)

        # Detect tickers từ query
        tickers = self._detect_tickers(query)
        if tickers:
            print(f"  🏷  Detected tickers (query): {tickers}")

        # ── Chế độ thích ứng (unified) ──
        if unified_top_k > 0:
            return self._unified_retrieve(
                query, query_vec, tickers, unified_top_k,
            )

        # ── Chế độ cố định (legacy) ──
        # === TEXT PIPELINE (chạy trước) ===
        text_chunks = []
        if text_top_k > 0:
            text_chunks = self._text_pipeline(query, query_vec, text_top_k, tickers)

        # === Ticker fallback: suy luận từ text results ===
        if not tickers and text_chunks:
            tickers = self._infer_tickers_from_chunks(text_chunks)
            if tickers:
                print(f"  🏷  Inferred tickers (text): {tickers}")

        # === IMAGE PIPELINE ===
        image_chunks = []
        if image_top_k > 0:
            image_chunks = self._image_pipeline(query, query_vec, image_top_k, tickers)
            image_chunks = [
                c for c in image_chunks if c.score >= self.IMAGE_MIN_SCORE
            ]

        return RetrievalResult(
            text_chunks=text_chunks,
            image_chunks=image_chunks,
        )

    def _unified_retrieve(
        self,
        query: str,
        query_vec: list[float],
        tickers: list[str],
        total_k: int,
    ) -> RetrievalResult:
        """
        Truy xuất thích ứng: lấy dư từ cả 2 collection,
        chuẩn hoá score theo từng luồng, gộp và chọn top-k chung.

        Score chuẩn hoá min-max trong mỗi luồng (text, image) để
        so sánh công bằng giữa 2 không gian embedding khác nhau.
        """
        # Lấy dư (total_k mỗi bên) để có đủ ứng viên
        text_chunks = self._text_pipeline(query, query_vec, total_k, tickers)

        # Ticker fallback
        if not tickers and text_chunks:
            tickers = self._infer_tickers_from_chunks(text_chunks)
            if tickers:
                print(f"  🏷  Inferred tickers (text): {tickers}")

        image_chunks = self._image_pipeline(query, query_vec, total_k, tickers)
        image_chunks = [
            c for c in image_chunks if c.score >= self.IMAGE_MIN_SCORE
        ]

        # ── Chuẩn hoá score min-max trong mỗi luồng ──
        def _normalize(chunks):
            if not chunks:
                return []
            scores = [c.score for c in chunks]
            s_min, s_max = min(scores), max(scores)
            spread = s_max - s_min if s_max > s_min else 1.0
            result = []
            for c in chunks:
                norm_score = (c.score - s_min) / spread
                result.append((c, norm_score))
            return result

        norm_text = _normalize(text_chunks)
        norm_image = _normalize(image_chunks)

        # Gộp và sort theo score chuẩn hoá giảm dần
        all_chunks = [(c, ns, "text") for c, ns in norm_text] + \
                     [(c, ns, "image") for c, ns in norm_image]
        all_chunks.sort(key=lambda x: x[1], reverse=True)

        # Lấy top-k chung, tách lại thành text/image
        final_text = []
        final_image = []
        for chunk, norm_score, ctype in all_chunks[:total_k]:
            if ctype == "text":
                final_text.append(chunk)
            else:
                final_image.append(chunk)

        n_text = len(final_text)
        n_image = len(final_image)
        total = n_text + n_image
        pct_t = (n_text / total * 100) if total > 0 else 0
        pct_i = (n_image / total * 100) if total > 0 else 0
        print(f"  📊 Unified top-{total_k}: {n_text} text + {n_image} image "
              f"(tự phân bổ {pct_t:.0f}%/{pct_i:.0f}%)")

        return RetrievalResult(
            text_chunks=final_text,
            image_chunks=final_image,
        )

    # ──────────────────────────────────────────
    # Private: Ticker detection
    # ──────────────────────────────────────────

    # Mapping tên công ty / tên thường gọi → mã cổ phiếu.
    # Dùng khi query viết "Vinamilk" thay vì "VNM", "Hòa Phát" thay vì "HPG", v.v.
    _COMPANY_ALIASES: dict[str, str] = {
        "vinamilk":       "VNM",
        "sữa việt nam":   "VNM",
        "hòa phát":       "HPG",
        "hoa phat":       "HPG",
        "masan":          "MSN",
        "thế giới di động": "MWG",
        "the gioi di dong": "MWG",
        "digiworld":      "DGW",
        "coteccons":      "CTD",
        "vietcombank":    "VCB",
        "hdb":            "HDB",
        "hdbank":         "HDB",
        "techcombank":    "TPB",
        "vietinbank":     "CTG",
        "pv gas":         "GAS",
        "pvgas":          "GAS",
        "petrovietnam gas": "GAS",
        "sabeco":         "SAB",
        "bia sài gòn":    "SAB",
        "ree":            "REE",
        "nam long":       "NLG",
        "nhà khang điền": "KDH",
        "khang điền":     "KDH",
        "khang dien":     "KDH",
        "hưng thịnh incons": "HT1",
        "gemadept":       "GMD",
        "pnj":            "PNJ",
        "phú nhuận":      "PNJ",
        "phu nhuan":      "PNJ",
        "nhựa tiền phong": "NTP",
        "pvtrans":        "PVT",
        "hải an":         "HAH",
        "hai an":         "HAH",
        "hà đô":          "HDG",
        "ha do":          "HDG",
        "vingroup":       "VIC",
        "tập đoàn vingroup": "VIC",
        "vinhomes":       "VHM",
        "petrolimex":     "PLX",
        "xăng dầu":       "PLX",
    }

    def _detect_tickers(self, query: str) -> list[str]:
        """
        Detect mã cổ phiếu trong query.

        Hai bước:
            1. Tìm mã ticker chính xác (VNM, HPG, ...) — ưu tiên dài hơn
            2. Fallback: tìm tên công ty / alias (Vinamilk, Hòa Phát, ...)

        Returns:
            Danh sách các Ticker string (e.g. ["VNM", "HPG"]) hoặc [] nếu không tìm thấy.
        """
        q_upper = query.upper()
        detected = set()

        # Bước 1: match mã ticker trực tiếp
        for ticker in sorted(self._known_tickers, key=len, reverse=True):
            if re.search(r'\b' + re.escape(ticker) + r'\b', q_upper):
                detected.add(ticker)

        # Bước 2: match tên công ty / alias (case-insensitive)
        q_lower = query.lower()
        for alias, ticker in sorted(
            self._COMPANY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if alias in q_lower:
                detected.add(ticker)

        return list(detected)

    def _infer_tickers_from_chunks(self, chunks: list[RetrievedChunk]) -> list[str]:
        """
        Suy luận ticker từ text results bằng majority vote.

        Khi query không nhắc ticker (vd: "tỷ lệ nợ xấu gần đây"),
        nhưng text results đa số từ cùng 1 doc (vd: HDB) → suy luận
        đó là ticker chính → dùng filter cho image pipeline.

        Chỉ trả về ticker nếu chiếm ≥ 50% top results.
        """
        from collections import Counter
        tickers = [_ticker_of(c.chunk_id) for c in chunks if _ticker_of(c.chunk_id)]
        if not tickers:
            return []
        counter = Counter(tickers)
        top_ticker, top_count = counter.most_common(1)[0]
        if top_count / len(tickers) >= 0.5:
            return [top_ticker]
        return []

    # ──────────────────────────────────────────
    # Private: Pipelines
    # ──────────────────────────────────────────

    def _text_pipeline(
        self, query: str, query_vec: list[float], top_k: int, tickers: list[str] = None
    ) -> list[RetrievedChunk]:
        """
        Text pipeline: Dense search → ticker boost → [Reranker] → top-k.

        Nếu detect ticker, áp dụng soft boost (x1.5) cho chunks cùng ticker.
        Không hard filter — text context từ doc khác vẫn có thể hữu ích
        (ví dụ: so sánh ngành, thị trường chung).
        """
        candidate_k = top_k * 4 if not (self._reranker and self.use_reranker) else top_k * 2

        # Dense search (Gemini Embedding 2 cosine similarity)
        dense_results = self._dense_search(
            query_vec, self.text_col, top_n=candidate_k
        )

        # Ticker boost: ưu tiên chunks cùng mã, nhưng không loại bỏ mã khác
        if tickers:
            dense_results = _ticker_boost(dense_results, tickers, boost=1.5)

        # ── Reranker: chỉ khi KHÔNG detect được ticker ──
        # Khi có ticker → ticker boost đã đủ chính xác
        # Khi không có ticker → dense search dễ nhiễu → reranker giúp lọc
        if self._reranker and self.use_reranker and not tickers:
            candidates_for_rerank = []
            for chunk_id, score in dense_results:
                content = self._get_chunk_content(chunk_id, self.text_col)
                candidates_for_rerank.append((chunk_id, content, score))
            if candidates_for_rerank:
                dense_results = self._reranker.rerank(query, candidates_for_rerank)

        results = []
        for chunk_id, score in dense_results[:top_k]:
            chunk = self._build_text_chunk(chunk_id, score)
            if chunk:
                results.append(chunk)
        return results

    def _image_pipeline(
        self, query: str, query_vec: list[float], top_k: int, tickers: list[str] = None
    ) -> list[RetrievedChunk]:
        """
        Image pipeline: Dense search → ticker hard filter → [Reranker] → top-k.

        Nếu detect ticker → HARD FILTER:
            Phase 1: chỉ search trong ảnh đúng ticker
            Phase 2: nếu thiếu, bổ sung từ search toàn bộ (dedup)

        Biểu đồ GAS không bao giờ relevant khi hỏi về VNM
        → hard filter là hành vi đúng cho image pipeline.
        """
        candidate_k = top_k * 4 if not (self._reranker and self.use_reranker) else top_k * 2

        # ── Dense Search ──
        where_clause = {"ticker": tickers[0]} if (tickers and len(tickers) == 1) else ({"ticker": {"$in": tickers}} if tickers else None)
        dense_results = self._dense_search(
            query_vec, self.image_col, top_n=candidate_k, where=where_clause
        )

        # ── Bổ sung nếu thiếu (khi có tickers) ──
        if tickers and len(dense_results) < top_k:
            seen = {cid for cid, score in dense_results}
            extra_results = self._dense_search(
                query_vec, self.image_col, top_n=candidate_k
            )
            for cid, score in extra_results:
                if cid not in seen:
                    dense_results.append((cid, score))
                    seen.add(cid)
                    if len(dense_results) >= candidate_k: break

        # ── Reranker: chỉ khi KHÔNG detect được ticker ──
        if self._reranker and self.use_reranker and not tickers:
            candidates = []
            for chunk_id, score in dense_results:
                content = self._get_chunk_content(chunk_id, self.image_col)
                candidates.append((chunk_id, content, score))
            if candidates:
                dense_results = self._reranker.rerank(query, candidates)

        results = []
        for chunk_id, score in dense_results[:top_k]:
            chunk = self._build_image_chunk(chunk_id, score)
            if chunk:
                results.append(chunk)
        return results

    # ──────────────────────────────────────────
    # Private: search methods
    # ──────────────────────────────────────────

    def _get_chunk_content(self, chunk_id: str, collection: chromadb.Collection) -> str:
        """Lấy nội dung text/document của 1 chunk từ ChromaDB."""
        try:
            result = collection.get(ids=[chunk_id], include=["documents"])
            if result["documents"]:
                return result["documents"][0] or ""
        except Exception:
            pass
        return ""

    def _dense_search(
        self,
        query_vec: list[float],
        collection: chromadb.Collection,
        top_n: int,
        where: Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """
        Dense search (cosine similarity) trên ChromaDB collection.

        Args:
            where: optional metadata filter, e.g. {"ticker": "VNM"}.
                   Truyền thẳng vào ChromaDB query — chỉ search trong
                   subset documents thoả điều kiện.
        """
        count = collection.count()
        if count == 0:
            return []

        kwargs: dict = {
            "query_embeddings": [query_vec],
            "n_results": min(top_n, count),
            "include": ["distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = collection.query(**kwargs)
        except Exception:
            # Filter trả 0 kết quả hoặc lỗi → trả rỗng
            return []

        ids   = results["ids"][0]
        dists = results["distances"][0]
        # cosine distance → similarity (cao hơn = tốt hơn)
        scored = [(cid, 1 - d) for cid, d in zip(ids, dists)]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    # ──────────────────────────────────────────
    # Private: corpus loading
    # ──────────────────────────────────────────

    def _load_corpus(self):
        """Load metadata từ cả 2 collections để thu thập known tickers."""
        # --- Text corpus metadata ---
        text_data = self.text_col.get(include=["metadatas"])
        text_metas = text_data["metadatas"]
        print(f"  ✓ Text collection: {len(text_metas)} chunks")

        # --- Image corpus metadata ---
        img_data = self.image_col.get(include=["metadatas"])
        img_metas = img_data["metadatas"]
        print(f"  ✓ Image collection: {len(img_metas)} chunks")

        # --- Collect known tickers ---
        for meta in text_metas:
            t = meta.get("ticker", "")
            if t:
                self._known_tickers.add(t)
        for meta in img_metas:
            t = meta.get("ticker", "")
            if t:
                self._known_tickers.add(t)
        print(f"  Known tickers: {sorted(self._known_tickers)}")

    # ──────────────────────────────────────────
    # Private: chunk builders
    # ──────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed query bằng Gemini Embedding 2.
        Cache kết quả để tránh gọi API lại cho query giống hệt.
        Auto-retry khi bị 429 RESOURCE_EXHAUSTED (quota per minute).
        Áp dụng Matryoshka truncation theo embed_dim trong config.
        """
        # ── Cache hit → skip API call ──
        if query in self._query_cache:
            return self._query_cache[query]

        import time
        target_dim = CFG.get("embedding", {}).get("embed_dim", 3072)
        max_retries = 6
        for attempt in range(max_retries):
            try:
                result = self._genai_client.models.embed_content(
                    model=self._embed_model,
                    contents=query,
                )
                vec = list(result.embeddings[0].values)
                # Matryoshka truncation + L2 renormalize
                if target_dim < len(vec):
                    vec = vec[:target_dim]
                    norm = sum(x * x for x in vec) ** 0.5
                    if norm > 0:
                        vec = [x / norm for x in vec]
                self._query_cache[query] = vec  # cache for reuse
                return vec
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 10 * (attempt + 1)   # 10s, 20s, 30s, ...
                    print(f"  ⏳ Quota exceeded, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Embed query failed after {max_retries} retries")

    def _build_text_chunk(self, chunk_id: str, score: float) -> Optional[RetrievedChunk]:
        """Build RetrievedChunk từ text collection."""
        try:
            result = self.text_col.get(ids=[chunk_id], include=["documents", "metadatas"])
        except Exception:
            return None

        if not result["ids"]:
            return None

        doc  = result["documents"][0]
        meta = result["metadatas"][0]

        return RetrievedChunk(
            chunk_id   = chunk_id,
            content    = doc,
            chunk_type = meta.get("chunk_type", "text"),
            score      = score,
            page       = int(meta.get("page", 0)),
            doc        = meta.get("doc", ""),
            img_path   = "",
            has_chart  = False,
            caption    = "",
        )

    def _build_image_chunk(self, chunk_id: str, score: float) -> Optional[RetrievedChunk]:
        """Build RetrievedChunk từ image collection."""
        try:
            result = self.image_col.get(ids=[chunk_id], include=["documents", "metadatas"])
        except Exception:
            return None

        if not result["ids"]:
            return None

        doc  = result["documents"][0]   # rich caption
        meta = result["metadatas"][0]

        return RetrievedChunk(
            chunk_id   = chunk_id,
            content    = doc,           # caption text
            chunk_type = "image",
            score      = score,
            page       = int(meta.get("page", 0)),
            doc        = meta.get("doc", ""),
            img_path   = meta.get("img_path", ""),
            has_chart  = True,
            caption    = meta.get("caption", doc),
        )


# ──────────────────────────────────────────────
# Ticker utilities
# ──────────────────────────────────────────────

def _ticker_of(chunk_id: str) -> str:
    """Extract ticker từ chunk_id (phần đầu trước _)."""
    return chunk_id.split("_")[0] if "_" in chunk_id else ""


def _ticker_boost(
    merged: list[tuple[str, float]],
    tickers: list[str],
    boost: float = 1.5,
) -> list[tuple[str, float]]:
    """
    Boost score cho chunks thuộc ticker được detect.

    Không loại bỏ chunks khác ticker — chỉ đẩy chunks đúng ticker lên trên.
    Dùng cho text pipeline (context từ doc khác vẫn có thể hữu ích).
    """
    boosted = []
    for chunk_id, score in merged:
        if _ticker_of(chunk_id) in tickers:
            boosted.append((chunk_id, score * boost))
        else:
            boosted.append((chunk_id, score))
    return sorted(boosted, key=lambda x: x[1], reverse=True)
