# src/retrieval/lora_reranker.py

"""
LoRA Cross-Encoder Reranker — Local reranker thay thế Gemini Flash API.

Fine-tuned cross-encoder (BAAI/bge-reranker-v2-m3 + LoRA adapter) để
scoring relevance (query, passage) → sắp xếp lại candidates.

Drop-in replacement cho GeminiReranker — cùng interface rerank():
    rerank(query, candidates) → list[(chunk_id, final_score)]

Ưu điểm so với GeminiReranker:
    - Không tốn API call (chạy local)
    - Latency ~50-100ms vs ~1-2s (API)
    - Fine-tuned trên domain-specific Vietnamese financial QA data
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class LoRAReranker:
    """
    Cross-encoder reranker với LoRA fine-tuned weights.

    Flow:
        1. Nhận list (chunk_id, content, dense_score)
        2. Tokenize (query, passage) pairs
        3. Forward pass → logit → sigmoid → relevance score
        4. Kết hợp: final = dense_weight × dense_score + lora_weight × ce_score
        5. Sort giảm dần

    Args:
        adapter_path:   đường dẫn đến LoRA adapter weights
        base_model:     tên base model trên HuggingFace
        device:         "cuda", "mps", hoặc "cpu"
        max_length:     max token length cho tokenizer
        dense_weight:   trọng số cho dense score (mặc định 0.4)
        lora_weight:    trọng số cho cross-encoder score (mặc định 0.6)
    """

    def __init__(
        self,
        adapter_path: str,
        base_model: str = "BAAI/bge-reranker-v2-m3",
        device: str = None,
        max_length: int = 512,
        dense_weight: float = 0.4,
        lora_weight: float = 0.6,
    ):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.max_length = max_length
        self.dense_weight = dense_weight
        self.lora_weight = lora_weight

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load model + merge LoRA adapter
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            # LoRA adapter found → load base + merge
            from peft import PeftModel
            base = AutoModelForSequenceClassification.from_pretrained(
                base_model, num_labels=1, torch_dtype=torch.float32,
            )
            self.model = PeftModel.from_pretrained(base, str(adapter_dir))
            self.model = self.model.merge_and_unload()  # Merge for faster inference
            print(f"  ✓ LoRA Reranker loaded: {base_model} + {adapter_path}")
        else:
            # No adapter → use base model only (zero-shot)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model, num_labels=1, torch_dtype=torch.float32,
            )
            print(f"  ⚠ No LoRA adapter found at {adapter_path}, using base model")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
    ) -> list[tuple[str, float]]:
        """
        Rerank candidates bằng cross-encoder fine-tuned.

        Args:
            query:      câu hỏi người dùng
            candidates: list of (chunk_id, content, dense_score)

        Returns:
            list of (chunk_id, final_score) — đã sắp xếp giảm dần
        """
        if not candidates:
            return []

        # Nếu ≤ 2 candidates, không cần rerank
        if len(candidates) <= 2:
            return [(cid, score) for cid, _, score in candidates]

        try:
            ce_scores = self._score_pairs(query, candidates)
            return self._merge_scores(candidates, ce_scores)
        except Exception as e:
            print(f"  ⚠ LoRA Reranker fallback: {e}")
            return [(cid, score) for cid, _, score in candidates]

    def _score_pairs(
        self, query: str, candidates: list[tuple[str, str, float]]
    ) -> list[float]:
        """Score (query, passage) pairs qua cross-encoder."""
        # Prepare inputs
        pairs = []
        for _, content, _ in candidates:
            # Truncate content to avoid exceeding max_length
            truncated = content[:2000] if len(content) > 2000 else content
            pairs.append((query, truncated))

        # Batch tokenize
        queries = [p[0] for p in pairs]
        passages = [p[1] for p in pairs]

        inputs = self.tokenizer(
            queries, passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(-1)  # (batch_size,)

        # Sigmoid → relevance scores [0, 1]
        scores = torch.sigmoid(logits).cpu().tolist()

        # Handle single element case
        if isinstance(scores, float):
            scores = [scores]

        return scores

    def _merge_scores(
        self,
        candidates: list[tuple[str, str, float]],
        ce_scores: list[float],
    ) -> list[tuple[str, float]]:
        """Kết hợp dense_score + cross-encoder score → final_score, sort giảm dần."""
        merged = []
        for i, (chunk_id, _, dense_score) in enumerate(candidates):
            ce_score = ce_scores[i] if i < len(ce_scores) else 0.5
            final = self.dense_weight * dense_score + self.lora_weight * ce_score
            merged.append((chunk_id, final))

        return sorted(merged, key=lambda x: x[1], reverse=True)
