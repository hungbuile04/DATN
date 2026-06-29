# Multimodal RAG — Phân tích báo cáo tài chính (ĐATN)

Hệ thống RAG đa phương thức (text + table + chart/image) cho báo cáo tài chính doanh nghiệp Việt Nam. Đồ án tốt nghiệp.

## Kiến trúc

- **Extraction**: Docling + PyMuPDF → chunks (text/table/image)
- **Embedding**: Gemini Embedding 2 (3072-dim), 2 collection ChromaDB:
  - `rag_text` — text + table
  - `rag_image` — blend `α × caption + (1-α) × image`, mặc định α=0.8
- **Retrieval**: Dual retriever (dense + ticker-aware boost, không BM25)
- **Reasoning — 3 Agents / 2 Stage**:
  - Stage 1 (song song qua `ThreadPoolExecutor`):
    - `TextAgent` — phân tích text/table, self-correction, entity resolution
    - `ImageAgent` — phân tích chart bằng Vision LLM (OCR, trend)
  - Stage 2: `SumAgent` (Editor) — gộp draft, giải mâu thuẫn, giữ nguyên fact
  - Tổng cộng **3 LLM calls/câu hỏi**, giảm ~40% latency
- **LLM**: Gemini 2.5 Flash qua OpenRouter
- **UI**: CLI (`ask.py`) + Web Streamlit (`app.py`)

## Môi trường

- **Python venv**: `ragenv/` ở project root
  - Activate: `source ragenv/bin/activate`
  - **Luôn** dùng venv này khi chạy lệnh `python` — không dùng Python hệ thống
- **API keys** (file `.env` ở project root):
  - `GOOGLE_API_KEY` — Gemini Embedding 2 + Gemini Vision (caption ảnh)
  - `OPENROUTER_API_KEY` — Gemini 2.5 Flash (agents)
  - **KHÔNG** đọc nội dung, sửa, in ra, hay commit file `.env`. Nếu cần biết key có hợp lệ chưa thì hỏi user, đừng tự `cat .env`.

## Cấu trúc thư mục

```
source_code_2/
├── app.py                  # Streamlit UI
├── ask.py                  # CLI hỏi đáp
├── config/
│   ├── config.yaml         # Cấu hình (paths, top_k, batch_size, α…)
│   └── settings.py         # Loader → đổi paths sang absolute, expose CFG
├── src/
│   ├── extraction/         # extractor.py, figure_extractor.py, run_extraction.py
│   ├── embedding/          # embedder.py, run_embedding.py, fix_captions.py, reembed_images.py
│   ├── retrieval/          # retriever.py (DualRetriever), query_router.py, run_retrieval.py
│   ├── agents/             # agents.py (Text/Image/SumAgent), orchestrator.py (MultiAgentOrchestrator)
│   └── eval/               # runner.py, judge.py, ablation_agents.py, eval_retrieval.py, questions*.json
├── pdfs/
│   ├── raw/                # Input PDF
│   └── processed/          # chunks/, images/, metadata.json, vector_db/ (git-ignored)
├── results/                # Output evaluation (git-ignored)
├── requirements.txt
└── .env                    # API keys (git-ignored, KHÔNG đụng)
```

## Lệnh thường dùng

| Mục đích | Lệnh |
|---|---|
| Extract PDFs trong `pdfs/raw/` | `python -m src.extraction.run_extraction` |
| Build embeddings vào ChromaDB | `python -m src.embedding.run_embedding` |
| Hỏi đáp CLI (interactive) | `python ask.py` |
| Hỏi đáp CLI (one-shot) | `python ask.py --q "Biên lợi nhuận gộp VNM thay đổi thế nào?"` |
| Web UI | `streamlit run app.py` |
| Eval 4-mode + judge | `python -m src.eval.runner --judge --resume` |
| Eval chỉ multi-agent | `python -m src.eval.runner --skip_baselines --judge --judge_target multi_agent --resume` |
| Retrieval metrics (P@k, R@k, MRR) | `python -m src.eval.eval_retrieval` |
| Ablation agents | `python -m src.eval.ablation_agents --questions src/eval/question_test.json --judge` |

**Lưu ý**:
- Trước eval >50 câu, **luôn** dùng cờ `--resume` để khôi phục nếu bị ngắt.
- Lệnh `run_extraction` và `run_embedding` tốn API quota + ghi đè dữ liệu — phải xác nhận với user trước khi chạy.

## Quy ước

- **PDF naming**: `TICKER_YYYY_MM_DD_SOURCE.pdf` — ví dụ `VNM_2025_03_10_SSIResearch.pdf`
- **Chunk types**:
  - `*_text.md` — đoạn văn bản
  - `*_tableN.md` — bảng markdown
  - `*_chartN.png` — ảnh chart (có caption tự sinh)
- **Agent output schema**: JSON với `analysis`, `confidence` (0.0–1.0), `key_facts`, `self_correction`, `entity_resolved`
- **Config pattern**: Mọi đường dẫn trong `config.yaml` được `config/settings.py` convert sang absolute để tránh lỗi `cwd` khi import từ vị trí khác — không dùng relative path trực tiếp trong code.

## Khi chỉnh sửa code

- **Logic agent / pipeline**: bắt đầu từ [src/agents/orchestrator.py](src/agents/orchestrator.py) (`MultiAgentOrchestrator`, `ThreadPoolExecutor` cho Stage 1) → [src/agents/agents.py](src/agents/agents.py).
- **Retrieval**: [src/retrieval/retriever.py](src/retrieval/retriever.py) (`DualRetriever`). Đổi α của image blend → cập nhật `config.yaml`, KHÔNG hardcode.
- **Embedding**: rebuild ChromaDB sau khi đổi embedder/blend bằng `run_embedding`.
- **Dataset eval** (`src/eval/questions*.json`): không sửa trừ khi user yêu cầu rõ — số câu hỏi và mã ground-truth được dùng để so sánh giữa các lần chạy.
- **Khi thay đổi prompt agent**: chạy `ablation_agents` trên `question_test.json` trước để sanity check, đừng chạy full eval ngay.

## Tech stack

| Component | Tech |
|---|---|
| PDF extraction | Docling, PyMuPDF |
| Embedding | Gemini Embedding 2 (3072-dim) |
| Vector DB | ChromaDB 0.5.3 (persistent) |
| LLM (text + vision) | Gemini 2.5 Flash via OpenRouter |
| Retrieval | Dense + ticker-aware scoring |
| Image embedding | α-blend caption + image (α=0.8) |
| UI | Streamlit |

## Phong cách phản hồi

- User là sinh viên đang làm đồ án tốt nghiệp — giải thích **rõ ràng, có ví dụ**, nhưng không dài dòng.
- Tiếng Việt là OK, thuật ngữ kỹ thuật giữ tiếng Anh (embedding, retrieval, agent, chunk…).
- Khi trả về kết quả eval, luôn nói rõ scale (0–20 cho judge, 0–1 cho retrieval metrics).
