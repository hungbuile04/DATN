# Tài liệu chi tiết — Multimodal RAG cho Báo cáo Tài chính

> Hệ thống RAG đa phương thức (text + table + image) phân tích báo cáo tài chính tiếng Việt, dùng Gemini Embedding 2 + ChromaDB + 3-Agent Reasoning (Text + Image → Sum) qua OpenRouter.

---

## Mục lục

1. [Cấu trúc dự án](#1-cấu-trúc-dự-án)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)

> **Thay đổi chính so với phiên bản cũ:**
> - Pipeline 4 Agents → **3 Agents** (CriticalAgent đã loại bỏ)
> - Category câu hỏi chuẩn hóa: `text_lookup`, `multi_hop_reasoning`, `visual_analysis`, `numerical_reasoning`
> - Judge hỗ trợ **context-aware Faithfulness** (truyền retrieved chunks)
> - Runner lưu retrieved chunks vào output JSON
3. [Cấu hình hệ thống](#3-cấu-hình-hệ-thống)
4. [Pipeline 1 — Extraction (PDF → metadata.json)](#4-pipeline-1--extraction-pdf--metadatajson)
5. [Pipeline 2 — Embedding (metadata → ChromaDB)](#5-pipeline-2--embedding-metadata--chromadb)
6. [Pipeline 3 — Retrieval (Dual-Pipeline)](#6-pipeline-3--retrieval-dual-pipeline)
7. [Pipeline 4 — Multi-Agent Reasoning](#7-pipeline-4--multi-agent-reasoning)
8. [Entrypoints — Hỏi đáp người dùng](#8-entrypoints--hỏi-đáp-người-dùng)
9. [Evaluation — 4 modes & Ablation studies](#9-evaluation--4-modes--ablation-studies)
10. [Bảng tham chiếu nhanh các lệnh Python](#10-bảng-tham-chiếu-nhanh-các-lệnh-python)

---

## 1. Cấu trúc dự án

```
source_code_2/
├── app.py                       # Streamlit UI
├── ask.py                       # CLI hỏi đáp
├── requirements.txt
├── .env                         # GOOGLE_API_KEY, OPENROUTER_API_KEY
├── config/
│   ├── config.yaml              # Cấu hình paths, embedding, retrieval, agents
│   └── settings.py              # Loader CFG = load_config()
├── src/
│   ├── extraction/              # Pipeline 1: PDF → chunks
│   │   ├── extractor.py
│   │   ├── figure_extractor.py
│   │   └── run_extraction.py
│   ├── embedding/               # Pipeline 2: chunks → vectors
│   │   ├── embedder.py
│   │   ├── run_embedding.py
│   │   ├── fix_captions.py
│   │   └── reembed_images.py
│   ├── retrieval/               # Pipeline 3: query → chunks
│   │   ├── retriever.py
│   │   ├── query_router.py
│   │   └── run_retrieval.py
│   ├── agents/                  # Pipeline 4: chunks → answer
│   │   ├── agents.py
│   │   └── orchestrator.py
│   └── eval/                    # Đánh giá & ablation
│       ├── runner.py            # 4-mode comparison
│       ├── runner_dual_encode.py
│       ├── modes.py             # TextOnlyRetriever / TextTableRetriever / FullMultimodalRetriever
│       ├── generator.py         # AnswerGenerator (single LLM)
│       ├── judge.py             # LLMJudge (4 tiêu chí, 0-20, context-aware)
│       ├── ablation_agents.py
│       ├── ablation_embed_image.py
│       ├── ablation_embed_weight.py
│       ├── eval_retrieval.py    # Precision/Recall/MRR/HitRate
│       ├── eval_image_pipeline.py
│       ├── questions.json               # Main eval (265 câu)
│       ├── questions_hard.json          # Hard reasoning (100 câu)
│       ├── questions_151_250.json       # Extended eval
│       ├── questions_sab_ree_pvt_pvs.json  # SAB, REE, PVT, PVS
│       ├── questions_vcb_tpb_tng_stb.json  # VCB, TPB, TNG, STB
│       ├── questions_vpb_vnm_vhm_vea.json  # VPB, VNM, VHM, VEA
│       └── question_test.json           # Small test set (quick runs)
├── pdfs/
│   ├── raw/                     # Input PDF
│   └── processed/
│       ├── chunks/              # *_text.md, *_table*.md
│       ├── images/              # *_chart*.png
│       ├── metadata.json
│       └── vector_db/           # ChromaDB persistent
└── results/                     # Output evaluation JSON
```

---

## 2. Cài đặt môi trường

### 2.1. Tạo môi trường ảo & cài dependencies

```bash
cd source_code_2

python -m venv ragenv
source ragenv/bin/activate          # macOS / Linux

pip install -r requirements.txt
```

**Dependencies chính** (xem [requirements.txt](requirements.txt)):

| Package | Vai trò |
|---|---|
| `docling>=2.0.0` | Trích xuất PDF (table structure, picture extraction) |
| `PyMuPDF>=1.24.10` (fitz) | Crop ảnh, đọc text blocks, screenshot |
| `chromadb==0.5.3` | Vector database (HNSW cosine) |
| `google-genai>=1.0.0` | Gemini Embedding 2 + Vision |
| `openai>=1.30.0` | Client cho OpenRouter (LLM agents) |
| `rank-bm25==0.2.2` | (legacy — đã loại khỏi pipeline chính) |
| `pyyaml`, `python-dotenv` | Config & env |
| `tqdm`, `Pillow`, `tabulate` | Tiện ích |
| `streamlit>=1.30.0` | Web UI |

### 2.2. Tạo file `.env`

```env
GOOGLE_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-v1-...
```

- **GOOGLE_API_KEY** → cho Gemini Embedding 2 + Gemini Vision (caption ảnh)
- **OPENROUTER_API_KEY** → cho LLM agents (`google/gemini-2.5-flash` qua OpenRouter)

---

## 3. Cấu hình hệ thống

### 3.1. [config/config.yaml](config/config.yaml)

```yaml
paths:
  pdf_raw:       "pdfs/raw"
  pdf_processed: "pdfs/processed"
  vector_db:     "pdfs/processed/vector_db"
  metadata:      "pdfs/processed/metadata.json"
  bm25_cache:    "pdfs/processed/vector_db"

extraction:
  dpi: 216                    # images_scale = 216/72 = 3.0
  min_text_length: 80
  min_table_pipes: 4

embedding:
  model: "gemini-embedding-2-preview"
  embed_dim: 3072
  batch_size: 10
  sleep_between_batches: 1.5
  min_token_count: 50
  text_collection:  "rag_text"   # text + table
  image_collection: "rag_image"  # caption + image blend

retrieval:
  text_top_k:  5
  image_top_k: 3
  rrf_k: 60                       # legacy

agents:
  llm_model:    "google/gemini-2.5-flash"
  vision_model: "google/gemini-2.5-flash"
  max_context_chunks: 3
  temperature: 0.1
```

### 3.2. [config/settings.py](config/settings.py)

```python
from config.settings import CFG     # CFG là dict đã load 1 lần

CFG["paths"]["vector_db"]            # Path tuyệt đối
CFG["embedding"]["model"]
CFG["agents"]["llm_model"]
```

`load_config()` tự convert tất cả `paths.*` thành `Path` tuyệt đối từ ROOT, chạy được từ bất kỳ cwd nào.

---

## 4. Pipeline 1 — Extraction (PDF → metadata.json)

### 4.1. Lệnh chạy

```bash
# Đặt PDF vào pdfs/raw/ trước
python -m src.extraction.run_extraction
```

> File phải đặt tên theo format `TICKER_YYYY_MM_DD_SOURCE.pdf`
> Ví dụ: `VNM_2026_02_24_SSIResearch.pdf`, `HDB_2025_11_21_SSIResearch.pdf`
> Nếu không đúng format, ticker/report_date/source sẽ rỗng (xem [`_parse_doc_name`](src/extraction/extractor.py#L652)).

### 4.2. Đầu vào / Đầu ra

| | Đường dẫn | Mô tả |
|---|---|---|
| Input | `pdfs/raw/*.pdf` | PDF báo cáo tài chính |
| Output | `pdfs/processed/chunks/*.md` | Text + table chunks (markdown) |
| Output | `pdfs/processed/images/*.png` | Chart/figure images |
| Output | `pdfs/processed/metadata.json` | Danh sách chunk + metadata |

### 4.3. Cấu trúc 1 chunk metadata

```json
{
  "chunk_id":   "VNM_2026_02_24_SSIResearch_p005_c00_text",
  "chunk_type": "text",                  // "text" | "table" | "image"
  "doc":        "VNM_2026_02_24_SSIResearch",
  "page":       5,
  "ticker":     "VNM",
  "report_date":"2026-02-24",
  "source":     "SSIResearch",
  "page_type":  "text",                  // "text" | "table" | "chart" | "mixed" | "header_only"
  "text_path":  ".../chunks/...md",
  "img_path":   null,
  "content_for_embedding": ".../chunks/...md",
  "has_text": true, "has_table": false, "has_chart": false,
  "token_count": 380
}
```

### 4.4. Bên trong [`extract_pdf()`](src/extraction/extractor.py#L50)

1. **Docling conversion** — `DocumentConverter` với `PdfPipelineOptions(do_table_structure=True, mode=ACCURATE, generate_picture_images=True, images_scale=dpi/72)`
2. **Group items by page**: `doc.iterate_items()` → text/captions; `doc.tables` → tables; `doc.pictures` → charts (link với caption FIFO)
3. **Caption fallback** (khi caption chỉ là "Nguồn: SSI Research"):
   - Fallback A: text block ngay phía trên ảnh ([`_find_title_above_pic`](src/extraction/extractor.py#L797))
   - Fallback B: regex tìm "biểu đồ", "đồ thị", "chart", "figure" trong text page
4. **Phân loại trang**: `text` / `table` / `chart` / `mixed` / `header_only` (skip header_only)
5. **Text chunks**: sliding window 400 tokens, overlap 80 ([`_sliding_window_chunk`](src/extraction/extractor.py#L633))
6. **Table chunks**: `table.export_to_dataframe(doc).to_markdown()`
7. **Image chunks**:
   - SSI cover page: chỉ giữ ảnh thứ 3 (chart giá CP), bỏ logo + header
   - Detect 2 chart trong 1 picture → tách bằng [`_split_chart_regions`](src/extraction/extractor.py#L684)
   - Crop mở rộng 40pt phía trên (caption), 20pt dưới (nguồn) — [`_crop_chart_expanded`](src/extraction/extractor.py#L527)
   - Min chart size = 80pt (loại logo/icon)
8. **Làm sạch text tiếng Việt** (Docling hay tách dấu): [`_clean_ocr_text`](src/extraction/extractor.py#L460) — 3-pass regex sửa "v ề"→"về", "Lợ i"→"Lợi", "Bi ểu"→"Biểu"
9. **Loại nhãn trục** (axis ticks rò vào text): [`_is_axis_noise`](src/extraction/extractor.py#L578), [`_strip_axis_prefix`](src/extraction/extractor.py#L493)

---

## 5. Pipeline 2 — Embedding (metadata → ChromaDB)

### 5.1. Lệnh chạy chính

```bash
python -m src.embedding.run_embedding
```

Yêu cầu: `metadata.json` đã tạo, `GOOGLE_API_KEY` trong `.env`.

### 5.2. Hai pipeline song song

```
metadata.json
   ├── text  + table chunks  →  Pipeline 1  →  rag_text  collection
   └── image chunks          →  Pipeline 2  →  rag_image collection
```

#### Pipeline 1 (Text + Table)

```python
# embedder.py
contents = [Path(e["text_path"]).read_text() for e in entries]
for batch in chunks(contents, batch_size=10):
    vectors = embedder.embed_texts_batch(batch)
    time.sleep(1.5)
text_collection.upsert(ids, vectors, documents, metadatas)
```

Lọc: bỏ text chunks có `token_count < 50` (giảm nhiễu).

#### Pipeline 2 (Image)

```python
# Resume: skip ảnh đã có trong ChromaDB
existing_ids = set(image_collection.get(include=[])["ids"])
todo = [e for e in image_entries if e["chunk_id"] not in existing_ids]

for entry in todo:
    rich_caption = embedder.caption_image(img_path)         # Gemini Vision
    vector = embedder.embed_image(img_path, rich_caption)   # weighted blend α=0.8
    image_collection.upsert(...)
```

### 5.3. Aggregated Embedding (weighted blend)

[`embed_image()`](src/embedding/embedder.py#L170) — **2 API calls/ảnh**, blend thủ công:

```python
caption_vec = embed_text(caption)         # 1 API call
image_vec   = embed_content([image_bytes]) # 1 API call
blended = α × caption_vec + (1-α) × image_vec
return L2_normalize(blended)
```

- **α = 0.8** mặc định (caption nặng hơn image)
- Tốt hơn 22.3% so với Gemini aggregated tự blend (≈α=0.2 không kiểm soát được)
- Confirm bằng [`ablation_embed_weight.py`](src/eval/ablation_embed_weight.py)

### 5.4. Caption Prompt

Xem [`CAPTION_PROMPT`](src/embedding/embedder.py#L28) — yêu cầu Gemini Vision sinh 3-5 câu mô tả tiếng Việt:
1. Tiêu đề + loại biểu đồ (đường/cột/tròn)
2. Trục (tên, đơn vị, khoảng giá trị)
3. Đường/cột chính + legend
4. Xu hướng + cực đại/cực tiểu + giá trị cụ thể
5. Mã cổ phiếu + thuật ngữ song ngữ (GPM/biên lợi nhuận gộp...)

### 5.5. Helper scripts

#### Re-caption những ảnh fallback

Khi Gemini bị 429/503 lúc embed lần đầu → caption rơi vào fallback `"Biểu đồ tài chính (chunk_id)"` → image retrieval thất bại.

```bash
# Xem trước (không sửa)
python -m src.embedding.fix_captions --dry-run

# Sửa thật: re-caption + re-embed + update ChromaDB + metadata.json
python -m src.embedding.fix_captions
```

Script [`fix_captions.py`](src/embedding/fix_captions.py): re-caption từng ảnh → `embed_image()` (aggregated) → `col.update(ids, embeddings, documents, metadatas)`.

#### Re-embed với α khác (caption đã có)

```bash
# Mặc định α=0.8
python -m src.embedding.reembed_images

# Đổi α
python -m src.embedding.reembed_images --alpha 0.6

# Dry run
python -m src.embedding.reembed_images --dry-run

# Chỉ re-embed 1 ticker
python -m src.embedding.reembed_images --ticker VNM
```

Script [`reembed_images.py`](src/embedding/reembed_images.py): không gọi caption_image (đã có), chỉ gọi `embed_text(caption) + embed_content(image)` rồi blend với α mới → `col.update(ids, embeddings)`.

---

## 6. Pipeline 3 — Retrieval (Dual-Pipeline)

### 6.1. Lệnh test CLI

```bash
python src/retrieval/run_retrieval.py --query "biến động giá cổ phiếu VNM"

python src/retrieval/run_retrieval.py \
    --query "NIM HDB quý 3" \
    --text_top_k 3 --image_top_k 2
```

### 6.2. Class [`DualRetriever`](src/retrieval/retriever.py#L67)

```python
from src.retrieval.retriever import DualRetriever
from config.settings import CFG

retriever = DualRetriever(
    chroma_path = str(CFG["paths"]["vector_db"]),
    api_key     = google_api_key,
    cfg         = CFG,
)

result = retriever.retrieve(
    query        = "Biên lợi nhuận VNM thay đổi thế nào?",
    text_top_k   = 5,
    image_top_k  = 3,
)

# result.text_chunks  : list[RetrievedChunk]
# result.image_chunks : list[RetrievedChunk]
# result.all_chunks   : property gộp text + image
```

### 6.3. Luồng `retrieve()`

```
query
  ├── _embed_query(query)            # Gemini, có cache
  ├── _detect_ticker(query)          # exact ticker / _COMPANY_ALIASES
  │
  ├── _text_pipeline(query, vec, k, ticker)
  │     ├── _dense_search(vec, text_col, top_n=k*4)
  │     ├── if ticker: _ticker_boost(boost=1.5)
  │     └── top-k
  │
  ├── _infer_ticker_from_chunks(text_chunks)   # majority vote nếu chưa có ticker
  │
  └── _image_pipeline(query, vec, k, ticker)
        if ticker:
          Phase 1: _dense_search(vec, image_col, where={"ticker": X})
          Phase 2: _dense_search(vec, image_col)  # bổ sung không filter, dedup
        else:
          _dense_search(vec, image_col)
        Lọc: score >= IMAGE_MIN_SCORE (0.015)
```

### 6.4. Ticker-aware

| Pipeline | Cách áp dụng ticker |
|---|---|
| **Text** | Soft boost `score × 1.5` cho chunks cùng ticker (không loại doc khác — context so sánh ngành vẫn dùng được) |
| **Image** | Hard filter `where={"ticker": X}` — biểu đồ GAS không bao giờ relevant cho câu hỏi VNM |

[`_COMPANY_ALIASES`](src/retrieval/retriever.py#L178) map tên thông dụng → ticker:
- `vinamilk` → `VNM`
- `hòa phát`, `hoa phat` → `HPG`
- `vietcombank` → `VCB`
- `vietinbank` → `CTG`
- `nhà khang điền`, `khang dien` → `KDH`
- ... (~30 alias)

### 6.5. Query Router (tùy chọn)

```python
from src.retrieval.query_router import classify_query

classify_query("biểu đồ giá cổ phiếu")     # → "visual"
classify_query("so sánh NIM Q3 và Q2")     # → "comparative"
classify_query("EPS của VNM là bao nhiêu") # → "factual"
```

Trong `run_retrieval.py`: nếu `query_type == "visual"` thì tăng `image_top_k` lên tối thiểu 3, nhưng **không cắt về 0** cho factual (câu factual vẫn có thể có biểu đồ liên quan).

### 6.6. Lưu ý quan trọng

- **BM25+RRF đã bị loại bỏ** sau eval cho thấy Dense-only (Gemini Embedding 2, 3072-dim) tốt hơn:
  - Precision@k +10%
  - Hit Rate@k +20%
  - MRR +22%
  - BM25 tokenizer (`str.split`) yếu cho tiếng Việt → thêm nhiễu
- Cosine distance từ ChromaDB → similarity = `1 - distance`
- Query embedding cache: cùng 1 query không gọi API lại

---

## 7. Pipeline 4 — Multi-Agent Reasoning

### 7.1. Kiến trúc 3 agents (2 stages)

> **Lưu ý:** CriticalAgent (4-agent) đã bị loại bỏ khỏi pipeline chính. Chỉ còn stub backward compatibility trong `agents.py`.

```
                 RetrievalResult
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   Text Agent                 Image Agent       ← Stage 1 (PARALLEL)
   (text + table)             (charts)
   gemini-2.5-flash           gemini-2.5-flash
        │                           │
        └─────────────┬─────────────┘
                      ▼
                  Sum Agent                     ← Stage 2 (Editor)
                  gemini-2.5-flash
                      │
                      ▼
              {answer, confidence,
               reasoning, sources, ...}
```

3 LLM calls/câu hỏi. Stage 1 chạy parallel qua `ThreadPoolExecutor(max_workers=2)` → giảm ~40% latency.

> CriticalAgent, TableAgent (gộp vào TextAgent), GeneralAgent — đều có stub class cho backward compat nhưng **không được dùng** trong pipeline hiện tại.

### 7.2. Class [`MultiAgentOrchestrator`](src/agents/orchestrator.py#L24)

```python
from src.agents.orchestrator import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator(
    api_key      = openrouter_key,
    text_model   = "google/gemini-2.5-flash",
    vision_model = "google/gemini-2.5-flash",
)

final = orchestrator.run(question, retrieval_result)
# {
#     "answer":        str,
#     "confidence":    float,         # 0.0 - 1.0
#     "reasoning":     str,
#     "sources":       list[dict],
#     "agent_outputs": {"text": {...}, "image": {...}},
#     "model":         "multi-agent-3",
#     "has_image":     bool,
#     "num_images":    int,
# }
```

### 7.3. 3 agents

| Agent | File | Input | Đặc điểm |
|---|---|---|---|
| `TextAgent` | [agents.py:89](src/agents/agents.py#L89) | text + table chunks | `[SELF-CORRECTION]` (sửa số sai trong câu hỏi — chỉ khi chắc 100%, thỏa 3 điều kiện), `[FUZZ ENTITY]` (suy luận công ty khi mơ hồ). Output JSON: `analysis`, `confidence`, `key_facts`, `self_correction`, `entity_resolved` |
| `ImageAgent` | [agents.py:249](src/agents/agents.py#L249) | image chunks (resize 1280px, base64, tối đa 3 ảnh) | OCR text + xu hướng + so sánh tương đối. Cho phép giá trị xấp xỉ ("~4.8%", "khoảng 5%"). Ưu tiên tín hiệu hữu ích, không kết luận "không có thông tin" khi có thể suy ra xu hướng. Output JSON: `analysis`, `confidence`, `chart_type`, `trend` |
| `SumAgent` | [agents.py:363](src/agents/agents.py#L363) | draft từ text+image | **Editor, không writer**: giữ NGUYÊN số liệu, gộp + loại trùng + xử lý mâu thuẫn (ghi chú nguồn, VD: "Theo text: 16,6%, theo biểu đồ: ~17%"). Output JSON: `answer`, `confidence`, `reasoning` |

### 7.4. Backward compatibility stubs

Các class cũ vẫn import được nhưng không hoạt động:

```python
# agents.py — stubs
class CriticalAgent:    # Deprecated — trả dict rỗng
TableAgent = TextAgent  # Gộp vào TextAgent
class GeneralAgent:     # Deprecated — trả AgentOutput rỗng
```

### 7.5. Helper trong [agents.py](src/agents/agents.py)

```python
_format_chunks(chunks)    # → [1] TYPE | Trang N | doc\n content\n--- ...
_encode_image(path, max_long_edge=1280)  # PIL resize + base64 PNG
_parse_json(text)         # robust parse JSON từ ```json``` block
```

### 7.6. Cách dùng SumAgent độc lập

```python
from src.agents.agents import TextAgent, ImageAgent, SumAgent, AgentOutput

text_out  = text_agent.analyze(question, text_chunks)
image_out = image_agent.analyze(question, image_chunks)
final     = sum_agent.synthesize(question, text_out, image_out)
```

---

## 8. Entrypoints — Hỏi đáp người dùng

### 8.1. CLI [ask.py](ask.py)

```bash
# Chế độ tương tác
python ask.py

# 1 câu hỏi cụ thể
python ask.py --q "Biên lợi nhuận gộp VNM thay đổi thế nào?"

# Tùy chỉnh top-k
python ask.py --q "NIM HDB quý 3" --text_top_k 3 --image_top_k 2
```

**Tham số:**

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--q` | (không có → loop) | Câu hỏi (đặt trong dấu ngoặc kép) |
| `--text_top_k` | 5 | Số text chunks |
| `--image_top_k` | 3 | Số image chunks |

**Output mẫu:**

```
🔍 Đang tìm kiếm tài liệu... ✅ (Tìm thấy 5 đoạn text, 2 ảnh/biểu đồ)
🧠 Các đặc vụ đang phân tích (Reasoning)...
    Text chunks  : 5
    Image chunks : 2
    → Text + Image Agents (parallel)...
      text=0.85, image=0.70
    → Sum Agent... confidence=0.82

💡 CÂU TRẢ LỜI:
Biên lợi nhuận gộp (GPM) của VNM trong Q4/2025 đạt 40,4%, ...

🤖 Lực lượng tham gia: multi-agent-3 | Tự tin: 0.82
📝 Nguồn: 7 chunks
```

### 8.2. Web UI [app.py](app.py)

```bash
streamlit run app.py
```

**Tính năng:**

- **2 chế độ trả lời** (chọn ở sidebar):
  - 🤖 **Multi-Agent (3 Agents)** → Text + Image (song song) → Sum Agent (editor)
  - 💬 **Single LLM** → 1 lần gọi `AnswerGenerator.generate(mode=full_multimodal)` với toàn bộ text + image context
- **Slider top-k**: text 1-15, image 0-10
- **Thống kê hệ thống**: số PDF, ticker, chunks
- **Tab "Hỏi đáp"** (chat) + **Tab "Kiến trúc hệ thống"** (sơ đồ)
- **Hiển thị**:
  - Confidence bar (xanh/vàng/đỏ)
  - Retrieved chunks (text + image preview)
  - Agent pipeline reasoning (Stage 1a/1b/2)
  - Source list
  - Timing breakdown

`init_system()` được cache với `@st.cache_resource` → khởi tạo retriever/orchestrator/generator 1 lần duy nhất.

---

## 9. Evaluation — 4 modes & Ablation studies

### 9.1. So sánh 4 modes — [src/eval/runner.py](src/eval/runner.py)

#### Định nghĩa 4 modes

| Mode | Retriever | Generator | Mô tả |
|---|---|---|---|
| `text_only` | `TextOnlyRetriever` (text chunks only, no table) | text LLM | Baseline thuần text |
| `text_table` | `TextTableRetriever` (text + table) | text LLM | Thêm bảng số liệu |
| `full_multimodal` | `FullMultimodalRetriever` (text + table + image) | vision LLM (1 call) | Đầy đủ multimodal, 1 LLM call |
| `multi_agent` | `DualRetriever` đầy đủ | `MultiAgentOrchestrator` (3 agents) | Pipeline đầy đủ 3 agents |

3 retriever wrapper trong [modes.py](src/eval/modes.py) đều dùng chung 1 `DualRetriever` instance, chỉ khác filter chunk_type.

> **Mới:** `runner.py` lưu cả `text_chunks` và `image_chunks` (chunk_id, score, content preview) vào output JSON để dùng cho context-aware judging.

#### Lệnh chạy cơ bản

```bash
python -m src.eval.runner
```

Mặc định:
- `--questions src/eval/questions.json`
- `--output    results/comparison.json`
- `--text_top_k 5 --image_top_k 3`

#### Tất cả tham số

```bash
python -m src.eval.runner \
    --questions src/eval/questions.json \
    --output results/comparison.json \
    --text_top_k 5 \
    --image_top_k 3 \
    --judge \
    --judge_model openai/gpt-4o-mini \
    --judge_target all \
    --skip_baselines \
    --skip_multi_agent \
    --resume
```

| Cờ | Giá trị | Tác dụng |
|---|---|---|
| `--questions` | path | File JSON câu hỏi (id, question, category, requires, expected_answer_hint, doc, source_page) |
| `--output` | path | File JSON lưu kết quả (incremental save sau mỗi câu) |
| `--text_top_k` | int (5) | Top-k text |
| `--image_top_k` | int (3) | Top-k image |
| `--judge` | flag | Bật LLM-as-a-Judge chấm 0-20 |
| `--judge_model` | str | Model judge (mặc định `openai/gpt-4o-mini`) |
| `--judge_target` | `all` / `baselines` / `multi_agent` | Chỉ chấm 1 nhóm modes |
| `--skip_baselines` | flag | Bỏ 3 modes đầu (chỉ chạy multi_agent) |
| `--skip_multi_agent` | flag | Bỏ multi_agent (chỉ chạy 3 baselines) |
| `--resume` | flag | Đọc output cũ, skip những câu/mode đã có `answer` |

#### Use case phổ biến

```bash
# Lần đầu, chạy đầy đủ 4 modes + judge
python -m src.eval.runner --judge

# Bị crash giữa chừng → resume
python -m src.eval.runner --judge --resume

# Chỉ test multi_agent (đã có 3 baselines)
python -m src.eval.runner --skip_baselines --judge --judge_target multi_agent --resume

# Đổi judge model sang Gemini
python -m src.eval.runner --judge --judge_model google/gemini-2.5-flash
```

### 9.2. LLM-as-a-Judge — [src/eval/judge.py](src/eval/judge.py)

`LLMJudge.evaluate(question, answer, expected_hint, context="")` chấm 4 tiêu chí (mỗi cái 0-5, total 0-20):

| Tiêu chí | Định nghĩa |
|---|---|
| **Correctness** | Khớp với `expected_answer_hint` không (số liệu đúng?) |
| **Completeness** | Bao phủ đủ các khía cạnh câu hỏi yêu cầu |
| **Relevance** | Bám sát câu hỏi, không lan man |
| **Faithfulness** | Trung thành với retrieved context — dựa trên dữ liệu, không hallucinate |

> **Context-aware Faithfulness (mới):** Nếu truyền tham số `context` (retrieved chunks), Judge sẽ kiểm tra câu trả lời có dựa trên context hay bịa thông tin (kiểu RAGAS). `ablation_agents.py` tự động build context string từ text + image chunks.

Output JSON:
```json
{
  "correctness": 4, "completeness": 5, "relevance": 5, "faithfulness": 4,
  "total": 18,
  "explanation": "..."
}
```

`evaluate_batch(items)` — chấm nhiều câu lần lượt, mỗi item có `question`, `answer`, `expected_hint`, `context` (optional).

`print_judge_summary(results)` in 3 phần: per-question, average per-mode, win-rate %.

### 9.3. Ablation — đóng góp từng agent

[`src/eval/ablation_agents.py`](src/eval/ablation_agents.py) so sánh 5 biến thể:

| Variant | Mô tả |
|---|---|
| `full` | Text + Image + Sum (3 agents, 3 LLM calls) |
| `no_text` | Bỏ TextAgent → Sum chỉ nhận image_output |
| `no_image` | Bỏ ImageAgent → Sum chỉ nhận text_output |
| `no_sum` | Ghép thô `analysis_text + " " + analysis_image` (không qua editor) |
| `single_llm` | 1 LLM call duy nhất qua `AnswerGenerator.generate(full_multimodal)` |

```bash
python -m src.eval.ablation_agents \
    --questions src/eval/question_test.json \
    --output    results/ablation_agents.json \
    --judge \
    --text_top_k 5 --image_top_k 3

# Chỉ chạy biến thể full (không ablation)
python -m src.eval.ablation_agents --only_full --judge
```

| Cờ | Giá trị | Tác dụng |
|---|---|---|
| `--only_full` | flag | Chỉ chạy variant `full`, bỏ 4 ablation |
| `--judge`, `--judge_model` | | Như runner.py |

Tối ưu: chỉ retrieve + chạy text/image agent **1 lần/câu**, sau đó gọi sum_agent với các tổ hợp khác nhau → tiết kiệm API call.

> **Context-aware judging:** `ablation_agents.py` tự build context string từ `text_chunks` + `image_chunks` (caption) rồi truyền vào `judge.evaluate(..., context=context_str)` để đánh giá Faithfulness chặt hơn.

### 9.4. Ablation — cách embed ảnh

[`src/eval/ablation_embed_image.py`](src/eval/ablation_embed_image.py) so sánh 3 phương pháp:

| Phương pháp | Vector |
|---|---|
| `caption_only` | embed text caption |
| `image_only` | embed image bytes |
| `aggregated` | weighted blend α=0.8 |

```bash
python -m src.eval.ablation_embed_image
python -m src.eval.ablation_embed_image --questions src/eval/questions.json
```

Đo: cosine similarity (query ↔ image vector) + retrieval rank.

### 9.5. Ablation — sweep α

[`src/eval/ablation_embed_weight.py`](src/eval/ablation_embed_weight.py) — quét α ∈ [0.0, 0.1, ..., 1.0]:

```bash
# Chạy ablation_embed_image trước (để có vectors)
python -m src.eval.ablation_embed_image

# Rồi sweep α (dùng vectors cached)
python -m src.eval.ablation_embed_weight
```

Output: bảng trực quan ASCII bar + best α + so sánh với α=0.2 (Gemini aggregated).

### 9.6. Ablation — Single vs Dual Encode

[`src/eval/runner_dual_encode.py`](src/eval/runner_dual_encode.py) so sánh:

| Variant | Encode |
|---|---|
| `single_encode` | text + image pipeline cùng dùng 1 query vector |
| `dual_encode` | image pipeline encode lại với prefix "biểu đồ hình ảnh..." |

Chạy 4 tổ hợp: `{full_multimodal, multi_agent} × {single, dual}`.

```bash
python -m src.eval.runner_dual_encode \
    --questions src/eval/questions.json \
    --output    results/ablation_dual_encode.json \
    --text_top_k 5 --image_top_k 3
```

> **Lưu ý:** code này gọi `retriever.retrieve(..., dual_encode=...)`. Nếu phiên bản hiện tại của `DualRetriever.retrieve()` chưa có tham số `dual_encode`, cần thêm vào hoặc bỏ tham số này khi chạy.

### 9.7. Retrieval-level metrics

[`src/eval/eval_retrieval.py`](src/eval/eval_retrieval.py) — đánh giá retrieval thuần (không qua LLM):

3 thí nghiệm:

1. **Single vs Dual Collection**
   - `dual` — text(5) + image(3) riêng
   - `text_only` — chỉ text(8)
   - `merged` — gộp text+image, sort theo score, top-8
2. **Dense Search Quality** (BM25 đã loại nên chỉ so có/không boost)
   - `dense_with_boost` (current)
   - `dense_no_boost`
3. **Ticker-aware vs Không ticker**
   - `with_ticker` — boost text + filter image
   - `without_ticker` — pure dense

Metrics: Precision@k, Recall@k, Hit Rate@k, MRR.

```bash
python -m src.eval.eval_retrieval
python -m src.eval.eval_retrieval --questions src/eval/questions.json
```

Yêu cầu `questions.json` có:
- `requires`: `["text"]`, `["table"]`, `["image"]`, `["text", "image"]`, ...
- `doc`: tên doc đúng (substring check)
- `source_page`: trang đúng

### 9.8. Đánh giá Image Pipeline

[`src/eval/eval_image_pipeline.py`](src/eval/eval_image_pipeline.py) — chứng minh image pipeline:
1. Giúp tìm context cho câu hỏi visual/cross_modal
2. Không làm giảm retrieval cho câu hỏi text/table

Chia câu hỏi thành 2 nhóm:
- `needs_image`: requires có "image"
- `text_only_q`: requires KHÔNG có "image"

So sánh 2 chiến lược:
- `with_image`: Dual pipeline — text(5) + image(3)
- `without_image`: chỉ text(8)

```bash
python -m src.eval.eval_image_pipeline
python -m src.eval.eval_image_pipeline --questions src/eval/questions.json
```

### 9.9. Format `questions.json`

```json
[
  {
    "id":     "q001",
    "question": "Biên lợi nhuận gộp của VNM trong Q4/2025?",
    "category": "text_lookup",
    "requires": ["text", "table"],
    "doc":      "VNM_2026_02_24",
    "source_page": 5,
    "expected_answer_hint": "GPM Q4/2025 là 40,4%, tăng từ 39,2% Q3"
  },
  ...
]
```

#### Phân loại Category (đã chuẩn hóa)

| Category | Mô tả | Ví dụ |
|---|---|---|
| `text_lookup` | Tra cứu trực tiếp từ text | "EPS của VNM là bao nhiêu?" |
| `table_lookup` | Tra cứu từ bảng số liệu | "Doanh thu theo quý trong bảng KQKD?" |
| `numerical_reasoning` | Tính toán, so sánh số liệu | "GPM thay đổi bao nhiêu % so với cùng kỳ?" |
| `multi_hop_reasoning` | Suy luận đa bước, cross-doc | "So sánh NIM của HDB và CTG" |
| `visual_analysis` | Phân tích biểu đồ/hình ảnh | "Xu hướng giá cổ phiếu theo biểu đồ?" |
| `cross_modal` | Kết hợp text + image | "Doanh thu trong bảng khớp với biểu đồ không?" |

> **Lưu ý:** Các nhãn cũ (`factual`, `visual`, `comparative`, `reasoning`) đã được chuyển đổi sang hệ thống mới. Khi thêm câu hỏi, dùng các nhãn trong bảng trên.

#### Các file câu hỏi hiện có

| File | Số câu | Mô tả |
|---|---|---|
| `questions.json` | 265 | Bộ chính, đa dạng ticker và loại câu hỏi |
| `questions_hard.json` | 100 | Câu hỏi khó: multi-hop, numerical reasoning, visual analysis |
| `questions_151_250.json` | 100 | Extended eval, bổ sung thêm |
| `questions_sab_ree_pvt_pvs.json` | ~40 | SAB, REE, PVT, PVS |
| `questions_vcb_tpb_tng_stb.json` | ~40 | VCB, TPB, TNG, STB |
| `questions_vpb_vnm_vhm_vea.json` | ~45 | VPB, VNM, VHM, VEA |
| `question_test.json` | ~15 | Bộ nhỏ để test nhanh |

---

## 10. Bảng tham chiếu nhanh các lệnh Python

### 10.1. Workflow đầy đủ từ đầu

```bash
# 0. Setup
python -m venv ragenv && source ragenv/bin/activate
pip install -r requirements.txt
# Tạo .env với GOOGLE_API_KEY + OPENROUTER_API_KEY

# 1. Đặt PDF vào pdfs/raw/, chạy extraction
python -m src.extraction.run_extraction

# 2. Embedding (gồm caption ảnh + dual pipeline)
python -m src.embedding.run_embedding

# 3a. (nếu lỗi 429/503) Re-caption những ảnh fallback
python -m src.embedding.fix_captions --dry-run     # xem trước
python -m src.embedding.fix_captions               # sửa thật

# 3b. (tùy chọn) Re-embed với α khác
python -m src.embedding.reembed_images --alpha 0.8

# 4. Test retrieval
python src/retrieval/run_retrieval.py --query "biểu đồ giá VNM"

# 5. Hỏi đáp CLI
python ask.py
python ask.py --q "Biên lợi nhuận VNM thay đổi thế nào?"

# 6. Web UI
streamlit run app.py
```

### 10.2. Workflow đánh giá

```bash
# Chạy 4 modes + judge
python -m src.eval.runner --judge

# Resume nếu bị ngắt
python -m src.eval.runner --judge --resume

# Chỉ chấm điểm lại (đã có answers)
python -m src.eval.runner --judge --resume --skip_baselines --skip_multi_agent

# Ablation 5 biến thể agent
python -m src.eval.ablation_agents --judge

# Ablation cách embed ảnh
python -m src.eval.ablation_embed_image
python -m src.eval.ablation_embed_weight

# Ablation single vs dual encode
python -m src.eval.runner_dual_encode

# Retrieval metrics
python -m src.eval.eval_retrieval
python -m src.eval.eval_image_pipeline
```

### 10.3. Bảng tóm tắt tất cả entrypoints

| Lệnh | File | Vai trò | Cờ chính |
|---|---|---|---|
| `python -m src.extraction.run_extraction` | [run_extraction.py](src/extraction/run_extraction.py) | PDF → metadata.json | (không) |
| `python -m src.embedding.run_embedding` | [run_embedding.py](src/embedding/run_embedding.py) | metadata → ChromaDB | (không) |
| `python -m src.embedding.fix_captions` | [fix_captions.py](src/embedding/fix_captions.py) | Re-caption ảnh fallback | `--dry-run` |
| `python -m src.embedding.reembed_images` | [reembed_images.py](src/embedding/reembed_images.py) | Re-embed với α mới | `--alpha`, `--dry-run`, `--ticker` |
| `python src/retrieval/run_retrieval.py` | [run_retrieval.py](src/retrieval/run_retrieval.py) | Test retrieval | `--query`, `--text_top_k`, `--image_top_k` |
| `python ask.py` | [ask.py](ask.py) | CLI hỏi đáp | `--q`, `--text_top_k`, `--image_top_k` |
| `streamlit run app.py` | [app.py](app.py) | Web UI | (UI controls) |
| `python -m src.eval.runner` | [runner.py](src/eval/runner.py) | So sánh 4 modes | `--judge`, `--resume`, `--skip_*`, `--judge_target` |
| `python -m src.eval.runner_dual_encode` | [runner_dual_encode.py](src/eval/runner_dual_encode.py) | Single vs Dual encode | `--questions`, `--output`, `--*_top_k` |
| `python -m src.eval.ablation_agents` | [ablation_agents.py](src/eval/ablation_agents.py) | 5 variants 3-agent | `--judge`, `--only_full` |
| `python -m src.eval.ablation_embed_image` | [ablation_embed_image.py](src/eval/ablation_embed_image.py) | caption_only/image_only/aggregated | `--questions` |
| `python -m src.eval.ablation_embed_weight` | [ablation_embed_weight.py](src/eval/ablation_embed_weight.py) | Sweep α | (không) |
| `python -m src.eval.eval_retrieval` | [eval_retrieval.py](src/eval/eval_retrieval.py) | Precision/Recall/MRR | `--questions` |
| `python -m src.eval.eval_image_pipeline` | [eval_image_pipeline.py](src/eval/eval_image_pipeline.py) | with_image vs without_image | `--questions` |

### 10.4. Debug & monitoring

```bash
# Kiểm tra ChromaDB count
python -c "import chromadb; c = chromadb.PersistentClient('pdfs/processed/vector_db'); \
print('text:', c.get_collection('rag_text').count()); \
print('image:', c.get_collection('rag_image').count())"

# Xem 1 chunk metadata
python -c "import json; m = json.load(open('pdfs/processed/metadata.json')); \
print(json.dumps(m[0], ensure_ascii=False, indent=2))"

# Đếm chunks theo loại
python -c "import json; m = json.load(open('pdfs/processed/metadata.json')); \
from collections import Counter; print(Counter(c['chunk_type'] for c in m))"

# Đếm tickers
python -c "import json; m = json.load(open('pdfs/processed/metadata.json')); \
from collections import Counter; print(Counter(c['ticker'] for c in m if c.get('ticker')))"
```

---

## Phụ lục — Luồng gọi hàm tổng thể

### A.1. Khi user hỏi 1 câu (qua `ask.py`)

```
ask.main()
 ├── DualRetriever(chroma_path, api_key, cfg)
 │    ├── chromadb.PersistentClient(path)
 │    ├── client.get_collection("rag_text")
 │    ├── client.get_collection("rag_image")
 │    └── _load_corpus()
 │         └── _known_tickers ← từ metadatas của 2 collections
 │
 ├── MultiAgentOrchestrator(api_key, text_model, vision_model)
 │    ├── OpenAI(api_key, base_url=openrouter.ai/api/v1)
 │    ├── TextAgent(client, text_model)
 │    ├── ImageAgent(client, vision_model)
 │    └── SumAgent(client, text_model)
 │
 └── process_question(question, retriever, orchestrator)
      ├── retriever.retrieve(q, text_top_k=5, image_top_k=3)
      │    ├── _embed_query(q)                 [Gemini, có cache]
      │    ├── _detect_ticker(q)               [exact / alias]
      │    ├── _text_pipeline()
      │    │    ├── _dense_search(text_col, top_n=20)
      │    │    └── _ticker_boost(boost=1.5)
      │    ├── _infer_ticker_from_chunks()     [majority vote]
      │    └── _image_pipeline()
      │         ├── Phase 1: _dense_search(where={"ticker": X})
      │         └── Phase 2: _dense_search(no filter, dedup)
      │              + filter score >= 0.015
      │
      └── orchestrator.run(q, result)
           ├── ThreadPoolExecutor(max_workers=2):
           │    ├── text_agent.analyze()
           │    │    ├── _format_chunks()
           │    │    └── _call_llm() → JSON {analysis, confidence, ...}
           │    └── image_agent.analyze()
           │         ├── _encode_image() [resize 1280px → base64]
           │         └── _call_vision() → JSON {analysis, confidence, ...}
           │
           └── sum_agent.synthesize()
                ├── draft = " ".join(text.analysis, image.analysis)
                └── _call_llm() → {answer, confidence, reasoning}
```

### A.2. Khi chạy `runner.py --judge`

```
run_comparison()
 ├── DualRetriever (1 instance)
 ├── 3 wrapper retrievers: TextOnly, TextTable, FullMultimodal
 ├── AnswerGenerator (cho 3 modes đầu)
 ├── MultiAgentOrchestrator (cho mode 4 — 3 agents)
 ├── LLMJudge (nếu --judge)
 │
 └── for each question:
      ├── for mode in [text_only, text_table, full_multimodal]:
      │    ├── retriever.retrieve()
      │    ├── generator.generate(question, result, mode)
      │    └── save text_chunks + image_chunks vào output JSON
      │
      ├── multi_agent:
      │    ├── base.retrieve()
      │    ├── orchestrator.run()  (3 LLM calls: Text + Image || Sum)
      │    └── save chunks + agent_outputs vào output JSON
      │
      └── for mode in MODES_TO_JUDGE:
           └── judge.evaluate(q, answer, expected_hint, context=...)
                → {correctness, completeness, relevance, faithfulness, total, explanation}
```

---

## Phụ lục — Tham số quan trọng & ý nghĩa

| Tham số | Vị trí | Mặc định | Ý nghĩa |
|---|---|---|---|
| `images_dpi` | extraction | 216 | DPI ảnh chart (216/72=3.0 → ảnh sắc nét) |
| `CHUNK_SIZE` | extractor.py | 400 | Tokens / text chunk |
| `CHUNK_OVERLAP` | extractor.py | 80 | Overlap giữa chunks |
| `MIN_CHART_SIZE_PT` | extractor.py | 80 | Loại ảnh < 80pt (logo) |
| `min_token_count` | embedding | 50 | Bỏ text chunks quá ngắn |
| `batch_size` | embedding | 10 | Số text chunks / API call |
| `sleep_between_batches` | embedding | 1.5 | Delay (s) giữa batches |
| `alpha` | embed_image | 0.8 | Caption weight (0=image_only, 1=caption_only) |
| `text_top_k` | retrieval | 5 | Top-k text |
| `image_top_k` | retrieval | 3 | Top-k image |
| `IMAGE_MIN_SCORE` | retriever.py | 0.015 | Lọc ảnh score thấp |
| `_ticker_boost` | retriever.py | 1.5× | Soft boost cho text cùng ticker |
| `max_long_edge` | _encode_image | 1280px | Resize ảnh trước khi gửi vision LLM |
| `max_workers` | orchestrator | 2 | Parallel TextAgent + ImageAgent |
| `temperature` | agents | 0.1 | Low temperature → deterministic |
| `max_tokens` | text/image agent | 1500 | Sum agent: 2000 |
