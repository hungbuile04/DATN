# Multimodal RAG for Financial Report Analysis

A **Multimodal Retrieval-Augmented Generation (RAG)** system for analyzing Vietnamese corporate financial reports. Combines processing of **text**, **tables**, and **charts/images** to answer financial questions accurately.

> Graduation thesis project (Đồ án Tốt nghiệp)

## System Architecture

```
PDF Reports  -->  Extraction (Docling)  -->  Embedding (Gemini)  -->  ChromaDB
                   |                          |
                   text / table / image       Dual Pipeline:
                                              - rag_text  (text + table)
                                              - rag_image (caption + image blend α=0.8)

Question  -->  Dual Retrieval  -->  3-Agent Reasoning  -->  Answer
               |                     |
               text: dense+boost     Text + Image (parallel) → Sum (Editor)
               image: dense+filter
```

### Multi-Agent Pipeline (3 Agents, 2 Stages)

| Stage | Agent | Role |
|-------|-------|------|
| 1a (parallel) | **Text Agent** | Analyzes text & tables, self-correction, entity resolution |
| 1b (parallel) | **Image Agent** | Analyzes charts via Vision LLM (OCR, trend detection) |
| 2 | **Sum Agent** | Editor — merges drafts, resolves conflicts, preserves all facts |

> Stage 1 runs **parallel** via `ThreadPoolExecutor` → ~40% latency reduction. Total: **3 LLM calls/question**.

## Project Structure

```
source_code_2/
├── app.py                 # Streamlit web UI
├── ask.py                 # CLI question answering
├── config/
│   ├── config.yaml        # System configuration
│   └── settings.py        # Loads config as Python dict
├── src/
│   ├── extraction/        # PDF extraction (Docling + figure detection)
│   │   ├── extractor.py
│   │   ├── figure_extractor.py
│   │   └── run_extraction.py
│   ├── embedding/         # Embedding & ChromaDB storage
│   │   ├── embedder.py
│   │   ├── run_embedding.py
│   │   ├── fix_captions.py
│   │   └── reembed_images.py
│   ├── retrieval/         # Dual retrieval (text + image)
│   │   ├── retriever.py
│   │   ├── query_router.py
│   │   └── run_retrieval.py
│   ├── agents/            # 3-Agent reasoning
│   │   ├── agents.py      # TextAgent, ImageAgent, SumAgent
│   │   └── orchestrator.py
│   └── eval/              # Evaluation & ablation studies
│       ├── judge.py       # LLM-as-a-Judge (4 criteria, 0-20)
│       ├── runner.py      # 4-mode comparison
│       ├── ablation_agents.py
│       ├── eval_retrieval.py
│       ├── questions.json               # Main eval dataset (265 questions)
│       ├── questions_hard.json          # Hard reasoning questions (100 questions)
│       ├── questions_151_250.json       # Extended eval questions
│       ├── questions_sab_ree_pvt_pvs.json
│       ├── questions_vcb_tpb_tng_stb.json
│       ├── questions_vpb_vnm_vhm_vea.json
│       ├── question_test.json           # Small test set for quick runs
│       └── ...
├── pdfs/
│   ├── raw/               # Raw financial report PDFs
│   └── processed/         # Metadata, vector DB (git-ignored)
├── results/               # Evaluation results (git-ignored)
├── requirements.txt
└── .env                   # API keys (git-ignored)
```

## Setup

### 1. Create virtual environment

```bash
python -m venv ragenv
source ragenv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

- **GOOGLE_API_KEY** — for Gemini Embedding 2 + Gemini Vision (image captioning)
- **OPENROUTER_API_KEY** — for LLM agents (Gemini 2.5 Flash via OpenRouter)

## Usage

### Step 1: Extract PDFs

Place financial report PDFs in `pdfs/raw/` (filename format: `TICKER_YYYY_MM_DD_SOURCE.pdf`), then run:

```bash
python -m src.extraction.run_extraction
```

### Step 2: Build embeddings

```bash
python -m src.embedding.run_embedding
```

### Step 3: Ask questions

**CLI:**

```bash
# Interactive mode
python ask.py

# Single question
python ask.py --q "Biên lợi nhuận gộp VNM thay đổi thế nào?"
```

**Web UI (Streamlit):**

```bash
streamlit run app.py
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| PDF Extraction | Docling, PyMuPDF |
| Embedding | Gemini Embedding 2 (3072-dim) |
| Vector DB | ChromaDB (Dual collection: `rag_text` + `rag_image`) |
| LLM | Gemini 2.5 Flash (text + vision, via OpenRouter) |
| Retrieval | Dense-only + ticker-aware scoring (BM25 removed) |
| Image Embedding | Aggregated blend: α×caption + (1-α)×image, α=0.8 |
| Web UI | Streamlit |

## Evaluation

The system supports comprehensive evaluation:

- **4-mode comparison** — text-only, text+table, full multimodal (1 LLM), multi-agent (3 LLM)
- **LLM-as-a-Judge** — 4 criteria: correctness, completeness, relevance, faithfulness (0-20 scale)
- **Context-aware judging** — Judge receives retrieved context to evaluate faithfulness (RAGAS-style)
- **Ablation studies** — agent contribution, image embedding method, α sweep, single vs dual encode
- **Retrieval metrics** — Precision@k, Recall@k, MRR, Hit Rate@k

### Question Categories

| Category | Description |
|----------|-------------|
| `text_lookup` | Direct fact lookup from text |
| `table_lookup` | Lookup from table data |
| `numerical_reasoning` | Calculation and numerical comparison |
| `multi_hop_reasoning` | Cross-document or multi-step reasoning |
| `visual_analysis` | Chart/image-based analysis |
| `cross_modal` | Requires both text and image |

### Run evaluation

```bash
# Full 4-mode comparison + judge scoring
python -m src.eval.runner --judge

# Resume if interrupted
python -m src.eval.runner --judge --resume

# Only multi-agent mode
python -m src.eval.runner --skip_baselines --judge --judge_target multi_agent --resume

# Ablation: agent contribution (5 variants)
python -m src.eval.ablation_agents --questions src/eval/question_test.json --judge

# Retrieval-level metrics
python -m src.eval.eval_retrieval
```
