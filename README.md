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

Question  -->  Dual Retrieval  -->  Reranker  -->  3-Agent Reasoning  -->  Answer
               |                     |              |
               text: dense+boost     Gemini Flash   Text + Image (parallel) → Sum
               image: dense+filter   or LoRA CE ◄── Fine-tuned cross-encoder (local)
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
├── scripts/
│   ├── generate_reranker_data.py  # Tạo training data cho LoRA reranker
│   └── finetune_reranker.py       # Fine-tune cross-encoder với LoRA
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
│   │   ├── retriever.py   # DualRetriever (supports gemini/lora reranker)
│   │   ├── reranker.py    # GeminiReranker (LLM-based, via API)
│   │   ├── lora_reranker.py  # LoRAReranker (fine-tuned cross-encoder, local)
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
│       ├── eval_lora_reranker.py  # Dense vs Gemini vs LoRA reranker
│       ├── eval_e2e_lora.py       # End-to-end LLM-as-Judge comparison
│       ├── questions.json               # Main eval dataset (200 questions)
│       ├── questions_hard.json          # Hard reasoning questions (100 questions)
│       ├── questions_151_250.json       # Extended eval questions
│       ├── question_test.json           # Small test set for quick runs
│       └── ...
├── models/
│   └── reranker_lora/     # LoRA adapter weights (after fine-tuning)
├── training_data/         # Generated training pairs for reranker
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
| Reranking | Gemini Flash (API) or **LoRA fine-tuned cross-encoder** (local) |
| Fine-tuning | LoRA/PEFT on `BAAI/bge-reranker-v2-m3`, PyTorch |
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

## Fine-tuning: LoRA Cross-Encoder Reranker

The system supports fine-tuning a local cross-encoder reranker to replace the Gemini Flash API-based reranker, reducing latency (~10x) and eliminating per-query API costs.

**Base model:** `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa, 568M params, multilingual)
**Method:** LoRA (Low-Rank Adaptation) — trains only ~0.5% of parameters
**Training data:** Auto-generated from 300 eval questions + ChromaDB (4350 pairs)

### Step 1: Generate training data

```bash
python scripts/generate_reranker_data.py
# Output: training_data/reranker_pairs.jsonl (4350 pairs, ratio 1:3)
```

### Step 2: Fine-tune with LoRA

```bash
# Mac (MPS) — batch_size=2, gradient checkpointing enabled
python -m scripts.finetune_reranker --epochs 3

# CUDA GPU — larger batch possible
python -m scripts.finetune_reranker --epochs 3 --batch_size 16 --lora_r 16
```

Output: LoRA adapter weights → `models/reranker_lora/`

### Step 3: Switch to LoRA reranker

Edit `config/config.yaml`:

```yaml
retrieval:
  reranker_type: "lora"   # "gemini" (API) → "lora" (local)
```

### Step 4: Evaluate

```bash
# Retrieval-level: Dense vs Gemini vs LoRA (Precision@k, MRR, Latency)
python -m src.eval.eval_lora_reranker --questions src/eval/questions_hard.json

# End-to-end: LLM-as-Judge comparison (0-20 scale)
python -m src.eval.eval_e2e_lora --questions src/eval/questions_hard.json --n 20
```
