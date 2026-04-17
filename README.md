# Multimodal RAG for Financial Report Analysis

A **Multimodal Retrieval-Augmented Generation (RAG)** system for analyzing Vietnamese corporate financial reports. Combines processing of **text**, **tables**, and **charts/images** to answer financial questions accurately.

> Graduation thesis project (Do an Tot nghiep)

## System Architecture

```
PDF Reports  -->  Extraction (Docling)  -->  Embedding (Gemini)  -->  ChromaDB
                   |                          |
                   text / table / image       Dual Pipeline:
                                              - rag_text  (text + table)
                                              - rag_image (caption + image blend)

Question  -->  Dual Retrieval  -->  Multi-Agent Reasoning (4 agents)  -->  Answer
               |                     |
               text: dense+boost     Critical -> Text + Image -> Sum (Editor)
               image: dense+filter
```

### Multi-Agent Pipeline (4 Agents)

| Stage | Agent | Role |
|-------|-------|------|
| 1 | **Critical Agent** | Identifies key information, guides other agents |
| 2a | **Text Agent** | Analyzes text & tables with self-correction |
| 2b | **Image Agent** | Analyzes charts via Vision LLM (OCR, trend detection) |
| 3 | **Sum Agent** | Edits & synthesizes, resolves conflicts, preserves all facts |

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
│   │   └── run_embedding.py
│   ├── retrieval/         # Dual retrieval (text + image)
│   │   ├── retriever.py
│   │   ├── query_router.py
│   │   └── run_retrieval.py
│   ├── agents/            # Multi-Agent reasoning
│   │   ├── agents.py
│   │   └── orchestrator.py
│   └── eval/              # Evaluation & ablation studies
│       ├── judge.py
│       ├── runner.py
│       ├── questions.json
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

- **GOOGLE_API_KEY** — for Gemini Embedding
- **OPENROUTER_API_KEY** — for LLM agents (Gemini Flash/Pro via OpenRouter)

## Usage

### Step 1: Extract PDFs

Place financial report PDFs in `pdfs/raw/`, then run:

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
python ask.py --q "How has VNM's gross profit margin changed recently?"
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
| Vector DB | ChromaDB |
| LLM | Gemini 2.5 Flash (text), Gemini 2.5 Pro (vision) |
| Retrieval | Dense retrieval + ticker-aware scoring |
| Web UI | Streamlit |

## Evaluation

The system supports multiple evaluation modes:

- **4-mode comparison** — text-only, image-only, multimodal, multimodal + agents
- **LLM-as-a-Judge** — 4 criteria: correctness, completeness, relevance, coherence
- **Ablation studies** — dual encode, image embedding, agent pipeline
- **Retrieval metrics** — Precision, Recall, MRR, Hit Rate

Run evaluation:

```bash
python -m src.eval.runner
```
