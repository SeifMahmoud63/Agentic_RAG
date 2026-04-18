<!-- <!-- # 🚀 Agentic RAG: High-Performance Hybrid Search & Semantic Cache System

A state-of-the-art Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, and **Qdrant**. This project features an advanced agentic pipeline designed for low latency, high precision, and incremental data synchronization.

---

## 🏗️ Core Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM Engine** | **Google Gemini 1.5 Flash** | Optimized for speed and high rate-limits. |
| **Vector Database** | **Qdrant** | Support for dense and sparse vectors via SPLADE. |
| **Semantic Cache** | **Redis** | Ultra-low latency query result caching. |
| **Sparse Embeddings** | **SPLADE (via FastEmbed)** | Local HuggingFace models for keyword-aware retrieval. |
| **Reranker** | **FlashRank** | Light-weight reranking using `TinyBERT-L-2-v2`. |
| **Orchestration** | **LangGraph** | Complex agentic workflows and message states. |

---

## 🌊 Advanced Pipelines

### 📂 1. Ingestion & Identity Resolution
Unlike standard RAG systems, this pipeline uses intelligence to prevent data duplication:
1.  **SHA-256 Hashing**: Every file is hashed upon upload to detect exact duplicates.
2.  **Identity Resolution**: If a file has 35%+ content overlap (`DUPLICATE_THRESHOLD=0.35`) with an existing document, it is treated as a **New Version** rather than a duplicate.
3.  **Incremental Sync**: Only changes (new/modified chunks) are updated in Qdrant, saving API costs and indexing time.

### 🔍 2. Retrieval & Generation
The search pipeline uses a **Multi-Step Refinement** process:
1.  **Parallel Multi-Query**: Simultaneously performs **Query Rewriting** and **HyDE (Hypothetical Document Embeddings)** generation using Gemini to capture intent.
2.  **Hybrid Search**: Executes a concurrent search in Qdrant:
    *   **Dense**: Semantic meaning (3072-dim).
    *   **Sparse**: Keyword matching (SPLADE).
3.  **RRF Fusion & Reranking**: Results are fused and then reranked by **FlashRank** to ensure the Top-K results are strictly relevant.

### ⚡ 3. Manual Semantic Caching
To minimize LLM costs and latency, we implemented a custom semantic cache:
*   **Relevance Threshold**: `0.88` Cosine Similarity. If a new query is 88%+ similar to a cached query, the system returns the answer in **milli-seconds**.
*   **TTL (Time To Live)**: **1 Hour**. Cached entries automatically expire to ensure information freshness.
*   **Performance**: O(1) Redis lookup, bypassing the entire LLM and Vector Search overhead.

---

## 🐳 Getting Started (Docker)

The project is fully containerized for cross-platform compatibility.

### 1. Prerequisites
- Docker & Docker Compose
- Google Gemini API Key

### 2. Setup environment
Copy `src/.env.example` to `src/.env` and fill in your API keys:
```bash
GOOGLE_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_bot_token
```

### 3. Run the project
```bash
docker-compose up --build -d
```
All services (API, Bot, Redis, Qdrant) will start automatically.

---

## 📊 Evaluation & Quality Control

The system is continuously evaluated using the **Ragas** framework to ensure production readiness:
*   **Faithfulness**: Measures if the answer is derived solely from the retrieved context (no hallucinations).
*   **Response Relevancy**: Measures how well the response addresses the user's specific query.

---

## 📈 Performance Notes
*   **Latency**: Average retrieval time is sub-2 seconds (excluding parallel LLM calls).
*   **Cold Start**: The first request may take longer as SPLADE models are loaded into memory from HuggingFace.
*   **Scalability**: Built to handle thousands of documents through Qdrant's payload indexing and SQLite-backed metadata management.

---

> [!IMPORTANT]
> This project is standardized to use **snake_case** for all file and directory paths, ensuring seamless deployment on Linux, Windows, and Docker environments. -->

# 🚀 Agentic RAG: High-Performance Hybrid Search & Semantic Cache System

> A state-of-the-art Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, and **Qdrant**. Features an advanced agentic pipeline designed for low latency, high precision, and incremental data synchronization.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-6B48FF?style=flat-square)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=flat-square)
![Redis](https://img.shields.io/badge/Redis-Semantic_Cache-FF4438?style=flat-square&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)

---
<p align="center">
  <img src="assets/banner.svg" alt="Agentic RAG — Telegram Bot Banner" width="100%"/>
</p>



## 🏗️ Core Technology Stack

| Component | Technology | Description |
|---|---|---|
| **LLM Engine** | Google Gemini 1.5 Flash | Optimized for speed and high rate-limits |
| **Vector Database** | Qdrant | Support for dense and sparse vectors via SPLADE |
| **Semantic Cache** | Redis | Ultra-low latency query result caching |
| **Sparse Embeddings** | SPLADE (via FastEmbed) | Local HuggingFace models for keyword-aware retrieval |
| **Reranker** | FlashRank | Lightweight reranking using `TinyBERT-L-2-v2` |
| **Orchestration** | LangGraph | Complex agentic workflows and message states |
| **Telegram Integration** | Voice STT Support | Native Speech-to-Text for voice message processing |

---

## 🌊 Advanced Pipelines

### 📂 1. Intelligent Ingestion & Identity Resolution

Our pipeline ensures **zero redundancy** through a tiered validation process:

1. **File Hashing (SHA-256)** — Generates a unique hash for each file. If the hash already exists in the database, ingestion is skipped entirely to save resources.
2. **Chunking & Overlap Analysis** — If the file is new, it is broken into overlapping chunks for semantic granularity.
3. **Identity Resolution (The 35% Rule)** — New content is compared against existing data. A `DUPLICATE_THRESHOLD=0.35` overlap triggers versioning rather than fresh ingestion, ensuring context continuity.
4. **Incremental Sync** — Only modified chunks are updated in Qdrant, keeping the index lean and consistent.

---

### 🔍 2. Retrieval & Generation

1. **Parallel Multi-Query** — Simultaneously performs Query Rewriting and HyDE (Hypothetical Document Embedding) generation.
2. **Hybrid Search** — Executes concurrent Dense + Sparse (SPLADE) retrieval for broader semantic and keyword coverage.
3. **RRF Fusion & Reranking** — Reciprocal Rank Fusion merges results; FlashRank enforces that Top-K results are strictly relevant.

---

### ⚡ 3. Semantic Caching (Powered by Redis)

To minimize LLM costs and latency, Redis is integrated for sophisticated semantic caching:

- **Relevance Threshold:** `0.88` Cosine Similarity — if a query matches a cached entry at 88%+, the system responds in milliseconds.
- **Performance:** Bypasses the entire LLM and Vector Search overhead using O(1) lookups.

---

## 📊 Evaluation & Quality Control

The system is continuously monitored using the **Ragas** framework:

| Metric | Score | Previous | Improvement |
|---|---|---|---|
| **Faithfulness** | 100% | 30% | +70% via context injection refinement |
| **Response Relevancy** | 92% | 20% | +72% via reranking + query expansion |

---

## 🤖 Telegram Bot Features

- **Text Interaction** — Seamless RAG-based conversational interface.
- **Voice Messages (STT)** — Integrated Speech-to-Text: voice notes are transcribed and processed through the full RAG pipeline.

---

## 🐳 Getting Started (Docker)

### 1. Configure environment variables

```bash
# .env
GOOGLE_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_bot_token
```

### 2. Build and run all services

```bash
docker-compose up --build -d
```

> [!TIP]
> The combination of SHA-256 hashing and the 35% overlap rule makes this system extremely cost-efficient for enterprise-level document management.

---

## 📁 Project Structure

```
src/
├── agent/                      # Autonomous State Machine (LangGraph)
│   ├── graph.py                # Workflow logic & node definitions
│   ├── tools.py                # Retrieval tools for the agent
│   └── system_prompt.txt       # Core instructions for the LLM
├── assets/                     # Local storage for uploaded documents
├── bot/                        # Client Interfaces
│   └── telegram_bot.py         # Telegram bot handler & integration
├── controllers/                # Request & Workflow Logic
│   ├── base_controller.py      # Shared path/file utilities
│   ├── data_controller.py      # Upload validation & naming
│   ├── project_controller.py   # Multi-project path management
│   └── process_controller.py   # Ingestion/Chunking orchestration
├── embeddingmodel/             # AI Vector Logic
│   └── emb_model.py            # Dense embedding model (Gemini/HF)
├── evaluationragas/            # Testing & Metrics
│   ├── evaluation.py           # Ragas scoring orchestration
│   └── test_data.txt           # Golden test set for quality checks
├── helpers/                    # Core Utilities
│   ├── config.py               # Pydantic Settings & .env loading
│   ├── hash_utils.py           # SHA-256 logic for deduplication
│   ├── redis.py                # Semantic cache implementation
│   └── clean_response.py       # LLM output post-processing
├── llm/                        # Language Model Config
│   └── llm.py                  # Provider setup (Groq/Google)
├── logs/                       # System Observability
│   └── logger.py               # Standardized application logging
├── models/                     # Shared Data Models
│   └── signals.py              # Standardized response enums
├── queryschema/                # API Validation Models
├── retriever/                  # Search Engine Logic
│   └── retrieve_chunks.py      # Hybrid + Reranker implementation
├── routes/                     # API Endpoints (FastAPI)
│   ├── base.py                 # Health checks & system routes
│   └── data.py                 # Ingestion, Ask_Q & Eval routes
├── vectordatabase/             # Persistence Layer
│   ├── ingestion_service.py    # Master sync/upsert flow
│   ├── qdrant_db.py            # Vector storage & Hybrid search
│   └── metadata_store.py       # SQLite registry for file versions
├── main.py                     # Application Entry Point
├── .env                        # Environment Secrets (DO NOT COMMIT)
└── requirements.txt            # Python Dependencies
```

---

## 🧠 Deep Dive: The 35% Identity Resolution Threshold

> **Config key:** `DUPLICATE_THRESHOLD: 0.35` in `helpers/config.py`

### The Problem — "Ghost Files" in RAG

In a naive RAG system, uploading `Report_v1.pdf` and then `Report_v2.pdf` (with minor changes) creates two independent entries. Search results then surface the same information twice — stale data polluting every query response and wasting vector DB space.

### How the Threshold Works

When a new file is uploaded, the system performs a **Global Content Scan** rather than a simple filename check:

1. The new file is broken into chunks.
2. A SHA-256 hash (fingerprint) is computed for every chunk.
3. The database is queried: *"How many of these chunk-hashes already exist in other files?"*

| Overlap | Decision | Action |
|---|---|---|
| **< 35%** of chunks match | Brand new document | Creates a fresh entry with `v1` metadata |
| **≥ 35%** of chunks match | Updated version detected | Triggers **Incremental Sync** against the matched file |

### Incremental Sync — What Actually Happens

When the 35% threshold is met, instead of a full re-upload:

- ✅ **Identical chunks** are kept as-is (no recompute)
- ➕ **New chunks** are embedded and inserted
- 🗑️ **Removed chunks** are deleted from Qdrant
- 🔖 **Version counter** is incremented in `metadata_store.py` (e.g., `v1 → v2`)

### Why 35%?

It is a deliberate sweet spot:

- **High enough** to avoid accidentally merging two different documents that share common boilerplate sentences.
- **Low enough** to catch a document even after a major rewrite covering more than half its pages.

> [!TIP]
> **Tuning guide:** If the system incorrectly merges distinct files, raise the threshold (e.g., `0.5`). If it fails to detect updates to heavily revised documents, lower it (e.g., `0.25`).

---

## 📄 License

This project is licensed under the MIT License.