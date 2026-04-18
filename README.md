# 🚀 Agentic RAG: High-Performance Hybrid Search & Semantic Cache System

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
> This project is standardized to use **snake_case** for all file and directory paths, ensuring seamless deployment on Linux, Windows, and Docker environments.