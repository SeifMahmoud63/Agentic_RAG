# 🚀 Advanced RAG MVC Framework
Welcome to your **High-Performance Retrieval-Augmented Generation (RAG)** system. This isn't just a basic RAG; it's a production-ready pipeline designed for speed, accuracy, and scalability.
---
## 🧠 System Architecture: How it Works
Your project follows the **MVC (Model-View-Controller)** pattern, separating business logic from infrastructure.
### 🌊 The "Super-Flow" Pipeline
When you ask a question, the system triggers a sophisticated 5-stage retrieval process:
1.  **🔍 Query Rewriting:** The LLM rewrites your question to make it more specific and optimized for search.
2.  **🎭 HyDE (Hypothetical Document Embeddings):** The system generates a "fake" ideal answer first, then uses *that* to find real documents that look like it.
3.  **🤝 Hybrid Search:**
    *   **Vector Search:** Finds meaning and semantics (ChromaDB).
    *   **Keyword Search (BM25):** Finds exact words and technical terms (Cached in Redis for ⚡ speed).
4.  **🎯 Reranking:** All results are passed through a **Flashrank Reranker** to ensure the most relevant context is moved to the very top.
5.  **💬 Final Answer:** The LLM uses the "gold-standard" context to provide a hallucination-free response.
---
## ✨ Key Features
*   **🛡️ Intelligent Deduplication:** Custom-built hashing system. It ignores file names and paths, hashing only the **content**. Even if you upload the same file 100 times with different names, it only stores it **once**.
*   **⚡ BM25 Redis Cache:** Uses a distributed locking mechanism to ensure the keyword index is built efficiently across multiple workers.
*   **📊 RAGAS Evaluation:** Built-in endpoint to calculate **Faithfulness** and **Answer Relevancy**, so you can scientifically measure how good your AI is.
*   **🏢 Scalable MVC:** Clean separation of concerns. Easy to swap Chroma for Pinecone or Llama for GPT-4.
---
## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Framework** | FastAPI (Python) |
| **Vector DB** | ChromaDB |
| **LLM** | Groq (Llama 3.3 70B) |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Cache** | Redis |
| **Reranker** | Flashrank |
---
## 📁 Project Structure
*   **`src/routes/`** 🛣️: The API gateways for data upload and asking questions.
*   **`src/controllers/`** 🎮: The brain of the app that coordinates chunking and processing.
*   **`src/retriever/`** 🐕: The advanced retrieval logic (HyDE, Rewriting, Reranking).
*   **`src/vectordatabase/`** 🗄️: Direct management of ChromaDB and deduplication.
*   **`src/Prompts/`** 📄: Clean, plain-text templates for orchestrating the LLM.
---
## 🚀 Getting Started
1.  **Environment Setup:**
    Create a `.env` file with your `GROQ_API_KEY` and `REDIS_HOST`.
2.  **Upload Data:**
    Use `POST /api/v25/data/upload/{project_id}` to send your PDFs.
3.  **Process Chunks:**
    Use `POST /api/v25/data/process-assets` to trigger indexing.
4.  **Ask Questions:**
    Use `POST /api/v25/data/Ask_Q` and experience the power of the hybrid pipeline!
---