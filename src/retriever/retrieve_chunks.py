# from langchain_groq import ChatGroq
# from langchain_community.retrievers import BM25Retriever
# from langchain_community.document_compressors import FlashrankRerank
# from langchain_core.documents import Document
# from helpers.config import get_settings
# from helpers.redis import (
#     get_cached_docs,
#     cache_docs,
#     acquire_lock,
#     release_lock
# )
# from llm.llm import get_llm
# from dotenv import load_dotenv

# from Prompts import HYDE_PROMPT,qa_prompt,REWRITE_PROMPT
# import logging
# logger = logging.getLogger('uvicorn.error')
# settings = get_settings()


# class RetrievalManager:
#     _bm25_retriever = None
#     _last_synced_version = None

#     @classmethod
#     def invalidate(cls):
#         """Resets the in-memory retriever singleton."""
#         cls._bm25_retriever = None
#         cls._last_synced_version = None
#         logger.info("--- [SYNC] In-memory BM25 invalidated ---")

#     @classmethod
#     def get_bm25(cls, vector_store, docs=None):
#         from helpers.redis import redis_client
#         from helpers.cache import BM25_VERSION_KEY

#         try:
#             redis_version = redis_client.get(BM25_VERSION_KEY)
#             if redis_version:
#                 if isinstance(redis_version, bytes):
#                     redis_version = redis_version.decode('utf-8')
#                 else:
#                     redis_version = str(redis_version)
#         except Exception as e:
#             logger.error(f"--- [SYNC] Redis version fetch failed: {e} ---")
#             redis_version = "error_fallback_" + str(time.time())


#         needs_refresh = (
#             cls._bm25_retriever is None or 
#             redis_version != cls._last_synced_version
#         )

#         if needs_refresh:
#             logger.info(f"--- [SYNC] BM25 Refresh Triggered: Local={cls._last_synced_version}, Redis={redis_version} ---")
            
#             stored_data = get_cached_docs()
            
#             # If no cached data in Redis, rebuild from vector store
#             if stored_data is None:
#                 if acquire_lock():
#                     logger.info("--- [SYNC] Rebuilding BM25 docs from Vector Store ---")
#                     try:
#                         stored_data = vector_store.get()
#                         if not stored_data or not stored_data.get("documents"):
#                             logger.warning("--- [SYNC] Vector store is empty. BM25 will be empty. ---")
#                             stored_data = {"documents": [], "metadatas": []}
#                         cache_docs(stored_data)
#                     finally:
#                         release_lock()
#                 else:
#                     # Wait for another worker to finish building
#                     logger.info("--- [SYNC] Waiting for another worker to rebuild cache ---")
#                     import time
#                     for _ in range(10): # Max 10 seconds wait
#                         time.sleep(1)
#                         stored_data = get_cached_docs()
#                         if stored_data: break
            
#             if not stored_data:
#                 stored_data = {"documents": [], "metadatas": []}

#             docs = [
#                 Document(page_content=text, metadata=meta)
#                 for text, meta in zip(
#                     stored_data.get("documents", []),
#                     stored_data.get("metadatas", [])
#                 )
#             ]

#             if docs:
#                 cls._bm25_retriever = BM25Retriever.from_documents(docs)
#                 cls._bm25_retriever.k = settings.TOP_K_BM25
#             else:
#                 # Placeholder for empty case
#                 cls._bm25_retriever = BM25Retriever.from_documents([Document(page_content="empty", metadata={})])
            
#             cls._last_synced_version = redis_version
#             logger.info("--- [SYNC] BM25 Refresh Complete ---")

#         return cls._bm25_retriever


# load_dotenv()
# settings = get_settings()

# llm = get_llm()

# def rewrite_query(query: str):
#     prompt = REWRITE_PROMPT.format(query=query)
#     return llm.invoke(prompt).content


# def generate_hyde(query: str):
#     prompt = HYDE_PROMPT.format(query=query)
#     return llm.invoke(prompt).content


# def hybrid_search(vector_store, query, docs=None):

#     vector_retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": settings.TOP_K_HYBRID},
#     )

#     vector_docs = vector_retriever.invoke(query)

#     bm25_retriever = RetrievalManager.get_bm25(vector_store, docs)
#     bm25_docs = bm25_retriever.invoke(query)

#     combined = {doc.page_content: doc for doc in vector_docs + bm25_docs}
#     return list(combined.values())


# def rerank(query, documents, top_k=settings.TOP_K_RERANKER):
#     compressor = FlashrankRerank(top_n=top_k)
#     return compressor.compress_documents(
#         documents=documents,
#         query=query,
#     )


# def advanced_retrieve(vector_store, query, docs=None):

#     rewritten_query = rewrite_query(query)

#     hyde_doc = generate_hyde(rewritten_query)

#     candidates = hybrid_search(vector_store, hyde_doc, docs)

#     final_docs = rerank(rewritten_query, candidates)

#     return final_docs

import logging
from langchain_community.retrievers import BM25Retriever
# from langchain_community.schema import Document
from langchain_core.documents import Document

from helpers.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class RetrievalManager:
    # Class-level variables to persist in memory during the application lifecycle
    _bm25_retriever = None
    _last_synced_version = None

    @classmethod
    def force_reindex(cls, vector_store):
        """
        Function: Triggered once when a new file is uploaded.
        It reads all data from the Vector Store and builds the BM25 Index.
        """
        logger.info("--- [REINDEX] Building BM25 Index... ---")
        
        # Fetch all stored data from ChromaDB or the provided Vector Store
        stored_data = vector_store.get()
        
        if not stored_data or not stored_data.get("documents"):
            logger.warning("--- [REINDEX] No docs found to index! ---")
            return

        # Convert data into Document objects for BM25 compatibility
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(
                stored_data.get("documents", []),
                stored_data.get("metadatas", [])
            )
        ]

        # Initialize the Retriever and store it in memory (cls._bm25_retriever)
        cls._bm25_retriever = BM25Retriever.from_documents(docs)
        cls._bm25_retriever.k = settings.TOP_K_BM25
        
        logger.info("--- [REINDEX] BM25 is now ready in Memory! ---")

    @classmethod
    def get_bm25(cls, vector_store):
        """
        Function: Called during the Q&A process.
        Returns the existing BM25 retriever from memory if available.
        If not (e.g., system startup), it initializes it once.
        """
        if cls._bm25_retriever is not None:
            return cls._bm25_retriever

        # Cold start logic: initialization on the first request after server start
        logger.info("--- [SYNC] Cold start: Initializing BM25 ---")
        cls.force_reindex(vector_store)
        return cls._bm25_retriever

def hybrid_search(vector_store, query):
    # 1. Perform Semantic Search using Vector Store (MMR)
    vector_retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": settings.TOP_K_HYBRID}
    )
    vector_docs = vector_retriever.invoke(query)

    # 2. Perform Keyword Search using BM25 from memory
    bm25_retriever = RetrievalManager.get_bm25(vector_store)
    bm25_docs = bm25_retriever.invoke(query)

    # Combine results and remove duplicates based on page content
    combined = {doc.page_content: doc for doc in vector_docs + bm25_docs}
    return list(combined.values())

def advanced_retrieve(vector_store, query):
    # Removed 'docs' parameter to streamline the flow
    # The process continues with hybrid search (HyDE, Rerank, etc., can be added)
    candidates = hybrid_search(vector_store, query)
    
    # ... Implement reranking logic here
    
    return candidates