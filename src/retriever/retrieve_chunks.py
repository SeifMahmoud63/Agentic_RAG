from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from helpers.config import get_settings
from helpers.redis import (
    get_cached_docs,
    cache_docs,
    acquire_lock,
    release_lock
)
from Prompts import HYDE_PROMPT, REWRITE_PROMPT

settings = get_settings()


class RetrievalManager:
    _bm25_retriever = None

    @classmethod
    def get_bm25(cls, vector_store, docs=None):

        if cls._bm25_retriever is not None:
            return cls._bm25_retriever

        print("--- Loading BM25 using Redis cache ---")

        stored_data = get_cached_docs()

        if stored_data is None:

            if acquire_lock():

                print("Building BM25 docs and caching...")

                stored_data = vector_store.get()
                cache_docs(stored_data)

                release_lock()

            else:
                import time
                while stored_data is None:
                    time.sleep(1)
                    stored_data = get_cached_docs()

        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(
                stored_data["documents"],
                stored_data["metadatas"]
            )
        ]

        cls._bm25_retriever = BM25Retriever.from_documents(docs)
        cls._bm25_retriever.k = settings.TOP_K_BM25

        return cls._bm25_retriever


def get_llm():
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.MODEL_NAME,
    )


llm = get_llm()


def rewrite_query(query: str):
    prompt = REWRITE_PROMPT.format(query=query)
    return llm.invoke(prompt).content


def generate_hyde(query: str):
    prompt = HYDE_PROMPT.format(query=query)
    return llm.invoke(prompt).content


def hybrid_search(vector_store, query, docs=None):

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.TOP_K_HYBRID},
    )

    vector_docs = vector_retriever.invoke(query)

    bm25_retriever = RetrievalManager.get_bm25(vector_store, docs)
    bm25_docs = bm25_retriever.invoke(query)

    combined = {doc.page_content: doc for doc in vector_docs + bm25_docs}
    return list(combined.values())


def rerank(query, documents, top_k=settings.TOP_K_RERANKER):
    compressor = FlashrankRerank(top_n=top_k)
    return compressor.compress_documents(
        documents=documents,
        query=query,
    )


def advanced_retrieve(vector_store, query, docs=None):

    rewritten_query = rewrite_query(query)

    hyde_doc = generate_hyde(rewritten_query)

    candidates = hybrid_search(vector_store, hyde_doc, docs)

    final_docs = rerank(rewritten_query, candidates)

    return final_docs