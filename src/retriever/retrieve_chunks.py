# from langchain_groq import ChatGroq
# from langchain_community.retrievers import BM25Retriever
# from langchain_community.document_compressors import FlashrankRerank
# from langchain_core.documents import Document
# from helpers.config import get_settings 
# from Prompts import HYDE_PROMPT, REWRITE_PROMPT
# import os

# settings = get_settings()

# # --- 1. مدير الاسترجاع (للكاشينج) ---
# class RetrievalManager:
#     _bm25_retriever = None

#     @classmethod
#     def get_bm25(cls, vector_store, docs=None):
#         # لو الفهرس مش موجود، ابنيه
#         if cls._bm25_retriever is None:
#             print("--- Initializing BM25 Index for the first time ---")
            
#             # لو الـ docs مش مبعوتة، اسحبها من الكرومبا
#             if docs is None:
#                 stored_data = vector_store.get()
#                 docs = [
#                     Document(page_content=text, metadata=meta) 
#                     for text, meta in zip(stored_data['documents'], stored_data['metadatas'])
#                 ]
            
#             cls._bm25_retriever = BM25Retriever.from_documents(docs)
#             cls._bm25_retriever.k = settings.TOP_K_BM25
            
#         return cls._bm25_retriever

# # --- 2. إعداد الـ LLM ---
# def get_llm():
#     return ChatGroq(
#         api_key=settings.GROQ_API_KEY, 
#         model_name=settings.MODEL_NAME, 
#     )

# llm = get_llm()

# # --- 3. دوال المعالجة ---
# def rewrite_query(query: str):
#     prompt = REWRITE_PROMPT.format(query=query)
#     return llm.invoke(prompt).content

# def generate_hyde(query: str):
#     prompt = HYDE_PROMPT.format(query=query)
#     return llm.invoke(prompt).content

# def hybrid_search(vector_store, query, docs=None):
#     # البحث بالمعنى (Vector)
#     vector_retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": settings.TOP_K_HYBRID}
#     )
#     vector_docs = vector_retriever.invoke(query)

#     # البحث بالكلمات (BM25) - بنطلبه من المدير عشان الكاش
#     bm25_retriever = RetrievalManager.get_bm25(vector_store, docs)
#     bm25_docs = bm25_retriever.invoke(query)

#     # دمج النتائج وحذف التكرار
#     combined = {doc.page_content: doc for doc in vector_docs + bm25_docs}
#     return list(combined.values())

# def rerank(query, documents, top_k=settings.TOP_K_RERANKER):
#     compressor = FlashrankRerank(top_n=top_k)
#     return compressor.compress_documents(
#         documents=documents,
#         query=query
#     )

# # --- 4. الدالة الأساسية ---
# def advanced_retrieve(vector_store, query, docs=None):
#     # تحسين السؤال
#     rewritten_query = rewrite_query(query)
    
#     # توليد الوثيقة الافتراضية (HyDE)
#     hyde_doc = generate_hyde(rewritten_query)

#     # البحث الهجين (بيستخدم الكاش تلقائياً للـ BM25)
#     candidates = hybrid_search(vector_store, hyde_doc, docs)
    
#     # إعادة الترتيب
#     final_docs = rerank(rewritten_query, candidates)
    
#     return final_docs
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


# --- 1. Retrieval Manager ---
class RetrievalManager:
    _bm25_retriever = None

    @classmethod
    def get_bm25(cls, vector_store, docs=None):

        if cls._bm25_retriever is not None:
            return cls._bm25_retriever

        print("--- Loading BM25 using Redis cache ---")

        # ✅ حاول تجيب docs من Redis
        stored_data = get_cached_docs()

        if stored_data is None:

            # --- Distributed Lock ---
            if acquire_lock():

                print("Building BM25 docs and caching...")

                stored_data = vector_store.get()
                cache_docs(stored_data)

                release_lock()

            else:
                # Worker تاني بيبني الكاش
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


# --- 2. LLM ---
def get_llm():
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.MODEL_NAME,
    )


llm = get_llm()


# --- 3. Processing ---
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


# --- 4. Main ---
def advanced_retrieve(vector_store, query, docs=None):

    rewritten_query = rewrite_query(query)

    hyde_doc = generate_hyde(rewritten_query)

    candidates = hybrid_search(vector_store, hyde_doc, docs)

    final_docs = rerank(rewritten_query, candidates)

    return final_docs