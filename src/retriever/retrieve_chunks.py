from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from helpers.config import get_settings 
import os
from Prompts import HYDE_PROMPT, REWRITE_PROMPT


settings = get_settings()

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

def hybrid_search(vector_store, docs, query):
    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.TOP_K_HYBRID}
    )
    vector_docs = vector_retriever.invoke(query)

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = settings.TOP_K_BM25
    bm25_docs = bm25_retriever.invoke(query)

    combined = {doc.page_content: doc for doc in vector_docs + bm25_docs}
    return list(combined.values())

def rerank(query, documents, top_k=settings.TOP_K_RERANKER):
    compressor = FlashrankRerank(top_n=top_k)
    return compressor.compress_documents(
        documents=documents,
        query=query
    )

def advanced_retrieve(vector_store, docs, query):
    rewritten_query = rewrite_query(query)
    
    hyde_doc = generate_hyde(rewritten_query)

    candidates = hybrid_search(vector_store, docs, hyde_doc)
    
    final_docs = rerank(rewritten_query, candidates)
    
    return final_docs