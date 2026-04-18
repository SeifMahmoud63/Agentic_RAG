from logs.logger import logger
from typing import List, Optional
import time

from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest # Added FlashRank for better precision
from langchain_core.prompts import PromptTemplate
from prompts import REWRITE_PROMPT, HYDE_PROMPT
from llm.llm import get_llm, get_fast_llm
from helpers import config
from VectorDatabase import QdrantDb


import asyncio
settings=config.get_settings()
ranker = Ranker(model_name=settings.FLASH_MODEL_RERANKER, cache_dir=settings.fLASH_CACHE_DIR)

async def advanced_retrieve(query: str, top_k: Optional[int] = None) -> List[Document]:
    """
    Retrieve documents using Qdrant hybrid search, then rerank them using FlashRank.
    This process ensures higher semantic relevance for the final context.
    """
    start_total = time.time()
    llm = get_llm()
    if top_k is None:
        top_k = settings.TOP_K_HYBRID

    # 1. & 2. Run Query Rewrite and HyDE Doc generation in parallel using the "Fast" LLM (Gemini)
    logger.info(f"Triggering Parallel Rewrite and HyDE for: '{query[:50]}...'")
    rewrite_template = PromptTemplate.from_template(REWRITE_PROMPT)
    hyde_template = PromptTemplate.from_template(HYDE_PROMPT)
    fast_llm = get_fast_llm()
    
    start_parallel = time.time()
    # We trigger both simultaneously using Gemini to avoid Cohere Trial rate limits.
    res_rewrite, res_hyde = await asyncio.gather(
        llm.ainvoke(rewrite_template.format(query=query)),
        llm.ainvoke(hyde_template.format(query=query))
    )
    
    rewritten_query = res_rewrite.content
    hyde_doc = res_hyde.content
    end_parallel = time.time()
    
    logger.info(f"--- [PARALLEL LLM CALLS] took {end_parallel - start_parallel:.4f}s ---")
    logger.info(f"Rewritten Query: {rewritten_query[:50]}...")
    logger.info(f"HyDE Doc length: {len(hyde_doc)} chars")

    # 3. Hybrid Search with specialized queries
    start_search = time.time()
    results = QdrantDb.hybrid_search(query=rewritten_query, dense_query=hyde_doc, top_k=top_k)
    end_search = time.time()
    logger.info(f"--- [HYBRID SEARCH] took {end_search - start_search:.4f}s ---")

    if not results:
        logger.info(f"No results found for query: '{query[:50]}...' after {time.time() - start_total:.4f}s")
        return []

    try:
        start_rerank = time.time()
        pass_passages = [
            {
                "id": i, 
                "text": doc.page_content, 
                "meta": doc.metadata
            }
            for i, doc in enumerate(results)
        ]

        rerank_request = RerankRequest(query=query, passages=pass_passages)
        rerank_results = ranker.rerank(rerank_request)

        final_docs = []
        for r in rerank_results[:top_k]:
            final_docs.append(
                Document(
                    page_content=r["text"],
                    metadata=r["meta"]
                )
            )
        end_rerank = time.time()

        logger.info(f"--- [RERANKING] took {end_rerank - start_rerank:.4f}s ---")
        logger.info(f"--- [TOTAL RETRIEVAL] took {time.time() - start_total:.4f}s ---")
        return final_docs

    except Exception as e:
        logger.error(f"Error during FlashRank reranking: {e}. Falling back to initial retrieval.")
        return results[:top_k]
