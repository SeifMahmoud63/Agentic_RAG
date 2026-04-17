import logging
from typing import List, Optional

from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest # Added FlashRank for better precision
from helpers.config import get_settings
from VectorDatabase import QdrantDb

logger = logging.getLogger("uvicorn.error")
settings = get_settings()


ranker = Ranker(model_name=settings.FLASH_MODEL_RERANKER, cache_dir=settings.fLASH_CACHE_DIR)

def advanced_retrieve(query: str, top_k: Optional[int] = None) -> List[Document]:
    """
    Retrieve documents using Qdrant hybrid search, then rerank them using FlashRank.
    This process ensures higher semantic relevance for the final context.
    """
    if top_k is None:
        top_k = settings.TOP_K_HYBRID

  
    fetch_k = top_k * 3 
    
    results = QdrantDb.hybrid_search(query=query, top_k=fetch_k)

    if not results:
        logger.info(f"No results found for query: '{query[:50]}...'")
        return []

    try:
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

        logger.info(f"Successfully reranked {len(results)} docs down to {len(final_docs)} for query: '{query[:50]}...'")
        return final_docs

    except Exception as e:
        logger.error(f"Error during FlashRank reranking: {e}. Falling back to initial retrieval.")
        return results[:top_k]
