from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedding():
    """
    Returns the HuggingFace BGE embedding model.
    Model: sentence-transformers/all-MiniLM-L6-v2
    Dimension: 384
    """
    return HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
