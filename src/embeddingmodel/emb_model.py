from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from functools import lru_cache
from helpers import config

@lru_cache(maxsize=1)

def get_embedding():
    model_name = config.get_settings().EMBEDDING_MODEL_NAME
    print(f"--- [LOADING EMBEDDING MODEL] {model_name} ---")
    return HuggingFaceBgeEmbeddings(
        model_name=model_name
    )