from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from functools import lru_cache
from helpers import config

@lru_cache(maxsize=1)

def get_embedding():

    return HuggingFaceBgeEmbeddings(

        model_name=config.get_settings().EMBEDDING_MODEL_NAME

    )