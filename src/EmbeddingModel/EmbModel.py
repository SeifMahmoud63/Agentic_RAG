from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from helpers import config


embedding_model = HuggingFaceBgeEmbeddings(model_name=config.get_settings().EMBEDDING_MODEL_NAME)

def get_embedding():
    return embedding_model