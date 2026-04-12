from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from helpers import config



def get_embedding():
    return HuggingFaceBgeEmbeddings(model_name=config.get_settings().EMBEDDING_MODEL_NAME)