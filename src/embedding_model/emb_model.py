# from langchain_huggingface import HuggingFaceEmbeddings
# from helpers import config
# from functools import lru_cache


# @lru_cache(maxsize=1)
# def get_embedding():
#     return HuggingFaceEmbeddings(model_name=config.get_settings().EMBEDDING_MODEL_NAME)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from helpers import config

def get_embedding():
    return GoogleGenerativeAIEmbeddings(
    model=config.get_settings().EMBEDDING_MODEL_NAME, 
    google_api_key=config.get_settings().GOOGLE_API_KEY
)

