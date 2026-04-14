from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from helpers import config

def get_embedding():
    return GoogleGenerativeAIEmbeddings(
    model=config.get_settings().EMBEDDING_MODEL_NAME, 
    google_api_key=config.get_settings().GOOGLE_API_KEY
)

