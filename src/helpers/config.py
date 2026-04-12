from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):

    GROQ_API_KEY:str
    FILE_ALLOWED_TYPES:str
    MODEL_NAME : str ="llama-3.3-70b-versatile"
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY :str
    LANGCHAIN_PROJECT :str
    LANGCHAIN_ENDPOINT : str = "https://api.smith.langchain.com"
    FILE_ALLOWED_TYPES :list
    FILE_MAX_SIZE : int
    APP_NAME:str
    APP_VERSION : str
    FILE_DEFAULT_CHUNK_SIZE : int
    EMBEDDING_MODEL_NAME :str ="sentence-transformers/all-MiniLM-L6-v2"
    persist_directory :str = "chroma_db"
    collection_name : str = "Files_1"
    TOP_K_RERANKER : int =3
    TOP_K_BM25 :int =3
    TOP_K_HYBRID : int =3

    
    model_config=SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
         extra="ignore"
    )


def get_settings():
    return Settings()