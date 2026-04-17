from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):

    FILE_ALLOWED_TYPES:str
    MODEL_NAME : str 
    LANGCHAIN_TRACING_V2: bool 
    LANGCHAIN_API_KEY :str
    LANGCHAIN_PROJECT :str
    LANGCHAIN_ENDPOINT : str 
    FILE_ALLOWED_TYPES :list
    FILE_MAX_SIZE : int
    APP_NAME:str
    APP_VERSION : str
    FILE_DEFAULT_CHUNK_SIZE : int
    EMBEDDING_MODEL_NAME :str 
    FLASH_MODEL_RERANKER: str
    fLASH_CACHE_DIR:str
    TOP_K_RERANKER : int 
    TOP_K_BM25 :int 
    TOP_K_HYBRID : int 
    chunk_overlap:int
    chunk_size:int
    TOP_K_TAVILY :str
    REDIS_HOST : str
    REDIS_PORT : int
    COHERE_API_KEY:str
    TAVILY_API_KEY:str
    SPLADEE_MODEL_NAME :str
    GOOGLE_API_KEY:str

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "rag_documents"
    GROQ_API_KEY:str

    DENSE_SEARCH_WEIGHT: float
    SPARSE_SEARCH_WEIGHT: float 

    MAX_RETRIES: int = 5

    
    model_config=SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
         extra="ignore"
    )

@lru_cache()
def get_settings():
    return Settings()
