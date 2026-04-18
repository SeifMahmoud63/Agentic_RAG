from pydantic_settings import BaseSettings,SettingsConfigDict
import os
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
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "rag_documents"
    QDRANT_DENSE_VECTOR_NAME: str = "dense"
    QDRANT_SPARSE_VECTOR_NAME: str = "sparse"
    QDRANT_DENSE_DIM: int = 384
    QDRANT_UPSERT_BATCH_SIZE: int = 4
    QDRANT_POINTS_UPSERT_BATCH_SIZE: int = 64
    QDRANT_SCROLL_LIMIT_OVERLAP: int = 1000
    QDRANT_SCROLL_LIMIT_SYNC: int = 500
    METADATA_DB_NAME: str = "file_metadata.db"
    
    EVAL_MAX_QUESTIONS: int = 3
    EVAL_RETRIEVAL_K: int = 3
    EVAL_MAX_CONTEXT_CHARS: int = 300
    EVAL_MAX_RESPONSE_CHARS: int = 600
    EVAL_SLEEP_BETWEEN_SAMPLES: int = 2
    EVAL_LLM_MODEL: str = "llama-3.1-8b-instant"
    EVAL_LLM_TEMPERATURE: float = 0.0
    EVAL_LLM_MAX_TOKENS: int = 1024
    EVAL_RELEVANCY_STRICTNESS: int = 1
    EVAL_MAX_WORKERS: int = 1
    EVAL_TIMEOUT: int = 300
    EVAL_MAX_RETRIES: int = 20
    EVAL_MAX_WAIT: int = 5

    TG_DEFAULT_PROJECT_ID: str = "telegram_uploads"
    TG_REQUEST_TIMEOUT: float = 120.0
    TG_WHISPER_MODEL: str = "whisper-large-v3"
    TG_WHISPER_LANGUAGE: str = "en"
    TG_CHUNK_SIZE: int = 500
    TG_CHUNK_OVERLAP: int = 50

    GROQ_API_KEY:str

    DENSE_SEARCH_WEIGHT: float
    SPARSE_SEARCH_WEIGHT: float 

    MAX_RETRIES: int = 5
    DUPLICATE_THRESHOLD: float = 0.35

    
    model_config=SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
         extra="ignore"
    )

@lru_cache(maxsize=1)
def get_settings():
    return Settings()
