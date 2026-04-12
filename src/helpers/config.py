from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):

    GOOGLE_API_KEY:str
    FILE_ALLOWED_TYPES:str
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY :str
    LANGCHAIN_PROJECT :str
    LANGCHAIN_ENDPOINT : str = "https://api.smith.langchain.com"
    FILE_ALLOWED_TYPES :list
    FILE_MAX_SIZE : int
    APP_NAME:str
    APP_VERSION : str
    FILE_DEFAULT_CHUNK_SIZE : int

    
    model_config=SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
         extra="ignore"
    )


def get_settings():
    return Settings()