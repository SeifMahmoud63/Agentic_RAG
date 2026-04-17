from langchain_cohere import ChatCohere
from helpers.config import get_settings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    settings = get_settings()
    return ChatCohere(
        model=settings.MODEL_NAME,
        api_key=settings.COHERE_API_KEY
    )