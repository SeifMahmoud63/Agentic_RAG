from langchain_cohere import ChatCohere
from helpers.config import get_settings

settings = get_settings()

def get_llm():
  return ChatCohere(
    model=settings.MODEL_NAME,
    api_key=settings.COHERE_API_KEY,
    timeout=120
  )
   