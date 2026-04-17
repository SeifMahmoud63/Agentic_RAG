from langchain_cohere import ChatCohere
from helpers.config import get_settings

def get_llm():
    return ChatCohere(
        model=get_settings().MODEL_NAME,
        api_key=get_settings().COHERE_API_KEY
    )
