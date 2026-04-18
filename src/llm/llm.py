import time
from langchain_cohere import ChatCohere
from helpers.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from functools import lru_cache
from langchain_cohere import ChatCohere

class ThrottledChatCohere(ChatCohere):
    """
    Subclass of ChatCohere that adds a mandatory delay between requests.
    This is essential for Trial Keys which have strict 20 RPM limits.
    """
    def _generate(self, *args, **kwargs):
        # Mandatory 7-second delay between every single call
        time.sleep(7.0) 
        return super()._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        # Mandatory delay for async calls as well
        import asyncio
        await asyncio.sleep(7.0)
        return await super()._agenerate(*args, **kwargs)


@lru_cache(maxsize=1)
def get_llm():
    """
    Returns the main reasoning LLM (now using Groq as requested).
    """
    settings = get_settings()
    return ChatCohere(
        model=settings.MODEL_NAME,
        api_key=settings.COHERE_API_KEY,
        max_tokens=2048,
        temperature=0
    )


@lru_cache(maxsize=1)
def get_fast_llm():
    """
    Returns a fast model (Google Gemini) for utility tasks like Query Rewriting and HyDE.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0
    )