import time
import asyncio 
from functools import lru_cache
from langchain_cohere import ChatCohere
from helpers.config import get_settings

class ThrottledChatCohere(ChatCohere):
    """
    Subclass of ChatCohere that adds a mandatory delay between requests.
    This is essential for Trial Keys which have strict 20 RPM limits.
    """
    def _generate(self, *args, **kwargs):
        time.sleep(7.0) 
        return super()._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        await asyncio.sleep(7.0) 
        return await super()._agenerate(*args, **kwargs)


@lru_cache(maxsize=1)
def get_llm():
    """
    Returns the main reasoning LLM using the Throttled version.
    """
    settings = get_settings()
    return ThrottledChatCohere(
        model=settings.MODEL_NAME,
        cohere_api_key=settings.COHERE_API_KEY,
        max_tokens=2048,
        temperature=0
    )