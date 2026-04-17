import time
from langchain_cohere import ChatCohere
from helpers.config import get_settings
from functools import lru_cache

class ThrottledChatCohere(ChatCohere):
    """
    Subclass of ChatCohere that adds a mandatory delay between requests.
    This is essential for Trial Keys which have strict 20 RPM limits.
    """
    def _generate(self, *args, **kwargs):
        # Mandatory 7-second delay between every single call
        # This guarantees max 8 calls per minute (60/7 = 8.5), extremely safe for Trial limits.
        time.sleep(7.0) 
        return super()._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        # Mandatory delay for async calls as well
        import asyncio
        await asyncio.sleep(7.0)
        return await super()._agenerate(*args, **kwargs)


@lru_cache(maxsize=1)
def get_llm():
    settings = get_settings()
    return ThrottledChatCohere(
        model=settings.MODEL_NAME,
        cohere_api_key=settings.COHERE_API_KEY,
        max_tokens=2048,
        temperature=0
    )