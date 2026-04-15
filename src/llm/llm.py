from langchain_cohere import ChatCohere
from helpers.config import get_settings
from langchain_community.cache import RedisSemanticCache
from embedding_model import emb_model

from langchain_core.globals import set_llm_cache
import langchain


settings = get_settings()

langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=emb_model,
    score_threshold=0.5
)


def get_llm():
    return ChatCohere(
        model=settings.MODEL_NAME,
        api_key=settings.COHERE_API_KEY
                )