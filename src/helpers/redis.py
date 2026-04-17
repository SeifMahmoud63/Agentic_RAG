from helpers.config import get_settings
from EmbeddingModel import EmbModel
from .SemanticCache import ManualSemanticCache

settings = get_settings()

_global_cache = None

def init_cache():
    global _global_cache
    # Using our manual, ultra-stable semantic cache
    _global_cache = ManualSemanticCache(
        redis_url=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
        embedding_model=EmbModel.get_embedding(),
        score_threshold=0.92
    )
    return _global_cache

def get_cache():
    """Returns the initialized semantic cache or initializes it if necessary."""
    global _global_cache
    if _global_cache is None:
        return init_cache()
    return _global_cache
