import redis
import json
from helpers.config import get_settings

settings = get_settings()

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    decode_responses=True
)

BM25_DOCS_KEY = "bm25:documents"


def get_cached_docs():
    data = redis_client.get(BM25_DOCS_KEY)

    if data:
        return json.loads(data)

    return None


def cache_docs(data):
    redis_client.set(
        BM25_DOCS_KEY,
        json.dumps(data)
    )


def acquire_lock(lock_name="bm25_lock", timeout=30):
    return redis_client.set(lock_name, "1", nx=True, ex=timeout)


def release_lock(lock_name="bm25_lock"):
    redis_client.delete(lock_name)