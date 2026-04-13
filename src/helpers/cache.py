from helpers.redis import redis_client

BM25_DOCS_KEY = "bm25:documents"

def invalidate_bm25_cache():
    redis_client.delete(BM25_DOCS_KEY)