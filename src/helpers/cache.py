from helpers.redis import redis_client
import time

BM25_DOCS_KEY = "bm25:documents"
BM25_VERSION_KEY = "bm25:version"

def invalidate_bm25_cache():
    """
    Invalidates the Redis cache and updates the version key
    to notify all workers to refresh their in-memory BM25 retrievers.
    """
    # 1. Delete the actual cached documents
    redis_client.delete(BM25_DOCS_KEY)
    
    new_version = str(time.time())
    redis_client.set(BM25_VERSION_KEY, new_version)
    
    print(f"--- Cache invalidated. New version: {new_version} ---")