import time
import json
import numpy as np
import redis

class ManualSemanticCache:
    def __init__(self, redis_url: str, embedding_model, score_threshold: float = 0.9):
        """
        Custom Semantic Cache using Redis for storage and Numpy for similarity search.
        threshold: 0.0 to 1.0 (Cosine Similarity, 1.0 is identical).
        """
        self.r = redis.Redis.from_url(redis_url)
        self.embedding_model = embedding_model
        self.threshold = score_threshold
        self.VECTOR_PREFIX = "manual:vector:"
        self.RESULT_PREFIX = "manual:result:"

    def lookup(self, query: str):
        start_time = time.time()
        
        query_vector = np.array(self.embedding_model.embed_query(query))
        norm_q = np.linalg.norm(query_vector)
        
        if norm_q == 0:
            return None

        vector_keys = self.r.keys(f"{self.VECTOR_PREFIX}*")
        if not vector_keys:
            print(f"--- [MANUAL CACHE MISS] Cache is empty (0.0000s) ---")
            return None

        vector_values = self.r.mget(vector_keys)
        
        best_match_id = None
        max_similarity = -1.0
        
        for i, val in enumerate(vector_values):
            if not val:
                continue
            
            cached_vector = np.array(json.loads(val))
            norm_c = np.linalg.norm(cached_vector)
            
            if norm_c > 0:
                similarity = np.dot(query_vector, cached_vector) / (norm_q * norm_c)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_id = vector_keys[i].decode().replace(self.VECTOR_PREFIX, "")

        duration = time.time() - start_time
        
        if best_match_id and max_similarity >= self.threshold:
            result_data = self.r.get(f"{self.RESULT_PREFIX}{best_match_id}")
            if result_data:
                print(f"--- [MANUAL CACHE HIT] Similarity: {max_similarity:.4f} (Ready in {duration:.4f}s) ---")
                return json.loads(result_data)
        
        best_score_str = f"{max_similarity:.4f}" if max_similarity > -1 else "N/A"
        print(f"--- [MANUAL CACHE MISS] Best similarity: {best_score_str} (Threshold: {self.threshold}) | Lookup: {duration:.4f}s ---")
        return None

    def update(self, query: str, response_text: str):
        """Store the query embedding and the answer with a 1-hour TTL."""
        import hashlib
        query_id = hashlib.md5(query.encode()).hexdigest()
        
        vector = self.embedding_model.embed_query(query)
        
        self.r.setex(f"{self.VECTOR_PREFIX}{query_id}", 3600, json.dumps(vector))
        self.r.setex(f"{self.RESULT_PREFIX}{query_id}", 3600, json.dumps(response_text))
        
        
        print(f"--- [MANUAL CACHE STORED] ID: {query_id} (TTL: 1 Hour) ---")

    def clear(self):
        """Clear all semantic cache entries."""
        count = 0
        for key in self.r.scan_iter(f"{self.VECTOR_PREFIX}*"):
            self.r.delete(key)
            count += 1
        for key in self.r.scan_iter(f"{self.RESULT_PREFIX}*"):
            self.r.delete(key)
        print(f"--- [MANUAL CACHE CLEARED] Removed {count} entries ---")

