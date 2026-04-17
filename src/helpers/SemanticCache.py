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
        # Prefixes for keys
        self.VECTOR_PREFIX = "manual:vector:"
        self.RESULT_PREFIX = "manual:result:"

    def lookup(self, query: str):
        start_time = time.time()
        
        query_vector = np.array(self.embedding_model.embed_query(query))
        
        best_match_id = None
        max_similarity = -1.0
        

        for key in self.r.scan_iter(f"{self.VECTOR_PREFIX}*"):
            vector_data = self.r.get(key)
            if not vector_data:
                continue
            
            cached_vector = np.array(json.loads(vector_data))
            

            norm_q = np.linalg.norm(query_vector)
            norm_c = np.linalg.norm(cached_vector)
            
            if norm_q > 0 and norm_c > 0:
                similarity = np.dot(query_vector, cached_vector) / (norm_q * norm_c)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_id = key.decode().replace(self.VECTOR_PREFIX, "")

        duration = time.time() - start_time
        
        if best_match_id and max_similarity >= self.threshold:
            result_data = self.r.get(f"{self.RESULT_PREFIX}{best_match_id}")
            if result_data:
                print(f"--- [MANUAL CACHE HIT] Similarity: {max_similarity:.4f} (Ready in {duration:.4f}s) ---")
                return json.loads(result_data)
        
        print(f"--- [MANUAL CACHE MISS] (Lookup took {duration:.4f}s) ---")
        return None

    def update(self, query: str, response_text: str):
        """Store the query embedding and the answer with a 1-hour TTL."""
        # Use a hash of the query for the key ID
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
