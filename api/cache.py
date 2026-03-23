import redis
import numpy as np
import uuid
from dotenv import load_dotenv
import os
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

load_dotenv()


class RedisSemanticCache:
    """Semantic cache using Redis with HNSW vector index.

    Stores question-answer pairs with their embeddings and retrieves cached answers
    when a semantically similar question is asked (cosine similarity above threshold).
    """
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URL"))
        self.similarity_threshold = 0.80
        self.ttl = 60 * 60 * 24  # 24 hours
        self._ensure_index()

    def _ensure_index(self):
        """Create the RediSearch HNSW vector index if it doesn't already exist.

        The index watches all keys with prefix 'cache:' and indexes their 'embedding' field
        for cosine similarity search.
        """
        try:
            self.redis_client.ft("cache_index").info()  # Check if vector index exists - ft = full-text
        except:
            schema = (
                TextField("question"),
                TextField("answer"),
                VectorField("embedding", # embedding is the vector representation of the question
                    "HNSW", # Hierarchical Navigable Small World
                    {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"} # COSINE is the similarity metric
                ),
            )
            self.redis_client.ft("cache_index").create_index(
                schema,
                definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH) # when we do hset("cache:abc123",..) in the set method, Redis automatically adds it to this index. We don't have to manually add things to the index. HASH index type allows us to store a dict in a key.
            )

    def get(self, embedding: list[float]):
        """Search cache for a semantically similar question using KNN vector search.

        Args:
            embedding: list[float] - 1536D vector of the query.

        Returns:
            str (cached answer) if similarity >= threshold, None otherwise.
        """
        query = (
            Query("*=>[KNN 1 @embedding $query_vec AS score]") # *=> KNN 1 means from ALL entries, find the 1 nearest neighbor. @embedding means, search the "embedding" field. $query_vec is the placeholder vector variable. AS score means, the distance is stored as "score".
            .return_fields("question", "answer", "score") # return the question, answer, and score fields
            .sort_by("score") # sort by score (distance. lowest = closest)
            .dialect(2) # RediSearch query syntax version
        )

        results = self.redis_client.ft("cache_index").search(
            query,
            query_params={"query_vec": np.array(embedding).astype(np.float32).tobytes()} # actual query_vec is filled with the embedding value
        )

        if results.total > 0:
            distance = float(results.docs[0].score)
            similarity = 1 - distance
            print(f"[CACHE] Closest match: '{results.docs[0].question}' | distance: {distance} | similarity: {similarity} | threshold: {self.similarity_threshold}")
            if similarity >= self.similarity_threshold:
                print("[CACHE] HIT - returning cached answer")
                return results.docs[0].answer
            print("[CACHE] MISS - similarity below threshold")
        else:
            print("[CACHE] MISS - no entries in cache")
        return None

    def set(self, question: str, answer: str, embedding: list[float]):
        """Store a question-answer pair with its embedding in Redis cache.

        Args:
            question: str - the original question text.
            answer: str - the LLM-generated answer.
            embedding: list[float] - 1536D vector of the question.

        The entry auto-expires after self.ttl seconds (default 24 hours).
        """
        key = f"cache:{uuid.uuid4()}"
        self.redis_client.hset(key, mapping={
            "question": question,
            "answer": answer,
            "embedding": np.array(embedding).astype(np.float32).tobytes() # stored in bytes
        })
        self.redis_client.expire(key, self.ttl)

    def clear(self):
        """Delete all entries from the Redis cache."""
        self.redis_client.flushdb() # Delete all cache entries

