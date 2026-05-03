import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)

CACHE_TTL = 60 * 60 * 24 * 7  # 7 days
