import hashlib
import json
from collections.abc import AsyncIterator

from redis.asyncio import Redis
from redis.exceptions import RedisError

from rag_platform.core.config import Settings, get_settings
from rag_platform.core.metrics import CACHE_HITS, CACHE_MISSES

QUERY_EMBEDDING_CACHE = "query_embedding"


class CacheUnavailable(RuntimeError):
    pass


def query_embedding_cache_key(query: str, settings: Settings) -> str:
    identity = {
        "backend": settings.embedding_backend,
        "dimension": settings.embedding_dimension,
        "model": settings.embedding_model,
        "normalization": settings.embedding_normalization,
        "query": query.strip(),
        "revision": settings.embedding_revision,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{settings.cache_namespace}:query-embedding:{digest}"


async def get_query_embedding(query: str) -> list[float] | None:
    settings = get_settings()
    if not settings.query_embedding_cache_enabled:
        return None
    cache = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        value = await cache.get(query_embedding_cache_key(query, settings))
        if value is None:
            CACHE_MISSES.labels(QUERY_EMBEDDING_CACHE).inc()
            return None
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not parsed:
            await cache.delete(query_embedding_cache_key(query, settings))
            CACHE_MISSES.labels(QUERY_EMBEDDING_CACHE).inc()
            return None
        CACHE_HITS.labels(QUERY_EMBEDDING_CACHE).inc()
        return [float(component) for component in parsed]
    except (RedisError, ValueError, TypeError):
        CACHE_MISSES.labels(QUERY_EMBEDDING_CACHE).inc()
        return None
    finally:
        await cache.aclose()


async def set_query_embedding(query: str, vector: list[float]) -> None:
    settings = get_settings()
    if not settings.query_embedding_cache_enabled:
        return
    cache = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await cache.set(
            query_embedding_cache_key(query, settings),
            json.dumps(vector, separators=(",", ":")),
            ex=settings.query_embedding_cache_ttl_seconds,
        )
    except RedisError:
        return
    finally:
        await cache.aclose()


async def clear_rag_cache() -> int:
    settings = get_settings()
    cache = Redis.from_url(settings.redis_url, decode_responses=True)
    deleted = 0
    try:
        keys: AsyncIterator[str] = cache.scan_iter(
            match=f"{settings.cache_namespace}:*",
            count=500,
        )
        async for key in keys:
            deleted += int(await cache.unlink(key))
    except RedisError as exc:
        raise CacheUnavailable("RAG cache is unavailable") from exc
    finally:
        await cache.aclose()
    return deleted
