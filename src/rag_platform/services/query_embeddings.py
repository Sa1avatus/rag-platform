import asyncio
import json
import logging
from contextlib import suppress

import redis as _sync_redis
from redis.asyncio import Redis

from celery.result import AsyncResult

from rag_platform.core.config import get_settings
from rag_platform.core.embedding_registry import get_active_model
from rag_platform.services.cache import get_query_embedding, set_query_embedding
from rag_platform.worker.celery_app import app

log = logging.getLogger(__name__)

# Transient errors that warrant a retry (corrupt Redis result under concurrency).
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (json.JSONDecodeError, ValueError, UnicodeDecodeError)


class QueryEmbeddingUnavailable(RuntimeError):
    pass


async def embed_query(query: str) -> list[float]:
    cfg = get_active_model()
    cached = await get_query_embedding(query, model_id=cfg.id)
    if cached is not None:
        return cached

    settings = get_settings()
    max_attempts = 3
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        result: AsyncResult = app.send_task(
            "rag_platform.worker.tasks.embed_query_task",
            args=[query],
            queue="search",
        )
        try:
            # The worker stores the embedding in a Redis key and returns
            # just the key string (tiny payload).  Read the actual vector
            # from Redis directly to bypass Celery's JSON result backend
            # which corrupts large payloads under concurrent load.
            redis_key: str = await asyncio.to_thread(
                result.get,
                timeout=settings.query_embedding_timeout_seconds,
                propagate=True,
            )
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            with suppress(Exception):
                result.forget()
            log.warning(
                "embed_query_retryable_error",
                attempt=attempt,
                max_attempts=max_attempts,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            if attempt < max_attempts:
                await asyncio.sleep(0.1 * attempt)
                continue
            raise QueryEmbeddingUnavailable(
                "query embedding worker returned corrupt result"
            ) from exc
        except Exception as exc:
            with suppress(Exception):
                result.forget()
            raise QueryEmbeddingUnavailable(
                "query embedding worker is unavailable"
            ) from exc
        else:
            with suppress(Exception):
                result.forget()
            # Read the actual vector from Redis (small, direct read).
            try:
                raw = await asyncio.to_thread(
                    _read_embed_result, redis_key, settings.redis_url
                )
            except Exception as exc:
                raise QueryEmbeddingUnavailable(
                    "query embedding result read failed"
                ) from exc
            if not isinstance(raw, list) or not raw:
                raise QueryEmbeddingUnavailable(
                    "query embedding worker returned an invalid vector"
                )
            vector = [float(component) for component in raw]
            await set_query_embedding(query, vector, model_id=cfg.id)
            return vector

    # Should not be reached, but satisfy type checker.
    raise QueryEmbeddingUnavailable("query embedding worker is unavailable") from last_exc


def _read_embed_result(key: str, redis_url: str) -> list:
    """Read embedding vector from Redis (sync, runs in thread)."""
    r = _sync_redis.Redis.from_url(redis_url, decode_responses=True)
    try:
        data = r.get(key)
        if data is None:
            raise ValueError(f"Embedding key {key} expired or missing")
        return json.loads(data)
    finally:
        r.delete(key)
        r.close()
