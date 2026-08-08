import asyncio
from contextlib import suppress

from celery.result import AsyncResult

from rag_platform.core.config import get_settings
from rag_platform.worker.celery_app import app


class QueryEmbeddingUnavailable(RuntimeError):
    pass


async def embed_query(query: str) -> list[float]:
    result: AsyncResult = app.send_task(
        "rag_platform.worker.tasks.embed_query",
        args=[query],
        queue="search",
    )
    try:
        value = await asyncio.to_thread(
            result.get,
            timeout=get_settings().query_embedding_timeout_seconds,
            propagate=True,
        )
    except Exception as exc:
        with suppress(Exception):
            result.forget()
        raise QueryEmbeddingUnavailable("query embedding worker is unavailable") from exc
    with suppress(Exception):
        result.forget()
    if not isinstance(value, list) or not value:
        raise QueryEmbeddingUnavailable("query embedding worker returned an invalid vector")
    return [float(component) for component in value]
