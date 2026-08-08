import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from opensearchpy import AsyncOpenSearch
from redis.asyncio import Redis
from sqlalchemy import text

from rag_platform.core.config import get_settings
from rag_platform.db.session import Session
from rag_platform.services.blobs import client as minio_client
from rag_platform.services.opensearch import client as opensearch_client
from rag_platform.services.readiness import readiness_status

Probe = Callable[[], Awaitable[dict[str, Any]]]
STARTED_AT = time.monotonic()


async def system_health() -> dict[str, Any]:
    probes: list[tuple[str, Probe]] = [
        ("postgresql", _postgresql),
        ("redis", _redis),
        ("opensearch", _opensearch),
        ("minio", _minio),
        ("reranker-service", _reranker),
        ("indexing-worker", _worker),
    ]
    results = await asyncio.gather(*(_timed_probe(name, probe) for name, probe in probes))
    components = [
        {
            "name": "rag-api",
            "status": "up",
            "uptime_seconds": round(time.monotonic() - STARTED_AT, 1),
        },
        *results,
    ]
    has_down = any(component["status"] == "down" for component in components)
    has_degraded = any(component["status"] in {"degraded", "not_ready"} for component in components)
    status = "down" if has_down else "degraded" if has_degraded else "operational"
    return {"status": status, "components": components}


async def _timed_probe(name: str, probe: Probe) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        details = await asyncio.wait_for(probe(), timeout=3.0)
    except Exception as exc:
        details = {"status": "down", "error": type(exc).__name__}
    return {
        "name": name,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **details,
    }


async def _postgresql() -> dict[str, Any]:
    async with Session() as session:
        version = await session.scalar(text("SHOW server_version"))
        vector_version = await session.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
    return {
        "status": "up" if vector_version else "degraded",
        "version": version,
        "pgvector_version": vector_version,
    }


async def _redis() -> dict[str, Any]:
    cache = Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        info = await cache.info(section="server")
    finally:
        await cache.aclose()
    return {"status": "up", "version": info.get("redis_version")}


async def _opensearch() -> dict[str, Any]:
    search_client: AsyncOpenSearch = opensearch_client()
    try:
        info = await search_client.info()
        health = await search_client.cluster.health()
    finally:
        await search_client.close()
    cluster_status = health.get("status")
    return {
        "status": "up" if cluster_status in {"green", "yellow"} else "degraded",
        "version": info.get("version", {}).get("number"),
        "cluster_status": cluster_status,
    }


async def _minio() -> dict[str, Any]:
    buckets = await asyncio.to_thread(minio_client().list_buckets)
    return {"status": "up", "bucket_count": len(buckets)}


async def _reranker() -> dict[str, Any]:
    settings = get_settings()
    if not settings.reranker_enabled:
        return {"status": "disabled"}
    async with httpx.AsyncClient(timeout=2.5) as client:
        response = await client.get(f"{settings.reranker_base_url}/health/ready")
        response.raise_for_status()
        body = response.json()
    return {
        "status": "up",
        "model": body.get("model"),
        "version": body.get("version"),
    }


async def _worker() -> dict[str, Any]:
    ready, components = await readiness_status()
    model = components.get("embedding_model", {})
    return {
        "status": "up" if ready else "not_ready",
        "model": model.get("model"),
        "dimension": model.get("dimension"),
        "device": model.get("device"),
    }
