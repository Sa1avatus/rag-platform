import time
from typing import Any

import httpx

from rag_platform.core.config import get_settings


async def reranker_status() -> dict[str, Any]:
    settings = get_settings()
    if not settings.reranker_enabled:
        return {"status": "disabled"}
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=settings.reranker_timeout_seconds) as client:
            response = await client.get(f"{settings.reranker_base_url}/health/ready")
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "status": "unavailable",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": type(exc).__name__,
        }
    return {
        "status": "up",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "model": body.get("model"),
        "version": body.get("version"),
        "device": body.get("device"),
    }


async def test_reranker_connection() -> dict[str, Any]:
    settings = get_settings()
    if not settings.reranker_enabled:
        return {"status": "disabled"}
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=settings.reranker_timeout_seconds) as client:
            response = await client.post(
                f"{settings.reranker_base_url}/v1/rerank",
                json={
                    "query": "RAG platform connection test",
                    "documents": ["Inert connectivity test document."],
                    "top_k": 1,
                },
            )
            response.raise_for_status()
            results = response.json().get("results")
            if not isinstance(results, list):
                raise ValueError("invalid reranker response")
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "status": "unavailable",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": type(exc).__name__,
        }
    return {
        "status": "up",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "result_count": len(results),
    }
