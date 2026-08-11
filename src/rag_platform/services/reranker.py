import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Self

import httpx
from pydantic import SecretStr

from rag_platform.core.config import Settings, get_settings


class RerankerUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RerankerDocument:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RerankerResult:
    id: str
    score: float
    rank: int


@dataclass(frozen=True)
class RerankerResponse:
    results: list[RerankerResult]
    model: str
    model_revision: str
    device: str
    usage: dict[str, Any]


class RerankerClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            base_url=settings.reranker_base_url,
            api_key=settings.reranker_api_key,
            timeout_seconds=settings.reranker_timeout_seconds,
            max_retries=settings.reranker_max_retries,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def rerank(
        self,
        *,
        query: str,
        documents: list[RerankerDocument],
        top_n: int,
        request_id: uuid.UUID,
        correlation_id: str,
    ) -> RerankerResponse:
        if self._api_key is None:
            raise RerankerUnavailable("reranker API key is not configured")
        payload = {
            "request_id": str(request_id),
            "query": query,
            "documents": [
                {"id": item.id, "text": item.text, "metadata": item.metadata} for item in documents
            ],
            "top_n": top_n,
            "return_documents": False,
            "truncate": True,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "x-request-id": str(request_id),
            "x-correlation-id": correlation_id,
        }
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post("/v1/rerank", headers=headers, json=payload)
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    raise RerankerUnavailable("reranker transport is unavailable") from exc
            else:
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
                if attempt >= self._max_retries:
                    break
            await asyncio.sleep(0.05 * (2**attempt))
        if response is None:
            raise RerankerUnavailable("reranker did not return a response")
        try:
            response.raise_for_status()
            body = response.json()
            return _parse_response(body, documents, top_n, request_id)
        except (httpx.HTTPError, TypeError, ValueError, KeyError) as exc:
            raise RerankerUnavailable("reranker returned an invalid response") from exc


def _parse_response(
    body: object,
    documents: list[RerankerDocument],
    top_n: int,
    request_id: uuid.UUID,
) -> RerankerResponse:
    if not isinstance(body, dict):
        raise ValueError("reranker response must be an object")
    if body.get("request_id") != str(request_id):
        raise ValueError("reranker request ID does not match")
    raw_results = body.get("results")
    expected_count = min(top_n, len(documents))
    if not isinstance(raw_results, list) or len(raw_results) != expected_count:
        raise ValueError("reranker result count is invalid")
    allowed_ids = {item.id for item in documents}
    results: list[RerankerResult] = []
    seen: set[str] = set()
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise ValueError("reranker result is invalid")
        item_id = raw.get("id")
        score = raw.get("score")
        rank = raw.get("rank")
        if (
            not isinstance(item_id, str)
            or item_id not in allowed_ids
            or item_id in seen
            or not isinstance(score, int | float)
            or not 0 <= float(score) <= 1
            or not isinstance(rank, int)
            or rank < 1
        ):
            raise ValueError("reranker result fields are invalid")
        seen.add(item_id)
        results.append(RerankerResult(item_id, float(score), rank))
    if [item.rank for item in results] != list(range(1, expected_count + 1)):
        raise ValueError("reranker ranks are invalid")
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("reranker usage is invalid")
    return RerankerResponse(
        results=results,
        model=str(body.get("model", "")),
        model_revision=str(body.get("model_revision", "")),
        device=str(body.get("device", "")),
        usage=usage,
    )


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
        async with RerankerClient.from_settings(settings) as client:
            response = await client.rerank(
                query="RAG platform connection test",
                documents=[
                    RerankerDocument(
                        id="connection-test",
                        text="Inert connectivity test document.",
                        metadata={},
                    )
                ],
                top_n=1,
                request_id=uuid.uuid4(),
                correlation_id="rag-admin-connection-test",
            )
    except RerankerUnavailable as exc:
        return {
            "status": "unavailable",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__,
        }
    return {
        "status": "up",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "result_count": len(response.results),
        "model": response.model,
        "model_revision": response.model_revision,
    }
