import json
import uuid
from types import SimpleNamespace

import pytest
import respx
from httpx import Request, Response
from pydantic import SecretStr

from rag_platform.services import reranker


def settings(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        reranker_enabled=enabled,
        reranker_base_url="http://reranker.test",
        reranker_api_key=SecretStr("test-reranker-key"),
        reranker_timeout_seconds=1,
        reranker_max_retries=0,
    )


@pytest.mark.asyncio
@respx.mock
async def test_reranker_status_and_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker, "get_settings", settings)
    respx.get("http://reranker.test/health/ready").mock(
        return_value=Response(
            200,
            json={"model": "cross-encoder", "version": "1", "device": "cpu"},
        )
    )

    def response(request: Request) -> Response:
        body = json.loads(request.content)
        return Response(
            200,
            json={
                "request_id": body["request_id"],
                "model": "cross-encoder",
                "model_revision": "1",
                "device": "cpu",
                "results": [{"id": "connection-test", "score": 0.9, "rank": 1}],
                "usage": {"documents_received": 1, "documents_scored": 1},
            },
        )

    rerank_route = respx.post("http://reranker.test/v1/rerank").mock(side_effect=response)

    status = await reranker.reranker_status()
    connection = await reranker.test_reranker_connection()
    assert status["status"] == "up"
    assert status["model"] == "cross-encoder"
    assert connection["status"] == "up"
    assert connection["result_count"] == 1
    sent = json.loads(rerank_route.calls.last.request.content)
    assert sent["documents"] == [
        {
            "id": "connection-test",
            "text": "Inert connectivity test document.",
            "metadata": {},
        }
    ]
    assert sent["top_n"] == 1
    assert rerank_route.calls.last.request.headers["Authorization"] == "Bearer test-reranker-key"


@pytest.mark.asyncio
@respx.mock
async def test_reranker_reports_disabled_and_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reranker, "get_settings", lambda: settings(False))
    assert await reranker.reranker_status() == {"status": "disabled"}
    assert await reranker.test_reranker_connection() == {"status": "disabled"}

    monkeypatch.setattr(reranker, "get_settings", settings)
    respx.get("http://reranker.test/health/ready").mock(return_value=Response(503))
    respx.post("http://reranker.test/v1/rerank").mock(
        return_value=Response(200, json={"unexpected": []})
    )
    assert (await reranker.reranker_status())["status"] == "unavailable"
    assert (await reranker.test_reranker_connection())["status"] == "unavailable"


@pytest.mark.asyncio
@respx.mock
async def test_reranker_client_rejects_unknown_or_duplicate_ids() -> None:
    request_id = uuid.uuid4()
    respx.post("http://reranker.test/v1/rerank").mock(
        return_value=Response(
            200,
            json={
                "request_id": str(request_id),
                "model": "model",
                "model_revision": "revision",
                "device": "cpu",
                "results": [
                    {"id": "unknown", "score": 0.9, "rank": 1},
                ],
                "usage": {},
            },
        )
    )
    async with reranker.RerankerClient(
        base_url="http://reranker.test",
        api_key=SecretStr("test-reranker-key"),
        timeout_seconds=1,
        max_retries=0,
    ) as client:
        with pytest.raises(reranker.RerankerUnavailable):
            await client.rerank(
                query="query",
                documents=[reranker.RerankerDocument("known", "text", {})],
                top_n=1,
                request_id=request_id,
                correlation_id="correlation",
            )
