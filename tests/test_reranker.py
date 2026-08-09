from types import SimpleNamespace

import pytest
import respx
from httpx import Response

from rag_platform.services import reranker


def settings(enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        reranker_enabled=enabled,
        reranker_base_url="http://reranker.test",
        reranker_timeout_seconds=1,
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
    respx.post("http://reranker.test/v1/rerank").mock(
        return_value=Response(200, json={"results": [{"index": 0, "score": 0.9}]})
    )

    status = await reranker.reranker_status()
    connection = await reranker.test_reranker_connection()
    assert status["status"] == "up"
    assert status["model"] == "cross-encoder"
    assert connection["status"] == "up"
    assert connection["result_count"] == 1


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
