from typing import Any

import pytest

from rag_platform.services import query_embeddings
from rag_platform.services.query_embeddings import QueryEmbeddingUnavailable


class FakeResult:
    def __init__(self, value: Any = None, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.forgotten = False

    def get(self, **kwargs: object) -> Any:
        if self.error:
            raise self.error
        return self.value

    def forget(self) -> None:
        self.forgotten = True


@pytest.mark.asyncio
async def test_embed_query_returns_numeric_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    result = FakeResult([1, 2.5])
    monkeypatch.setattr(query_embeddings.app, "send_task", lambda *args, **kwargs: result)

    assert await query_embeddings.embed_query("query") == [1.0, 2.5]
    assert result.forgotten is True


@pytest.mark.asyncio
async def test_embed_query_wraps_worker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    result = FakeResult(error=TimeoutError())
    monkeypatch.setattr(query_embeddings.app, "send_task", lambda *args, **kwargs: result)

    with pytest.raises(QueryEmbeddingUnavailable):
        await query_embeddings.embed_query("query")
    assert result.forgotten is True


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, [], "not-a-vector"])
async def test_embed_query_rejects_invalid_vectors(
    monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    result = FakeResult(value)
    monkeypatch.setattr(query_embeddings.app, "send_task", lambda *args, **kwargs: result)

    with pytest.raises(QueryEmbeddingUnavailable):
        await query_embeddings.embed_query("query")
