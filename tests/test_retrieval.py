import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from rag_platform.api.schemas import SearchRequest
from rag_platform.core.auth import Principal
from rag_platform.db.models import Chunk
from rag_platform.services import retrieval
from rag_platform.services.opensearch import OpenSearchUnavailable


class FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.committed = True


def chunk(chunk_id: uuid.UUID, document_id: uuid.UUID, content: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        content=content,
        metadata_={"category": "guide"},
    )


def request(project_id: uuid.UUID, **updates: Any) -> SearchRequest:
    return SearchRequest(
        project_id=project_id,
        collections=["manuals"],
        query="deployment guide",
        **updates,
    )


def principal(tenant_id: uuid.UUID, project_id: uuid.UUID) -> Principal:
    return Principal(
        tenant_id,
        frozenset({project_id}),
        frozenset({"manuals"}),
        frozenset({"retrieval:search"}),
    )


def settings(*, reranker_enabled: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_model="BAAI/bge-m3",
        reranker_enabled=reranker_enabled,
        reranker_timeout_seconds=1.0,
        reranker_base_url="http://reranker.test",
    )


@pytest.mark.asyncio
async def test_search_fuses_vector_and_lexical_results(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    first_id, second_id = uuid.uuid4(), uuid.uuid4()
    first, second = chunk(first_id, document_id, "first"), chunk(second_id, document_id, "second")
    session = FakeSession()

    async def vector(*args: object, **kwargs: object) -> list[tuple[Chunk, str, float]]:
        return [(first, "doc-1", 0.9), (second, "doc-1", 0.8)]

    async def lexical(*args: object, **kwargs: object) -> list[tuple[uuid.UUID, float]]:
        return [(second_id, 7.0), (first_id, 6.0)]

    monkeypatch.setattr(retrieval, "vector_search", vector)
    monkeypatch.setattr(retrieval, "bm25_search", lexical)
    monkeypatch.setattr(retrieval, "get_settings", settings)

    _, results, trace = await retrieval.search(
        session, principal(tenant_id, project_id), request(project_id), [0.1]
    )

    assert [item["chunk_id"] for item in results] == [str(first_id), str(second_id)]
    assert results[0]["vector_score"] == 0.9
    assert results[0]["bm25_score"] == 6.0
    assert trace["opensearch_degraded"] is False
    assert session.committed is True
    assert len(session.added) == 1


@pytest.mark.asyncio
async def test_search_degrades_when_opensearch_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    session = FakeSession()

    async def vector(*args: object, **kwargs: object) -> list[tuple[Chunk, str, float]]:
        return []

    async def lexical(*args: object, **kwargs: object) -> list[tuple[uuid.UUID, float]]:
        raise OpenSearchUnavailable

    async def embed(query: str) -> list[float]:
        return [0.2]

    monkeypatch.setattr(retrieval, "vector_search", vector)
    monkeypatch.setattr(retrieval, "bm25_search", lexical)
    monkeypatch.setattr(retrieval, "embed_query", embed)
    monkeypatch.setattr(retrieval, "get_settings", settings)

    _, results, trace = await retrieval.search(
        session, principal(tenant_id, project_id), request(project_id)
    )

    assert results == []
    assert trace["opensearch_degraded"] is True
