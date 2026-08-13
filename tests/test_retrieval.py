import uuid
from types import SimpleNamespace
from typing import Any

import pytest
import respx
from httpx import Request, Response
from pydantic import SecretStr

from rag_platform.api.schemas import SearchRequest
from rag_platform.core.auth import Principal
from rag_platform.db.models import Chunk
from rag_platform.services import retrieval
from rag_platform.services.opensearch import OpenSearchUnavailable
from rag_platform.services.vector_search import vector_search


class FakeSession:
    def __init__(self, rows: list[tuple[Chunk, str]] | None = None) -> None:
        self.added: list[object] = []
        self.committed = False
        self.rows = rows or []

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.committed = True

    async def execute(self, statement: object) -> object:
        class Rows:
            def __init__(self, values: list[tuple[Chunk, str]]) -> None:
                self.values = values

            def all(self) -> list[tuple[Chunk, str]]:
                return self.values

        return Rows(self.rows)


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
        uuid.uuid4(),
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
        reranker_api_key=SecretStr("test-reranker-key"),
        reranker_max_retries=0,
    )


def test_scoped_statement_excludes_historical_document_versions() -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()

    statement = retrieval.scoped_statement(
        principal(tenant_id, project_id),
        request(project_id),
    )
    compiled = str(statement.compile(compile_kwargs={"literal_binds": True}))

    assert "JOIN document_versions" in compiled
    assert "document_versions.is_current IS true" in compiled


@pytest.mark.asyncio
async def test_vector_search_excludes_historical_document_versions() -> None:
    class EmptyRows:
        def all(self) -> list[object]:
            return []

    class CapturingSession:
        statement: object | None = None

        async def execute(self, statement: object) -> EmptyRows:
            self.statement = statement
            return EmptyRows()

    session = CapturingSession()
    await vector_search(
        session,  # type: ignore[arg-type]
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        ["manuals"],
        {},
        [0.1, 0.2],
        "BAAI/bge-m3",
        10,
    )
    assert session.statement is not None
    compiled = str(session.statement.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN document_versions" in compiled
    assert "document_versions.is_current IS true" in compiled


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


@pytest.mark.asyncio
async def test_lexical_mode_skips_query_embedding_and_dense_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    chunk_id = uuid.uuid4()
    lexical_chunk = chunk(chunk_id, document_id, "lexical")

    async def lexical(*args: object, **kwargs: object) -> list[tuple[uuid.UUID, float]]:
        return [(chunk_id, 5.0)]

    async def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("dense stage must not run in lexical mode")

    monkeypatch.setattr(retrieval, "bm25_search", lexical)
    monkeypatch.setattr(retrieval, "vector_search", unexpected)
    monkeypatch.setattr(retrieval, "embed_query", unexpected)
    monkeypatch.setattr(retrieval, "get_settings", settings)

    _, results, trace = await retrieval.search(
        FakeSession(rows=[(lexical_chunk, "doc-1")]),
        principal(tenant_id, project_id),
        request(project_id, mode="lexical"),
    )

    assert [item["chunk_id"] for item in results] == [str(chunk_id)]
    assert results[0]["retrieval_sources"] == ["lexical"]
    assert trace["effective_mode"] == "lexical"


@pytest.mark.asyncio
async def test_dense_mode_skips_opensearch(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    dense_chunk = chunk(uuid.uuid4(), document_id, "dense")

    async def vector(*args: object, **kwargs: object) -> list[tuple[Chunk, str, float]]:
        return [(dense_chunk, "doc-1", 0.8)]

    async def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("lexical stage must not run in dense mode")

    monkeypatch.setattr(retrieval, "vector_search", vector)
    monkeypatch.setattr(retrieval, "bm25_search", unexpected)
    monkeypatch.setattr(retrieval, "get_settings", settings)

    _, results, trace = await retrieval.search(
        FakeSession(),
        principal(tenant_id, project_id),
        request(project_id, mode="dense"),
        [0.1],
    )

    assert [item["chunk_id"] for item in results] == [str(dense_chunk.id)]
    assert results[0]["retrieval_sources"] == ["dense"]
    assert trace["effective_mode"] == "dense"


@pytest.mark.asyncio
async def test_lexical_mode_falls_back_to_dense_when_opensearch_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    dense_chunk = chunk(uuid.uuid4(), document_id, "fallback")

    async def lexical(*args: object, **kwargs: object) -> list[tuple[uuid.UUID, float]]:
        raise OpenSearchUnavailable

    async def vector(*args: object, **kwargs: object) -> list[tuple[Chunk, str, float]]:
        return [(dense_chunk, "doc-1", 0.7)]

    monkeypatch.setattr(retrieval, "bm25_search", lexical)
    monkeypatch.setattr(retrieval, "vector_search", vector)
    monkeypatch.setattr(retrieval, "get_settings", settings)

    _, results, trace = await retrieval.search(
        FakeSession(),
        principal(tenant_id, project_id),
        request(project_id, mode="lexical"),
        [0.1],
    )

    assert [item["chunk_id"] for item in results] == [str(dense_chunk.id)]
    assert trace["requested_mode"] == "lexical"
    assert trace["effective_mode"] == "dense_fallback"
    assert trace["opensearch_degraded"] is True


@pytest.mark.asyncio
@respx.mock
async def test_search_uses_external_reranker_scores_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    first = chunk(uuid.uuid4(), document_id, "first")
    second = chunk(uuid.uuid4(), document_id, "second")

    async def vector(*args: object, **kwargs: object) -> list[tuple[Chunk, str, float]]:
        return [(first, "doc-1", 0.9), (second, "doc-1", 0.8)]

    async def lexical(*args: object, **kwargs: object) -> list[tuple[uuid.UUID, float]]:
        return []

    def rerank_response(request_value: Request) -> Response:
        import json

        payload = json.loads(request_value.content)
        return Response(
            200,
            json={
                "request_id": payload["request_id"],
                "model": "BAAI/bge-reranker-v2-m3",
                "model_revision": "953dc6f",
                "device": "cpu",
                "results": [
                    {"id": str(second.id), "score": 0.95, "rank": 1},
                    {"id": str(first.id), "score": 0.75, "rank": 2},
                ],
                "usage": {"documents_received": 2, "documents_scored": 2},
            },
        )

    monkeypatch.setattr(retrieval, "vector_search", vector)
    monkeypatch.setattr(retrieval, "bm25_search", lexical)
    monkeypatch.setattr(retrieval, "get_settings", lambda: settings(reranker_enabled=True))
    route = respx.post("http://reranker.test/v1/rerank").mock(side_effect=rerank_response)

    _, results, trace = await retrieval.search(
        FakeSession(),
        principal(tenant_id, project_id),
        request(project_id, mode="dense", rerank_top_k=2),
        [0.1],
    )

    assert [item["chunk_id"] for item in results] == [str(second.id), str(first.id)]
    assert results[0]["reranker_score"] == 0.95
    assert results[0]["final_score"] == 0.95
    assert trace["reranker"]["model_revision"] == "953dc6f"
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-reranker-key"
    assert route.calls.last.request.headers["x-correlation-id"]
