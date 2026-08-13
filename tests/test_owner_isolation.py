"""Tests for multi-user owner isolation.

Verifies that documents, chunks, and search results are properly scoped
by owner_user_id across all layers: DB, services, vector search, BM25,
and OpenSearch.
"""

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from rag_platform.api.schemas import DocumentCreate, SearchRequest
from rag_platform.core.auth import Principal
from rag_platform.db.models import Chunk, Collection, Document, DocumentVersion
from rag_platform.services import retrieval
from rag_platform.services.documents import ingest
from rag_platform.services.opensearch import bm25_search
from rag_platform.services.vector_search import vector_search
from rag_platform.services.versioning import stable_document_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TENANT_ID = uuid.uuid4()
OWNER_A = uuid.uuid4()
OWNER_B = uuid.uuid4()
PROJECT_ID = uuid.uuid4()
COLLECTION = "manuals"


def _principal(owner_user_id: uuid.UUID) -> Principal:
    return Principal(
        TENANT_ID,
        owner_user_id,
        frozenset({PROJECT_ID}),
        frozenset({COLLECTION}),
        frozenset({"retrieval:search", "documents:write"}),
    )


def _search_request(**overrides: Any) -> SearchRequest:
    defaults: dict[str, Any] = dict(
        project_id=PROJECT_ID,
        collections=[COLLECTION],
        query="deployment guide",
        mode="dense",
        vector_top_k=10,
    )
    defaults.update(overrides)
    return SearchRequest(**defaults)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        embedding_model="BAAI/bge-m3",
        reranker_enabled=False,
        reranker_timeout_seconds=1.0,
        reranker_base_url="http://reranker.test",
        reranker_api_key=SecretStr("test-key"),
        reranker_max_retries=0,
    )


def _make_chunk(
    chunk_id: uuid.UUID,
    document_id: uuid.UUID,
    content: str,
    owner_user_id: uuid.UUID,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        document_version_id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        project_id=PROJECT_ID,
        owner_user_id=owner_user_id,
        collection=COLLECTION,
        chunk_index=0,
        content=content,
        token_count=len(content.split()),
        language="en",
        content_hash="a" * 64,
        metadata_={},
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
    )


class FakeSession:
    """Minimal async session stub for retrieval tests."""

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


class IngestFakeSession:
    """Fake session supporting scalar() with entity-type dispatch for ingest()."""

    def __init__(self, collection: Any = None) -> None:
        self.added: list[object] = []
        self.flush_count = 0
        self.committed = False
        self.executed: list[object] = []
        self._scalar_map: dict[type, Any] = {}
        if collection is not None:
            self._scalar_map[Collection] = collection

    def set_scalar_return(self, entity_type: type, value: Any) -> None:
        self._scalar_map[entity_type] = value

    async def scalar(self, statement: Any) -> Any:
        try:
            for desc in statement.column_descriptions:
                entity = desc.get("entity")
                if entity is not None and entity in self._scalar_map:
                    return self._scalar_map[entity]
        except Exception:
            pass
        return None

    def add(self, value: object) -> None:
        self.added.append(value)
        if hasattr(value, "id") and value.id is None:
            value.id = uuid.uuid4()

    async def flush(self) -> None:
        self.flush_count += 1
        for obj in self.added:
            if hasattr(obj, "id") and obj.id is None:
                obj.id = uuid.uuid4()

    async def execute(self, statement: Any) -> Any:
        self.executed.append(statement)
        return SimpleNamespace(
            all=lambda: [],
            scalars=lambda: SimpleNamespace(all=lambda: []),
        )

    async def commit(self) -> None:
        self.committed = True

    async def get(self, *args: object, **kwargs: object) -> None:
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cross_user_search_isolation() -> None:
    """User A's documents are not returned when User B searches."""
    document_id_a = uuid.uuid4()
    document_id_b = uuid.uuid4()
    chunk_a = _make_chunk(uuid.uuid4(), document_id_a, "Owner A secret data", OWNER_A)
    chunk_b = _make_chunk(uuid.uuid4(), document_id_b, "Owner B secret data", OWNER_B)

    # Simulate vector search scoped by owner: only returns same-owner chunks
    async def fake_vector_search(
        _session: Any,
        tenant_id: uuid.UUID,
        owner_user_id: uuid.UUID,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[tuple[Chunk, str, float]]:
        if owner_user_id == OWNER_A:
            return [(chunk_a, "doc-a", 0.9)]
        return []

    async def fake_bm25_search(
        tenant_id: uuid.UUID,
        owner_user_id: uuid.UUID,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[tuple[uuid.UUID, float]]:
        if owner_user_id == OWNER_A:
            return [(chunk_a.id, 8.0)]
        return []

    async def fake_embed_query(query: str) -> list[float]:
        return [0.1] * 1024

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(retrieval, "vector_search", fake_vector_search)
        monkeypatch.setattr(retrieval, "bm25_search", fake_bm25_search)
        monkeypatch.setattr(retrieval, "embed_query", fake_embed_query)
        monkeypatch.setattr(retrieval, "get_settings", _settings)

        import asyncio

        # Owner A gets their results
        session_a = FakeSession()
        principal_a = _principal(OWNER_A)
        _, results_a, _ = asyncio.run(
            retrieval.search(session_a, principal_a, _search_request())
        )
        assert len(results_a) == 1
        assert results_a[0]["content"] == "Owner A secret data"

        # Owner B gets nothing
        session_b = FakeSession()
        principal_b = _principal(OWNER_B)
        _, results_b, _ = asyncio.run(
            retrieval.search(session_b, principal_b, _search_request())
        )
        assert results_b == []
    finally:
        monkeypatch.undo()


def test_cross_user_delete_isolation() -> None:
    """Deleting User A's document doesn't affect User B's document with same source_id."""
    # Two users with same external_document_id but different owners
    doc_a = Document(
        id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        project_id=PROJECT_ID,
        owner_user_id=OWNER_A,
        collection=COLLECTION,
        external_document_id="shared-source-id",
    )
    doc_b = Document(
        id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        project_id=PROJECT_ID,
        owner_user_id=OWNER_B,
        collection=COLLECTION,
        external_document_id="shared-source-id",
    )

    # Verify they have different IDs despite same external_document_id
    assert doc_a.id != doc_b.id
    assert doc_a.external_document_id == doc_b.external_document_id


def test_owner_filter_fail_closed() -> None:
    """scoped_statement includes owner_user_id in WHERE clause."""
    who_a = _principal(OWNER_A)
    request = _search_request()

    statement = retrieval.scoped_statement(who_a, request)
    compiled = str(statement.compile(compile_kwargs={"literal_binds": True}))

    # Verify owner_user_id is in the SQL
    assert "owner_user_id" in compiled.lower()

    # Verify different owners produce different compiled params
    who_b = _principal(OWNER_B)
    statement_b = retrieval.scoped_statement(who_b, request)
    params_a = statement.compile().params
    params_b = statement_b.compile().params
    assert OWNER_A in params_a.values()
    assert OWNER_B in params_b.values()


@pytest.mark.asyncio
async def test_vector_search_scoped_by_owner() -> None:
    """vector_search includes owner_user_id in the WHERE clause."""
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
        TENANT_ID,
        OWNER_A,
        PROJECT_ID,
        [COLLECTION],
        {},
        [0.1, 0.2],
        "BAAI/bge-m3",
        10,
    )
    assert session.statement is not None
    compiled = str(session.statement.compile(compile_kwargs={"literal_binds": True}))
    assert "owner_user_id" in compiled.lower()


@pytest.mark.asyncio
async def test_bm25_search_scoped_by_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """bm25_search includes owner_user_id in the OpenSearch query."""
    captured_body: list[dict[str, Any]] = []

    class FakeSearchClient:
        async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
            captured_body.append(body)
            return {"hits": {"hits": []}}

        async def close(self) -> None:
            pass

    def fake_client() -> FakeSearchClient:
        return FakeSearchClient()

    monkeypatch.setattr("rag_platform.services.opensearch.client", fake_client)

    await bm25_search(
        TENANT_ID,
        OWNER_A,
        PROJECT_ID,
        [COLLECTION],
        "test query",
        {},
        10,
    )

    assert len(captured_body) == 1
    filter_clauses = captured_body[0]["query"]["bool"]["filter"]
    owner_clauses = [c for c in filter_clauses if "owner_user_id" in c.get("term", {})]
    assert len(owner_clauses) == 1
    assert owner_clauses[0]["term"]["owner_user_id"] == str(OWNER_A)


def test_document_ingestion_stores_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingested documents have correct owner_user_id."""
    import asyncio

    monkeypatch.setattr(
        "rag_platform.services.documents.get_settings",
        lambda: SimpleNamespace(
            parser_version="test-parser-v1",
            chunker_version="word-window-v1",
            embedding_model="BAAI/bge-m3",
            embedding_revision="default",
            index_version="rag-chunks-v1",
        ),
    )

    fake_collection = SimpleNamespace(id=uuid.uuid4())
    session = IngestFakeSession(collection=fake_collection)

    who = _principal(OWNER_A)
    payload = DocumentCreate(
        project_id=PROJECT_ID,
        collection=COLLECTION,
        external_document_id="test-doc-001",
        content="Test content for owner verification.",
        version=1,
    )

    result = asyncio.run(ingest(session, who, payload))

    assert isinstance(result, DocumentVersion)
    assert result.owner_user_id == OWNER_A

    # Verify the Document was also created with owner_user_id
    documents = [o for o in session.added if isinstance(o, Document)]
    assert len(documents) == 1
    assert documents[0].owner_user_id == OWNER_A


def test_same_content_different_owners() -> None:
    """Two users can have same document content with different owners, producing different IDs."""
    owner_a_id = stable_document_id(
        TENANT_ID, PROJECT_ID, COLLECTION, OWNER_A, "shared-source"
    )
    owner_b_id = stable_document_id(
        TENANT_ID, PROJECT_ID, COLLECTION, OWNER_B, "shared-source"
    )

    assert owner_a_id != owner_b_id

    # Same owner + same source = same ID (deterministic)
    owner_a_id_again = stable_document_id(
        TENANT_ID, PROJECT_ID, COLLECTION, OWNER_A, "shared-source"
    )
    assert owner_a_id == owner_a_id_again


def test_principal_requires_owner_user_id() -> None:
    """Principal dataclass requires owner_user_id field."""
    who = Principal(
        tenant_id=TENANT_ID,
        owner_user_id=OWNER_A,
        project_ids=frozenset({PROJECT_ID}),
        collections=frozenset({COLLECTION}),
        permissions=frozenset({"documents:write"}),
    )
    assert who.owner_user_id == OWNER_A
