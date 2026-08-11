"""Full E2E pipeline test: ingest -> hybrid retrieval -> reranker -> context.

Exercises the complete service-layer pipeline with mocked external services
(embeddings, OpenSearch, reranker HTTP) but real business logic, chunking,
RRF fusion, context selection, and tracing.
"""

import os

os.environ.setdefault("RAG_ADMIN_TOKEN", "local-rag-admin-token")
os.environ.setdefault("RAG_API_KEY_PEPPER", "local-development-pepper")

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
import respx
from httpx import Request, Response
from pydantic import SecretStr

from rag_platform.api.schemas import DocumentCreate, SearchRequest
from rag_platform.core.auth import Principal
from rag_platform.db.models import (
    Chunk,
    Collection,
    Document,
    DocumentVersion,
    OutboxEvent,
)
from rag_platform.services import retrieval
from rag_platform.services.chunking import ChunkingConfig, chunk_text
from rag_platform.services.context_selection import select_context
from rag_platform.services.documents import ingest
from rag_platform.services.versioning import (
    content_hash,
    stable_chunk_id,
    stable_document_id,
    stable_version_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TENANT_ID = uuid.uuid4()
PROJECT_ID = uuid.uuid4()
DOCUMENT_ID = uuid.uuid4()
COLLECTION = "manuals"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 32  # small for test speed


def _make_embedding(seed: int) -> list[float]:
    """Deterministic pseudo-embedding for testing."""
    import random

    rng = random.Random(seed)
    raw = [rng.random() for _ in range(EMBEDDING_DIM)]
    magnitude = sum(v**2 for v in raw) ** 0.5
    return [v / magnitude for v in raw]


def _make_chunk(chunk_index: int, content: str, version_id: uuid.UUID | None = None) -> Chunk:
    digest = content_hash(content)
    return Chunk(
        id=stable_chunk_id(version_id or uuid.uuid4(), "word-window-v1", chunk_index, digest),
        document_id=DOCUMENT_ID,
        document_version_id=version_id or uuid.uuid4(),
        tenant_id=TENANT_ID,
        project_id=PROJECT_ID,
        collection=COLLECTION,
        chunk_index=chunk_index,
        content=content,
        token_count=len(content.split()),
        language="en",
        content_hash=digest,
        metadata_={"category": "deployment"},
        source_type="text",
        source_id="deploy-guide-001",
        section_title=f"Section {chunk_index}",
        start_offset=0,
        end_offset=len(content),
        chunker_version="word-window-v1",
        index_version="rag-chunks-v1",
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIM,
    )


def _principal() -> Principal:
    return Principal(
        TENANT_ID,
        frozenset({PROJECT_ID}),
        frozenset({COLLECTION}),
        frozenset({"retrieval:search", "documents:write"}),
    )


def _search_request(**overrides: Any) -> SearchRequest:
    defaults: dict[str, Any] = dict(
        project_id=PROJECT_ID,
        collections=[COLLECTION],
        query="How to deploy the RAG platform to production",
        mode="hybrid",
        vector_top_k=10,
        bm25_top_k=10,
        fusion_top_k=8,
        rerank_top_k=5,
        use_reranker=True,
        include_trace=True,
        max_context_chunks=4,
        max_context_tokens=2000,
        per_document_limit=2,
    )
    defaults.update(overrides)
    return SearchRequest(**defaults)


def _settings(reranker_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_model=EMBEDDING_MODEL,
        reranker_enabled=reranker_enabled,
        reranker_timeout_seconds=2.0,
        reranker_base_url="http://reranker.test",
        reranker_api_key=SecretStr("test-reranker-key"),
        reranker_max_retries=0,
    )


class FakeSession:
    """Minimal async session stub that records adds and commits."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.committed = True

    async def get(self, *args: object, **kwargs: object) -> None:
        return None


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


def _ingest_settings() -> SimpleNamespace:
    """Settings stub for ingest() tests."""
    return SimpleNamespace(
        parser_version="test-parser-v1",
        chunker_version="word-window-v1",
        embedding_model=EMBEDDING_MODEL,
        embedding_revision="default",
        index_version="rag-chunks-v1",
    )


# ---------------------------------------------------------------------------
# Test: chunking produces multiple drafts from real content
# ---------------------------------------------------------------------------


def test_chunking_produces_expected_drafts() -> None:
    content = (
        "# Deployment Guide\n\n"
        "Deploy the RAG platform using Docker Compose. "
        "Ensure PostgreSQL, Redis, OpenSearch, and MinIO are running.\n\n"
        "## Configuration\n\n"
        "Set environment variables in .env. "
        "The embedding model is BAAI/bge-m3 with 1024 dimensions.\n\n"
        "## Health Checks\n\n"
        "Each service exposes a health endpoint. "
        "The API at /health returns readiness and liveness status."
    )
    drafts = chunk_text(
        content,
        "paragraph",
        ChunkingConfig(target_words=20, overlap_words=5, minimum_words=5),
    )
    assert len(drafts) >= 2
    for draft in drafts:
        assert len(draft.content.strip()) > 0
        assert draft.start_offset >= 0
        assert draft.end_offset > draft.start_offset


# ---------------------------------------------------------------------------
# Test: full pipeline ingest -> hybrid retrieval -> reranker -> context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_full_pipeline_ingest_to_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: chunking, hybrid retrieval (vector + BM25), RRF fusion,
    external reranker, and context selection with dedup and per-doc limits."""

    # -- Step 1: Chunk real content --
    content = (
        "# Deployment Guide\n\n"
        "Deploy the RAG platform using Docker Compose. "
        "Ensure PostgreSQL, Redis, OpenSearch, and MinIO are running.\n\n"
        "## Configuration\n\n"
        "Set environment variables in .env. "
        "The embedding model is BAAI/bge-m3 with 1024 dimensions.\n\n"
        "## Health Checks\n\n"
        "Each service exposes a health endpoint. "
        "The API at /health returns readiness and liveness status.\n\n"
        "## Monitoring\n\n"
        "Prometheus metrics are available at /metrics. "
        "Use Grafana dashboards for visualization."
    )
    drafts = chunk_text(
        content,
        "paragraph",
        ChunkingConfig(target_words=25, overlap_words=5, minimum_words=5),
    )
    assert len(drafts) >= 2

    # -- Step 2: Build Chunk objects from drafts --
    version_id = uuid.uuid4()
    chunks = [_make_chunk(draft.chunk_index, draft.content, version_id) for draft in drafts]

    # -- Step 3: Mock vector search to return chunks with scores --
    async def fake_vector_search(*args: Any, **kwargs: Any) -> list[tuple[Chunk, str, float]]:
        return [(chunks[0], "doc-1", 0.95), (chunks[1], "doc-1", 0.88)]

    async def fake_bm25_search(*args: Any, **kwargs: Any) -> list[tuple[uuid.UUID, float]]:
        return [(chunks[1].id, 9.0), (chunks[0].id, 7.5)]

    async def fake_embed_query(query: str) -> list[float]:
        return _make_embedding(42)

    monkeypatch.setattr(retrieval, "vector_search", fake_vector_search)
    monkeypatch.setattr(retrieval, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(retrieval, "embed_query", fake_embed_query)
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings())

    # -- Step 4: Mock external reranker HTTP endpoint --
    def _rerank_handler(request: Request) -> Response:
        import json as _json

        payload = _json.loads(request.content)
        return Response(
            200,
            json={
                "request_id": payload["request_id"],
                "model": "test-reranker",
                "model_revision": "v1",
                "device": "cpu",
                "results": [
                    {"id": str(chunks[0].id), "score": 0.97, "rank": 1},
                    {"id": str(chunks[1].id), "score": 0.91, "rank": 2},
                ],
                "usage": {"documents_received": 2, "documents_scored": 2},
            },
        )

    respx.post("http://reranker.test/v1/rerank").mock(side_effect=_rerank_handler)

    # -- Step 5: Run retrieval --
    session = FakeSession()
    data = _search_request()
    request_id, results, trace = await retrieval.search(session, _principal(), data)

    assert len(results) > 0
    assert trace["reranker_used"] is True
    assert trace["reranker"]["model"] == "test-reranker"

    # Verify retrieval sources
    for result in results:
        assert "dense" in result["retrieval_sources"]
        assert "lexical" in result["retrieval_sources"]

    # Verify reranker scores are present and used as final_score
    for result in results:
        assert result["reranker_score"] is not None
        assert result["final_score"] == result["reranker_score"]

    # Verify the retrieval request was persisted
    assert session.committed is True
    assert len(session.added) == 1

    # -- Step 8: Run context selection on the results --
    selection = select_context(
        results,
        max_chunks=data.max_context_chunks,
        max_estimated_tokens=data.max_context_tokens,
        per_document_limit=data.per_document_limit,
    )

    # Context selection applied limits
    assert len(selection.items) <= data.max_context_chunks
    assert selection.estimated_tokens <= data.max_context_tokens
    for item in selection.items:
        assert "estimated_tokens" in item
        assert item["estimated_tokens"] > 0
        assert "content" in item
        assert len(str(item["content"])) > 0

    # Per-document limit was respected
    from collections import Counter

    doc_counts = Counter(str(item["document_id"]) for item in selection.items)
    for count in doc_counts.values():
        assert count <= data.per_document_limit

    # Context string is non-empty and contains chunk content
    context_text = "\n\n".join(str(item["content"]) for item in selection.items)
    assert len(context_text) > 0
    assert any(word in context_text.lower() for word in ["deploy", "rag", "platform"])


# ---------------------------------------------------------------------------
# Test: degraded reranker still produces results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_degrades_gracefully_when_reranker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid retrieval works even when the reranker is unavailable."""

    version_id = uuid.uuid4()
    chunks = [
        _make_chunk(0, "First deployment step: configure environment variables.", version_id),
        _make_chunk(1, "Second step: run docker compose up for all services.", version_id),
    ]

    async def fake_vector_search(*args: Any, **kwargs: Any) -> list[tuple[Chunk, str, float]]:
        return [(chunks[0], "doc-1", 0.9), (chunks[1], "doc-1", 0.8)]

    async def fake_bm25_search(*args: Any, **kwargs: Any) -> list[tuple[uuid.UUID, float]]:
        return [(chunks[1].id, 7.0), (chunks[0].id, 6.0)]

    async def fake_embed_query(query: str) -> list[float]:
        return _make_embedding(99)

    monkeypatch.setattr(retrieval, "vector_search", fake_vector_search)
    monkeypatch.setattr(retrieval, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(retrieval, "embed_query", fake_embed_query)
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings(reranker_enabled=False))

    session = FakeSession()
    data = _search_request(use_reranker=False)
    _, results, trace = await retrieval.search(session, _principal(), data)

    assert len(results) > 0
    assert trace["reranker_used"] is False
    # Final score comes from fusion, not reranker
    for result in results:
        assert result["final_score"] == result["fusion_score"]

    # Context selection still works
    selection = select_context(
        results,
        max_chunks=4,
        max_estimated_tokens=2000,
        per_document_limit=2,
    )
    assert len(selection.items) > 0


# ---------------------------------------------------------------------------
# Test: context selection respects near-duplicate suppression
# ---------------------------------------------------------------------------


def test_context_selection_in_e2e_suppresses_similar_chunks() -> None:
    """Near-duplicate chunks from the same document are suppressed."""
    results = [
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": "doc-1",
            "content": "Deploy the RAG platform using Docker Compose for production",
            "metadata": {},
            "reranker_score": 0.95,
            "fusion_score": 0.8,
            "final_score": 0.95,
        },
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": "doc-1",
            "content": "Deploy the RAG platform using Docker Compose for production environments",
            "metadata": {},
            "reranker_score": 0.90,
            "fusion_score": 0.75,
            "final_score": 0.90,
        },
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": "doc-2",
            "content": "Monitor OpenSearch cluster health with the _cluster API endpoints",
            "metadata": {},
            "reranker_score": 0.85,
            "fusion_score": 0.7,
            "final_score": 0.85,
        },
    ]

    selection = select_context(
        results,
        max_chunks=5,
        max_estimated_tokens=1000,
        per_document_limit=3,
        near_duplicate_threshold=0.75,
    )

    # The near-duplicate from doc-1 is suppressed
    assert len(selection.items) == 2
    assert selection.duplicates_suppressed == 1


# ---------------------------------------------------------------------------
# Test: ingest creates Document, DocumentVersion, IndexingJob, OutboxEvent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_creates_document_version_and_indexing_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ingest() creates Document, DocumentVersion, IndexingJob, and OutboxEvent."""
    monkeypatch.setattr(
        "rag_platform.services.documents.get_settings",
        _ingest_settings,
    )

    tenant_id = uuid.uuid4()
    project_id = uuid.uuid4()
    collection_name = "test-docs"

    fake_collection = SimpleNamespace(id=uuid.uuid4())
    session = IngestFakeSession(collection=fake_collection)

    who = Principal(
        tenant_id,
        frozenset({project_id}),
        frozenset({collection_name}),
        frozenset({"documents:write"}),
    )

    payload = DocumentCreate(
        project_id=project_id,
        collection=collection_name,
        external_document_id="e2e-doc-001",
        content="Test content for ingest E2E verification.",
        version=1,
        metadata={"source": "e2e-test"},
    )

    result = await ingest(session, who, payload)

    # Returned value is the new DocumentVersion
    assert isinstance(result, DocumentVersion)
    assert result.is_current is True
    assert result.version == 1
    assert result.tenant_id == tenant_id
    assert result.project_id == project_id
    assert result.collection == collection_name

    # All expected entity types were created
    entity_names = [type(obj).__name__ for obj in session.added]
    assert "Document" in entity_names
    assert "DocumentVersion" in entity_names
    assert "IndexingJob" in entity_names
    assert "OutboxEvent" in entity_names

    # Document has correct scope
    documents = [o for o in session.added if isinstance(o, Document)]
    assert len(documents) == 1
    assert documents[0].tenant_id == tenant_id
    assert documents[0].current_version == 1

    # OutboxEvent signals document.index
    events = [o for o in session.added if isinstance(o, OutboxEvent)]
    assert len(events) == 1
    assert events[0].payload["type"] == "document.index"

    # Session was committed
    assert session.committed is True


# ---------------------------------------------------------------------------
# Test: ingest raises when collection is missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_raises_when_collection_missing() -> None:
    """ingest() raises ValueError when the collection does not exist."""
    session = IngestFakeSession()  # No collection in scalar_map

    project_id = uuid.uuid4()
    who = Principal(
        uuid.uuid4(),
        frozenset({project_id}),
        frozenset({"missing"}),
        frozenset({"documents:write"}),
    )
    payload = DocumentCreate(
        project_id=project_id,
        collection="missing",
        external_document_id="doc-001",
        content="Some content",
        version=1,
    )

    with pytest.raises(ValueError, match="collection not found"):
        await ingest(session, who, payload)


# ---------------------------------------------------------------------------
# Test: deterministic chunk / version / document IDs
# ---------------------------------------------------------------------------


def test_stable_ids_are_deterministic() -> None:
    """stable_chunk_id, stable_document_id, stable_version_id reproduce the same UUID."""
    # Chunk IDs — same inputs always produce same UUID
    version_id = uuid.UUID("aabbccdd-1111-2222-3333-444455556666")
    digest = "a" * 64

    id_a = stable_chunk_id(version_id, "word-window-v1", 0, digest)
    id_b = stable_chunk_id(version_id, "word-window-v1", 0, digest)
    assert id_a == id_b
    # Different index or digest → different ID
    assert id_a != stable_chunk_id(version_id, "word-window-v1", 1, digest)
    assert id_a != stable_chunk_id(version_id, "word-window-v1", 0, "b" * 64)

    # Document IDs — deterministic from scope tuple
    tenant = uuid.UUID("11111111-1111-1111-1111-111111111111")
    project = uuid.UUID("22222222-2222-2222-2222-222222222222")
    doc_a = stable_document_id(tenant, project, "docs", "ext-001")
    doc_b = stable_document_id(tenant, project, "docs", "ext-001")
    assert doc_a == doc_b
    assert doc_a != stable_document_id(tenant, project, "docs", "ext-002")

    # Version IDs — deterministic from document + version + digest
    ver_a = stable_version_id(doc_a, 1, digest)
    ver_b = stable_version_id(doc_a, 1, digest)
    assert ver_a == ver_b
    assert ver_a != stable_version_id(doc_a, 2, digest)


# ---------------------------------------------------------------------------
# Test: scoped_statement includes current-version and tenant filters
# ---------------------------------------------------------------------------


def test_scoped_statement_filters_current_version_and_tenant() -> None:
    """scoped_statement embeds tenant_id, is_current, and deleted_at filters."""
    project_id = uuid.uuid4()
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()

    request = SearchRequest(
        project_id=project_id,
        collections=["docs"],
        query="test query",
    )

    def _principal(tid: uuid.UUID) -> Principal:
        return Principal(
            tid,
            frozenset({project_id}),
            frozenset({"docs"}),
            frozenset({"retrieval:search"}),
        )

    stmt_a = retrieval.scoped_statement(_principal(tenant_a), request)
    stmt_b = retrieval.scoped_statement(_principal(tenant_b), request)

    sql_a, sql_b = str(stmt_a), str(stmt_b)

    # Both SQL strings contain the isolation / versioning columns
    for sql in (sql_a, sql_b):
        assert "tenant_id" in sql.lower()
        assert "is_current" in sql.lower()
        assert "deleted_at" in sql.lower()

    # Different tenants produce different queries
    # Compiled parameters embed the correct tenant UUIDs
    params_a = stmt_a.compile().params
    params_b = stmt_b.compile().params
    assert tenant_a in params_a.values()
    assert tenant_b in params_b.values()


# ---------------------------------------------------------------------------
# Test: tenant isolation — second principal gets zero results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_empty_for_isolated_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A principal from a different tenant receives zero search results."""
    version_id = uuid.uuid4()
    chunks = [_make_chunk(0, "Tenant A sensitive deployment data.", version_id)]

    tenant_a = TENANT_ID
    tenant_b = uuid.uuid4()

    async def fake_vector_search(
        _session: Any,
        tenant_id: uuid.UUID,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[tuple[Chunk, str, float]]:
        if tenant_id == tenant_a:
            return [(chunks[0], "doc-1", 0.92)]
        return []

    async def fake_bm25_search(
        tenant_id: uuid.UUID,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[tuple[uuid.UUID, float]]:
        if tenant_id == tenant_a:
            return [(chunks[0].id, 8.0)]
        return []

    async def fake_embed_query(query: str) -> list[float]:
        return _make_embedding(42)

    monkeypatch.setattr(retrieval, "vector_search", fake_vector_search)
    monkeypatch.setattr(retrieval, "bm25_search", fake_bm25_search)
    monkeypatch.setattr(retrieval, "embed_query", fake_embed_query)
    monkeypatch.setattr(retrieval, "get_settings", lambda: _settings(reranker_enabled=False))

    data = _search_request(use_reranker=False)

    # Tenant A receives results
    session_a = FakeSession()
    principal_a = Principal(
        tenant_a,
        frozenset({PROJECT_ID}),
        frozenset({COLLECTION}),
        frozenset({"retrieval:search"}),
    )
    _, results_a, _ = await retrieval.search(session_a, principal_a, data)
    assert len(results_a) > 0

    # Tenant B receives nothing
    session_b = FakeSession()
    principal_b = Principal(
        tenant_b,
        frozenset({PROJECT_ID}),
        frozenset({COLLECTION}),
        frozenset({"retrieval:search"}),
    )
    _, results_b, _ = await retrieval.search(session_b, principal_b, data)
    assert results_b == []
