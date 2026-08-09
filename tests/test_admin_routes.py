import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import HTTPException

from rag_platform.api.routes import admin
from rag_platform.api.schemas import (
    ApiKeyCreate,
    CollectionCreate,
    CollectionUpdate,
    ConfigurationComparisonRequest,
    EmbeddingReindexRequest,
    EvaluationRunCreate,
    ProjectCreate,
    ProjectUpdate,
    RetrievalConfiguration,
    RuntimeSettingsUpdate,
    SearchRequest,
    TenantCreate,
)
from rag_platform.core.auth import Principal
from rag_platform.db.models import (
    AuditLog,
    Chunk,
    Document,
    EvaluationDataset,
    EvaluationResult,
    IndexingJob,
    Project,
    RetrievalFeedback,
    RetrievalRequest,
)


class ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def all(self) -> list[object]:
        return self.rows


class FakeSession:
    def __init__(self, *, scalar_values: list[Any] | None = None, rows: list[object] | None = None):
        self.scalar_values = list(scalar_values or [])
        self.rows = rows or []
        self.added: list[object] = []
        self.commits = 0

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.commits += 1

    async def flush(self) -> None:
        for value in self.added:
            if getattr(value, "id", None) is None:
                value.id = uuid.uuid4()

    async def refresh(self, value: Any) -> None:
        if getattr(value, "id", None) is None:
            value.id = uuid.uuid4()

    async def scalar(self, statement: object) -> Any:
        return self.scalar_values.pop(0)

    async def scalars(self, statement: object) -> ScalarRows:
        return ScalarRows(self.rows)

    async def get(self, model: object, identifier: uuid.UUID) -> Any:
        return self.scalar_values.pop(0)


@pytest.mark.asyncio
async def test_create_and_list_projects() -> None:
    tenant_id = uuid.uuid4()
    session = FakeSession()
    created = await admin.create_project(
        ProjectCreate(tenant_id=tenant_id, slug="docs", name="Documentation"), session
    )
    assert created["slug"] == "docs"
    assert session.commits == 1
    assert session.added[1].payload["action"] == "project.create"  # type: ignore[attr-defined]

    row = session.added[0]
    listed = await admin.projects(FakeSession(rows=[row]))
    assert listed[0]["tenant_id"] == tenant_id
    assert listed[0]["enabled"] is None or listed[0]["enabled"] is True


@pytest.mark.asyncio
async def test_admin_documents_and_chunks_are_explicitly_scoped() -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    document = Document(
        id=document_id,
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        external_document_id="guide",
        current_version=2,
        lock_version=2,
        metadata_={"source": "test"},
    )
    listed = await admin.admin_documents(
        tenant_id, project_id, "manuals", 100, 0, FakeSession(rows=[document])
    )
    assert listed[0]["tenant_id"] == tenant_id
    assert listed[0]["collection"] == "manuals"

    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=document_id,
        document_version_id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        chunk_index=0,
        chunk_type="child",
        content="Safe content",
        token_count=2,
        language="en",
        content_hash="hash",
        metadata_={},
        embedding_model="model",
        embedding_dimension=3,
    )
    chunks = await admin.admin_document_chunks(
        document_id,
        tenant_id,
        project_id,
        "manuals",
        100,
        0,
        FakeSession(scalar_values=[document], rows=[chunk]),
    )
    assert chunks == [
        {
            "id": chunk.id,
            "chunk_index": 0,
            "chunk_type": "child",
            "content": "Safe content",
            "token_count": 2,
            "language": "en",
        }
    ]


@pytest.mark.asyncio
async def test_admin_evaluation_listing_run_and_results() -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    dataset = EvaluationDataset(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        payload={"name": "baseline", "version": 1, "collections": ["manuals"]},
    )
    datasets = await admin.admin_evaluation_datasets(
        tenant_id, project_id, FakeSession(rows=[dataset])
    )
    assert datasets[0]["name"] == "baseline"

    create_session = FakeSession(scalar_values=[dataset])
    created = await admin.admin_create_evaluation_run(
        EvaluationRunCreate(dataset_id=dataset.id), tenant_id, project_id, create_session
    )
    assert created["status"] == "queued"
    assert create_session.commits == 1
    assert create_session.added[1].payload["type"] == "evaluation.run"
    assert create_session.added[2].payload["action"] == "evaluation.run"

    run = create_session.added[0]
    runs = await admin.admin_evaluation_runs(tenant_id, project_id, FakeSession(rows=[run]))
    assert runs[0]["dataset_id"] == str(dataset.id)
    result = EvaluationResult(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        payload={"run_id": str(run.id), "recall_at_5": 1.0},
    )
    detail = await admin.admin_evaluation_run(
        run.id, tenant_id, project_id, FakeSession(scalar_values=[run], rows=[result])
    )
    assert detail["results"][0]["recall_at_5"] == 1.0  # type: ignore[index]


@pytest.mark.asyncio
async def test_admin_feedback_is_explicitly_scoped() -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    row = RetrievalFeedback(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        created_at=datetime.now(UTC),
        payload={
            "request_id": str(uuid.uuid4()),
            "chunk_id": str(uuid.uuid4()),
            "collection": "manuals",
            "relevant": True,
            "relevance_grade": 3,
            "comment": "useful",
        },
    )
    result = await admin.admin_feedback(
        tenant_id, project_id, True, "manuals", 100, FakeSession(rows=[row])
    )
    assert result[0]["tenant_id"] == tenant_id
    assert result[0]["relevant"] is True


@pytest.mark.asyncio
async def test_get_and_update_project() -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    row = Project(
        id=project_id,
        tenant_id=tenant_id,
        slug="docs",
        name="Docs",
        description="Old",
        enabled=True,
    )
    session = FakeSession(scalar_values=[row, row])
    updated = await admin.update_project(
        project_id,
        ProjectUpdate(name="Knowledge", description="New", enabled=False),
        session,
    )
    assert updated["name"] == "Knowledge"
    assert updated["enabled"] is False
    assert updated["tenant_id"] == tenant_id
    assert session.commits == 1
    assert session.added[0].payload["action"] == "project.update"  # type: ignore[attr-defined]

    with pytest.raises(HTTPException) as error:
        await admin.project(project_id, FakeSession(scalar_values=[None]))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_create_tenant() -> None:
    session = FakeSession()
    created = await admin.create_tenant(TenantCreate(name="E2E Tenant"), session)
    assert created["name"] == "E2E Tenant"
    assert created["id"] is not None
    assert session.commits == 1
    assert session.added[1].tenant_id == created["id"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_create_collection_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    collection_session = FakeSession()
    collection = await admin.create_collection(
        CollectionCreate(tenant_id=tenant_id, project_id=project_id, name="manuals", settings={}),
        collection_session,
    )
    assert collection["name"] == "manuals"
    assert collection_session.added[1].payload["action"] == "collection.create"  # type: ignore[attr-defined]

    monkeypatch.setattr(admin.secrets, "token_urlsafe", lambda size: "stable-secret")
    monkeypatch.setattr(admin, "hash_key", lambda value: "hashed")
    key_session = FakeSession()
    key = await admin.create_key(
        ApiKeyCreate(
            tenant_id=tenant_id,
            allowed_project_ids=[project_id],
            allowed_collections=["manuals"],
            permissions=["retrieval:search"],
        ),
        key_session,
    )
    assert key["api_key"] == "rag_stable-secret"
    assert key_session.added[0].key_hash == "hashed"  # type: ignore[attr-defined]
    audit_payload = key_session.added[1].payload  # type: ignore[attr-defined]
    assert audit_payload == {
        "action": "api_key.create",
        "resource_type": "api_key",
        "resource_id": str(key_session.added[0].id),  # type: ignore[attr-defined]
    }
    assert "rag_stable-secret" not in str(audit_payload)


@pytest.mark.asyncio
async def test_list_and_revoke_api_keys() -> None:
    tenant_id, project_id, key_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = admin.ApiKey(
        id=key_id,
        tenant_id=tenant_id,
        prefix="rag_visible",
        key_hash="secret-hash",
        allowed_project_ids=[project_id],
        allowed_collections=["manuals"],
        permissions=["retrieval:search"],
        revoked=False,
    )
    listed = await admin.api_keys(FakeSession(rows=[row]))
    assert listed[0]["prefix"] == "rag_visible"
    assert "key_hash" not in listed[0]

    session = FakeSession(scalar_values=[row])
    await admin.revoke_api_key(key_id, session)
    assert row.revoked is True
    assert session.commits == 1
    assert session.added[0].payload["action"] == "api_key.revoke"  # type: ignore[attr-defined]

    await admin.revoke_api_key(key_id, FakeSession(scalar_values=[row]))
    with pytest.raises(HTTPException) as error:
        await admin.revoke_api_key(key_id, FakeSession(scalar_values=[None]))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_get_and_update_collection() -> None:
    tenant_id, project_id, collection_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = admin.Collection(
        id=collection_id,
        tenant_id=tenant_id,
        project_id=project_id,
        name="manuals",
        description="Old",
        settings={"vector_top_k": 20},
    )
    session = FakeSession(scalar_values=[row, row])
    updated = await admin.update_collection(
        collection_id,
        CollectionUpdate(description="New", settings={"vector_top_k": 30}),
        session,
    )
    assert updated["description"] == "New"
    assert updated["settings"] == {"vector_top_k": 30}
    assert updated["project_id"] == project_id
    assert session.added[0].payload["action"] == "collection.update"  # type: ignore[attr-defined]

    with pytest.raises(HTTPException) as error:
        await admin.collection(collection_id, FakeSession(scalar_values=[None]))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_reindex_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id, collection_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = admin.Collection(
        id=collection_id,
        tenant_id=tenant_id,
        project_id=project_id,
        name="manuals",
    )

    async def reindex(
        session: object,
        scoped_tenant_id: uuid.UUID,
        scoped_project_id: uuid.UUID,
        name: str,
    ) -> dict[str, int]:
        assert (scoped_tenant_id, scoped_project_id, name) == (
            tenant_id,
            project_id,
            "manuals",
        )
        return {"requeued": 2, "skipped_active": 1}

    monkeypatch.setattr(admin, "reindex_collection", reindex)
    result = await admin.reindex_admin_collection(
        collection_id,
        session := FakeSession(scalar_values=[row]),
    )
    assert result == {"requeued": 2, "skipped_active": 1}
    assert session.added[0].payload["action"] == "collection.reindex"  # type: ignore[attr-defined]

    with pytest.raises(HTTPException) as error:
        await admin.reindex_admin_collection(
            collection_id,
            FakeSession(scalar_values=[None]),
        )
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_compare_collection_configurations(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id, collection_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = admin.Collection(
        id=collection_id,
        tenant_id=tenant_id,
        project_id=project_id,
        name="manuals",
    )
    shared, baseline_only, candidate_only = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    calls: list[SearchRequest] = []

    async def search(
        session: object,
        who: Principal,
        request: SearchRequest,
    ) -> tuple[uuid.UUID, list[dict[str, object]], dict[str, object]]:
        assert who.tenant_id == tenant_id
        assert request.collections == ["manuals"]
        calls.append(request)
        chunks = [shared, baseline_only] if len(calls) == 1 else [shared, candidate_only]
        return uuid.uuid4(), [{"chunk_id": chunk} for chunk in chunks], {"latency_ms": 1.0}

    monkeypatch.setattr(admin, "search", search)
    result = await admin.compare_collection_configurations(
        collection_id,
        ConfigurationComparisonRequest(
            query="operations",
            filters={"approved": True},
            baseline=RetrievalConfiguration(vector_top_k=10, use_reranker=False),
            candidate=RetrievalConfiguration(vector_top_k=40, use_reranker=True),
        ),
        FakeSession(scalar_values=[row]),
    )
    assert [request.vector_top_k for request in calls] == [10, 40]
    assert all(request.filters == {"approved": True} for request in calls)
    assert result["overlap"] == {"shared_chunks": 1, "baseline_only": 1, "candidate_only": 1}

    with pytest.raises(HTTPException) as error:
        await admin.compare_collection_configurations(
            collection_id,
            ConfigurationComparisonRequest(
                query="operations",
                baseline=RetrievalConfiguration(),
                candidate=RetrievalConfiguration(),
            ),
            FakeSession(scalar_values=[None]),
        )
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_reconcile_project_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    row = Project(id=project_id, tenant_id=tenant_id, slug="docs", name="Docs")

    async def reconcile(session: object, selected_project_id: uuid.UUID) -> dict[str, int]:
        assert selected_project_id == project_id
        return {"indexing_requeued": 2, "deletion_requeued": 1}

    monkeypatch.setattr(admin, "reconcile", reconcile)
    session = FakeSession(scalar_values=[row])
    result = await admin.reconcile_project(project_id, session)
    assert result == {"indexing_requeued": 2, "deletion_requeued": 1}
    assert session.added[0].payload["action"] == "project.reconcile"  # type: ignore[attr-defined]

    with pytest.raises(HTTPException) as error:
        await admin.reconcile_project(project_id, FakeSession(scalar_values=[None]))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_retry_and_cancel_indexing_jobs() -> None:
    job = IndexingJob(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        payload={"status": "failed", "job_type": "document.index", "version_id": str(uuid.uuid4())},
    )
    retry_session = FakeSession(scalar_values=[job])
    result = await admin.retry_indexing_job(job.id, retry_session)
    assert result["status"] == "queued"
    assert len(retry_session.added) == 2
    assert retry_session.added[1].payload["action"] == "indexing_job.retry"  # type: ignore[attr-defined]

    cancel_session = FakeSession(scalar_values=[job])
    result = await admin.cancel_indexing_job(job.id, cancel_session)
    assert result["status"] == "canceled"
    assert cancel_session.added[0].payload["action"] == "indexing_job.cancel"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_retry_filtered_indexing_jobs_batches_valid_targets() -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    valid = IndexingJob(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        payload={
            "status": "dead_letter",
            "job_type": "document.delete",
            "document_id": str(uuid.uuid4()),
        },
    )
    invalid = IndexingJob(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        payload={"status": "dead_letter", "job_type": "document.index"},
    )
    session = FakeSession(rows=[valid, invalid])
    result = await admin.retry_filtered_indexing_jobs(
        status="dead_letter",
        project_id=project_id,
        limit=25,
        session=session,
    )
    assert result == {"matched": 2, "retried": 1, "skipped_invalid": 1}
    assert valid.payload["status"] == "queued"
    assert invalid.payload["status"] == "dead_letter"
    assert session.commits == 1
    assert session.added[0].payload["type"] == "document.delete"  # type: ignore[attr-defined]
    assert session.added[1].payload["action"] == "indexing_job.retry"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_get_indexing_job() -> None:
    tenant_id, project_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = IndexingJob(
        id=job_id,
        tenant_id=tenant_id,
        project_id=project_id,
        payload={"status": "completed", "chunks": 4},
    )
    result = await admin.indexing_job(job_id, FakeSession(scalar_values=[row]))
    assert result == {
        "id": job_id,
        "tenant_id": tenant_id,
        "project_id": project_id,
        "status": "completed",
        "chunks": 4,
    }

    with pytest.raises(HTTPException) as error:
        await admin.indexing_job(job_id, FakeSession(scalar_values=[None]))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_retry_rejects_invalid_job_states() -> None:
    job = IndexingJob(id=uuid.uuid4(), payload={"status": "running"})
    with pytest.raises(HTTPException) as error:
        await admin.retry_indexing_job(job.id, FakeSession(scalar_values=[job]))
    assert error.value.status_code == 409

    with pytest.raises(HTTPException) as missing:
        await admin.cancel_indexing_job(uuid.uuid4(), FakeSession(scalar_values=[None]))
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_admin_search_uses_project_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    project = Project(id=project_id, tenant_id=tenant_id, slug="docs", name="Docs")
    data = SearchRequest(project_id=project_id, collections=["manuals"], query="query")

    async def search(
        session: object, who: Principal, request: object
    ) -> tuple[uuid.UUID, list[object], dict[str, object]]:
        assert who.tenant_id == tenant_id
        return uuid.uuid4(), [], {"reranker_degraded": False}

    monkeypatch.setattr(admin, "search", search)
    result = await admin.admin_search(data, FakeSession(scalar_values=[project]))
    assert result["results"] == []

    with pytest.raises(HTTPException) as error:
        await admin.admin_search(data, FakeSession(scalar_values=[None]))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_retrieval_trace_detail_and_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id, request_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = RetrievalRequest(
        id=request_id,
        tenant_id=tenant_id,
        project_id=project_id,
        payload={
            "query": "operations",
            "collections": ["manuals"],
            "filters": {"approved": True},
            "configuration": {
                "vector_top_k": 10,
                "bm25_top_k": 11,
                "fusion_top_k": 9,
                "rerank_top_k": 3,
                "use_reranker": False,
            },
            "results": [],
            "trace": {"retrieval_strategy": "hybrid_rrf"},
        },
    )
    repeated_id = uuid.uuid4()

    async def search(
        session: object,
        who: Principal,
        data: SearchRequest,
    ) -> tuple[uuid.UUID, list[object], dict[str, object]]:
        assert who.tenant_id == tenant_id
        assert data.filters == {"approved": True}
        assert data.vector_top_k == 10
        assert data.use_reranker is False
        return repeated_id, [], {"retrieval_strategy": "hybrid_rrf"}

    monkeypatch.setattr(admin, "search", search)
    detail = await admin.retrieval_trace(request_id, FakeSession(scalar_values=[row]))
    assert detail["query"] == "operations"
    repeated = await admin.repeat_retrieval_trace(
        request_id,
        FakeSession(scalar_values=[row]),
    )
    assert repeated["request_id"] == repeated_id

    with pytest.raises(HTTPException) as error:
        await admin.retrieval_trace(request_id, FakeSession(scalar_values=[None]))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_dashboard_settings_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await admin.dashboard(FakeSession(scalar_values=[3, 9])) == {
        "documents": 3,
        "chunks": 9,
        "embeddings": 9,
    }
    settings = await admin.settings(FakeSession())
    vector = next(item for item in settings["settings"] if item["key"] == "default_vector_top_k")  # type: ignore[union-attr]
    assert vector["value"] == 30

    async def system_health() -> dict[str, object]:
        return {"status": "operational", "components": []}

    monkeypatch.setattr(admin, "system_health", system_health)
    assert (await admin.health())["status"] == "operational"

    monkeypatch.setattr(
        admin,
        "system_resources",
        lambda: {"scope": "rag-api-container", "cpu": {"count": 2}},
    )
    assert (await admin.resources())["scope"] == "rag-api-container"


@pytest.mark.asyncio
async def test_update_runtime_settings_is_persistent_and_audited() -> None:
    session = FakeSession()
    response = await admin.update_settings(
        RuntimeSettingsUpdate(default_vector_top_k=40, reranker_enabled=False),
        session,
    )
    assert response["settings"]
    assert session.commits == 1
    assert session.added[0].key == "default_vector_top_k"  # type: ignore[attr-defined]
    assert session.added[1].key == "reranker_enabled"  # type: ignore[attr-defined]
    assert session.added[2].payload["action"] == "runtime_settings.update"  # type: ignore[attr-defined]

    with pytest.raises(HTTPException) as error:
        await admin.update_settings(RuntimeSettingsUpdate(), FakeSession())
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_metrics_timeseries_validates_range(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)

    async def metric_timeseries(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [{"timestamp": now, "value": 4}]

    monkeypatch.setattr(admin, "metric_timeseries", metric_timeseries)
    response = await admin.metrics_timeseries(
        metric="documents",
        project_id=None,
        collection="manuals",
        from_=now - timedelta(hours=1),
        to=now,
        step="hour",
        aggregation="count",
        session=FakeSession(),
    )
    assert response["points"] == [{"timestamp": now, "value": 4}]

    with pytest.raises(HTTPException) as error:
        await admin.metrics_timeseries(
            metric="documents",
            project_id=None,
            collection=None,
            from_=now,
            to=now,
            step="hour",
            aggregation="count",
            session=FakeSession(),
        )
    assert error.value.status_code == 422

    with pytest.raises(HTTPException, match="90 days"):
        await admin.metrics_timeseries(
            metric="documents",
            project_id=None,
            collection=None,
            from_=now - timedelta(days=91),
            to=now,
            step="day",
            aggregation="count",
            session=FakeSession(),
        )


@pytest.mark.asyncio
async def test_audit_log_lists_safe_action_metadata() -> None:
    tenant_id, project_id, audit_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = AuditLog(
        id=audit_id,
        tenant_id=tenant_id,
        project_id=project_id,
        payload={
            "action": "project.update",
            "resource_type": "project",
            "resource_id": str(project_id),
        },
    )
    listed = await admin.audit_log(
        action="project.update",
        tenant_id=tenant_id,
        project_id=project_id,
        limit=25,
        session=FakeSession(rows=[row]),
    )
    assert listed == [
        {
            "id": audit_id,
            "tenant_id": tenant_id,
            "project_id": project_id,
            "created_at": None,
            "action": "project.update",
            "resource_type": "project",
            "resource_id": str(project_id),
        }
    ]


@pytest.mark.asyncio
async def test_reranker_admin_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    async def status() -> dict[str, object]:
        return {"status": "up", "model": "cross-encoder"}

    async def test_connection() -> dict[str, object]:
        return {"status": "up", "result_count": 1}

    monkeypatch.setattr(admin, "reranker_status", status)
    monkeypatch.setattr(admin, "test_reranker_connection", test_connection)
    assert (await admin.reranker())["model"] == "cross-encoder"
    assert (await admin.test_reranker())["result_count"] == 1


@pytest.mark.asyncio
async def test_embedding_admin_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    async def profile() -> dict[str, object]:
        return {
            "status": "ready",
            "model": "BAAI/bge-m3",
            "dimension": 1024,
            "compatible": True,
        }

    monkeypatch.setattr(admin, "embedding_profile", profile)
    assert (await admin.embeddings())["compatible"] is True
    assert (await admin.check_embeddings())["dimension"] == 1024


@pytest.mark.asyncio
async def test_embedding_reindex_requires_confirmation_and_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(HTTPException) as confirmation:
        await admin.reindex_embedding_model(
            EmbeddingReindexRequest(confirm=False),
            FakeSession(),
        )
    assert confirmation.value.status_code == 409

    async def incompatible() -> dict[str, object]:
        return {"compatible": False}

    monkeypatch.setattr(admin, "embedding_profile", incompatible)
    with pytest.raises(HTTPException) as compatibility:
        await admin.reindex_embedding_model(
            EmbeddingReindexRequest(confirm=True),
            FakeSession(),
        )
    assert compatibility.value.status_code == 409

    async def compatible() -> dict[str, object]:
        return {"compatible": True}

    async def reindex(session: object) -> dict[str, int]:
        return {"requeued": 5, "skipped_active": 2}

    monkeypatch.setattr(admin, "embedding_profile", compatible)
    monkeypatch.setattr(admin, "reindex_embeddings", reindex)
    result = await admin.reindex_embedding_model(
        EmbeddingReindexRequest(confirm=True),
        FakeSession(),
    )
    assert result == {"requeued": 5, "skipped_active": 2}
