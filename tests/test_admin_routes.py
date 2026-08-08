import uuid
from typing import Any

import pytest
from fastapi import HTTPException

from rag_platform.api.routes import admin
from rag_platform.api.schemas import ApiKeyCreate, CollectionCreate, ProjectCreate, SearchRequest
from rag_platform.core.auth import Principal
from rag_platform.db.models import IndexingJob, Project


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

    row = session.added[0]
    listed = await admin.projects(FakeSession(rows=[row]))
    assert listed[0]["tenant_id"] == tenant_id
    assert listed[0]["enabled"] is None or listed[0]["enabled"] is True


@pytest.mark.asyncio
async def test_create_collection_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    collection_session = FakeSession()
    collection = await admin.create_collection(
        CollectionCreate(tenant_id=tenant_id, project_id=project_id, name="manuals", settings={}),
        collection_session,
    )
    assert collection["name"] == "manuals"

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
    assert len(retry_session.added) == 1

    cancel_session = FakeSession(scalar_values=[job])
    result = await admin.cancel_indexing_job(job.id, cancel_session)
    assert result["status"] == "canceled"


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
async def test_dashboard_settings_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await admin.dashboard(FakeSession(scalar_values=[3, 9])) == {
        "documents": 3,
        "chunks": 9,
        "embeddings": 9,
    }
    assert (await admin.settings())["default_vector_top_k"] == 30

    async def system_health() -> dict[str, object]:
        return {"status": "operational", "components": []}

    monkeypatch.setattr(admin, "system_health", system_health)
    assert (await admin.health())["status"] == "operational"
