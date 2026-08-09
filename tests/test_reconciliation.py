import uuid
from typing import Any

import pytest

from rag_platform.db.models import Document, DocumentVersion, IndexingJob, Status
from rag_platform.services import reconciliation


class Rows:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def all(self) -> list[object]:
        return self.rows


class FakeSession:
    def __init__(self, batches: list[list[object]]) -> None:
        self.batches = batches
        self.committed = False

    async def scalars(self, statement: object) -> Rows:
        return Rows(self.batches.pop(0))

    async def commit(self) -> None:
        self.committed = True


@pytest.mark.asyncio
async def test_reconcile_requeues_only_inactive_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    active_version_id, pending_version_id = uuid.uuid4(), uuid.uuid4()
    active_document_id, deleted_document_id = uuid.uuid4(), uuid.uuid4()
    jobs = [
        IndexingJob(payload={"status": "queued", "version_id": str(active_version_id)}),
        IndexingJob(payload={"status": "running", "document_id": str(active_document_id)}),
    ]
    versions = [
        DocumentVersion(id=active_version_id, status=Status.partially_indexed),
        DocumentVersion(id=pending_version_id, status=Status.partially_indexed),
    ]
    documents = [Document(id=active_document_id), Document(id=deleted_document_id)]
    calls: list[tuple[str, uuid.UUID]] = []

    async def enqueue_indexing(session: object, value: DocumentVersion) -> Any:
        calls.append(("index", value.id))

    async def enqueue_deletion(session: object, value: Document) -> Any:
        calls.append(("delete", value.id))

    monkeypatch.setattr(reconciliation, "enqueue_indexing", enqueue_indexing)
    monkeypatch.setattr(reconciliation, "enqueue_deletion", enqueue_deletion)
    session = FakeSession([jobs, versions, documents])
    result = await reconciliation.reconcile(session, uuid.uuid4())

    assert result == {"indexing_requeued": 1, "deletion_requeued": 1}
    assert calls == [("index", pending_version_id), ("delete", deleted_document_id)]
    assert versions[1].status == Status.queued
    assert session.committed is True


@pytest.mark.asyncio
async def test_collection_reindex_skips_active_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    active = DocumentVersion(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        status=Status.processing,
    )
    inactive = DocumentVersion(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        status=Status.indexed,
    )
    jobs = [IndexingJob(payload={"status": "running", "version_id": str(active.id)})]
    calls: list[uuid.UUID] = []

    async def enqueue(session: object, value: DocumentVersion) -> Any:
        calls.append(value.id)

    monkeypatch.setattr(reconciliation, "enqueue_indexing", enqueue)
    session = FakeSession([jobs, [active, inactive]])
    result = await reconciliation.reindex_collection(
        session,
        tenant_id,
        project_id,
        "manuals",
    )

    assert result == {"requeued": 1, "skipped_active": 1}
    assert calls == [inactive.id]
    assert inactive.status == Status.queued
    assert session.committed is True


@pytest.mark.asyncio
async def test_embedding_reindex_covers_all_current_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = DocumentVersion(id=uuid.uuid4(), status=Status.processing, is_current=True)
    inactive = DocumentVersion(id=uuid.uuid4(), status=Status.indexed, is_current=True)
    jobs = [IndexingJob(payload={"status": "queued", "version_id": str(active.id)})]
    calls: list[uuid.UUID] = []

    async def enqueue(session: object, value: DocumentVersion) -> Any:
        calls.append(value.id)

    monkeypatch.setattr(reconciliation, "enqueue_indexing", enqueue)
    session = FakeSession([jobs, [active, inactive]])
    result = await reconciliation.reindex_embeddings(session)

    assert result == {"requeued": 1, "skipped_active": 1}
    assert calls == [inactive.id]
    assert inactive.status == Status.queued
