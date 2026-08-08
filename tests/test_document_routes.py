import io
import uuid
from typing import Any

import pytest
from fastapi import HTTPException, UploadFile

from rag_platform.api.routes import documents
from rag_platform.api.schemas import DocumentCreate
from rag_platform.core.auth import Principal
from rag_platform.db.models import Document, DocumentVersion, IndexingJob, Status
from rag_platform.services.extraction import ExtractedDocument


class FakeSession:
    def __init__(self, values: list[Any] | None = None) -> None:
        self.values = list(values or [])
        self.added: list[object] = []
        self.commits = 0

    async def scalar(self, statement: object) -> Any:
        return self.values.pop(0)

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.commits += 1


def scoped_principal(tenant_id: uuid.UUID, project_id: uuid.UUID, *permissions: str) -> Principal:
    return Principal(
        tenant_id,
        frozenset({project_id}),
        frozenset({"manuals"}),
        frozenset(permissions),
    )


def version(tenant_id: uuid.UUID, project_id: uuid.UUID) -> DocumentVersion:
    return DocumentVersion(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        external_document_id="guide-1",
        version=1,
        status=Status.queued,
        content_hash="a" * 64,
    )


@pytest.mark.asyncio
async def test_create_document_maps_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    data = DocumentCreate(
        project_id=project_id,
        collection="manuals",
        external_document_id="guide-1",
        content="content",
    )

    async def ingest(*args: object) -> DocumentVersion:
        return version(tenant_id, project_id)

    monkeypatch.setattr(documents, "ingest", ingest)
    result = await documents.create(
        data,
        scoped_principal(tenant_id, project_id, "documents:write"),
        FakeSession(),
    )
    assert result.external_document_id == "guide-1"

    async def conflict(*args: object) -> DocumentVersion:
        raise ValueError("newer version already exists")

    monkeypatch.setattr(documents, "ingest", conflict)
    with pytest.raises(HTTPException) as error:
        await documents.create(
            data,
            scoped_principal(tenant_id, project_id, "documents:write"),
            FakeSession(),
        )
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_get_and_delete_document(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = Document(
        id=document_id,
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        external_document_id="guide-1",
        current_version=2,
    )
    who = scoped_principal(tenant_id, project_id, "documents:read", "documents:delete")
    result = await documents.get(document_id, who, FakeSession([row]))
    assert result["current_version"] == 2

    deleted: list[Document] = []

    async def enqueue(session: object, document: Document) -> None:
        deleted.append(document)

    monkeypatch.setattr(documents, "enqueue_deletion", enqueue)
    await documents.delete(document_id, who, FakeSession([row]))
    assert deleted == [row]

    with pytest.raises(HTTPException) as error:
        await documents.get(document_id, who, FakeSession([None]))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_reindex_current_version(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = Document(
        id=document_id,
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        external_document_id="guide-1",
        current_version=1,
    )
    current = version(tenant_id, project_id)
    job = IndexingJob(id=uuid.uuid4(), payload={"status": "queued"})

    async def enqueue(session: object, item: DocumentVersion) -> IndexingJob:
        return job

    monkeypatch.setattr(documents, "enqueue_indexing", enqueue)
    session = FakeSession([row, current])
    result = await documents.reindex(
        document_id,
        scoped_principal(tenant_id, project_id, "admin:reindex"),
        session,
    )
    assert result == {"job_id": job.id, "status": "queued"}
    assert current.status == Status.queued
    assert session.commits == 1


@pytest.mark.asyncio
async def test_upload_extracts_and_persists_source(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    stored: list[tuple[str, bytes, str]] = []

    async def put(key: str, content: bytes, mime_type: str) -> None:
        stored.append((key, content, mime_type))

    async def ingest(*args: object) -> DocumentVersion:
        return version(tenant_id, project_id)

    monkeypatch.setattr(documents, "put", put)
    monkeypatch.setattr(documents, "ingest", ingest)
    monkeypatch.setattr(
        documents,
        "extract",
        lambda *args: [ExtractedDocument(filename="guide.txt", content="safe text")],
    )
    file = UploadFile(filename="guide.txt", file=io.BytesIO(b"safe text"))
    file.headers = {"content-type": "text/plain"}  # type: ignore[assignment]

    result = await documents.upload(
        project_id,
        "manuals",
        "guide-1",
        file,
        who=scoped_principal(tenant_id, project_id, "documents:write"),
        session=FakeSession([None]),
    )

    assert len(result.documents) == 1
    assert stored[0][1] == b"safe text"
    assert result.source_object_key.endswith("/guide.txt")


@pytest.mark.asyncio
async def test_upload_rejects_non_object_metadata() -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    file = UploadFile(filename="guide.txt", file=io.BytesIO(b"safe text"))
    with pytest.raises(HTTPException) as error:
        await documents.upload(
            project_id,
            "manuals",
            "guide-1",
            file,
            metadata="[]",
            who=scoped_principal(tenant_id, project_id, "documents:write"),
            session=FakeSession(),
        )
    assert error.value.status_code == 422
