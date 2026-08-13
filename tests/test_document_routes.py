import io
import uuid
from typing import Any

import pytest
from fastapi import HTTPException, UploadFile

from rag_platform.api.routes import documents
from rag_platform.api.schemas import DocumentBatchCreate, DocumentCreate, DocumentUpdate
from rag_platform.core.auth import Principal
from rag_platform.db.models import Chunk, Document, DocumentVersion, IndexingJob, Status
from rag_platform.services.extraction import ExtractedDocument


class FakeSession:
    def __init__(
        self,
        values: list[Any] | None = None,
        rows: list[object] | None = None,
    ) -> None:
        self.values = list(values or [])
        self.rows = rows or []
        self.added: list[object] = []
        self.commits = 0

    async def scalar(self, statement: object) -> Any:
        return self.values.pop(0)

    async def scalars(self, statement: object) -> Any:
        class Rows:
            def __init__(self, values: list[object]) -> None:
                self.values = values

            def all(self) -> list[object]:
                return self.values

        return Rows(self.rows)

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.commits += 1


def scoped_principal(tenant_id: uuid.UUID, project_id: uuid.UUID, *permissions: str) -> Principal:
    return Principal(
        tenant_id,
        uuid.uuid4(),
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
async def test_create_document_batch_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    payloads = [
        DocumentCreate(
            project_id=project_id,
            collection="manuals",
            external_document_id=f"guide-{index}",
            content=f"content {index}",
        )
        for index in range(2)
    ]

    async def ingest(session: object, who: Principal, data: DocumentCreate) -> DocumentVersion:
        row = version(tenant_id, project_id)
        row.external_document_id = data.external_document_id
        return row

    monkeypatch.setattr(documents, "ingest", ingest)
    result = await documents.create_batch(
        DocumentBatchCreate(documents=payloads),
        scoped_principal(tenant_id, project_id, "documents:write"),
        FakeSession(),
    )
    assert [item.external_document_id for item in result] == ["guide-0", "guide-1"]

    async def conflict(*args: object) -> DocumentVersion:
        raise ValueError("collection not found")

    monkeypatch.setattr(documents, "ingest", conflict)
    with pytest.raises(HTTPException) as error:
        await documents.create_batch(
            DocumentBatchCreate(documents=payloads),
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
        lock_version=1,
    )
    who = scoped_principal(tenant_id, project_id, "documents:read", "documents:delete")
    result = await documents.get(document_id, who, FakeSession([row]))
    assert result["current_version"] == 2
    assert result["lock_version"] == 1

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
async def test_list_documents_and_chunks_are_scoped() -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = Document(
        id=document_id,
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        external_document_id="guide-1",
        current_version=1,
        lock_version=1,
        metadata_={"topic": "operations"},
    )
    who = scoped_principal(tenant_id, project_id, "documents:read")
    listed = await documents.list_documents(
        project_id, "manuals", 50, 0, who, FakeSession(rows=[row])
    )
    assert listed == [
        {
            "id": document_id,
            "project_id": project_id,
            "collection": "manuals",
            "external_document_id": "guide-1",
            "current_version": 1,
            "lock_version": 1,
            "metadata": {"topic": "operations"},
        }
    ]

    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=document_id,
        document_version_id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        chunk_index=0,
        chunk_type="child",
        content="Operations guide",
        token_count=2,
        language="en",
        content_hash="b" * 64,
        metadata_={"topic": "operations"},
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
    )
    chunks = await documents.document_chunks(
        document_id,
        100,
        0,
        who,
        FakeSession([row], rows=[chunk]),
    )
    assert chunks[0]["content"] == "Operations guide"
    assert chunks[0]["embedding_dimension"] == 1024

    with pytest.raises(HTTPException) as error:
        await documents.document_chunks(document_id, 100, 0, who, FakeSession([None]))
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_update_document_creates_next_version(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    row = Document(
        id=document_id,
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
        external_document_id="guide-1",
        current_version=2,
        lock_version=3,
        metadata_={"topic": "operations"},
    )
    current = version(tenant_id, project_id)
    current.document_id = document_id
    current.version = 2
    current.document_type = "text"
    current.title = "Existing title"
    current.language = "en"
    captured: list[DocumentCreate] = []

    async def ingest(session: object, who: Principal, data: DocumentCreate) -> DocumentVersion:
        captured.append(data)
        created = version(tenant_id, project_id)
        created.document_id = document_id
        created.version = data.version
        return created

    monkeypatch.setattr(documents, "ingest", ingest)
    result = await documents.update_document(
        document_id,
        DocumentUpdate(expected_lock_version=3, content="Updated content"),
        scoped_principal(tenant_id, project_id, "documents:write"),
        FakeSession([row, current]),
    )
    assert result.version == 3
    assert captured[0].title == "Existing title"
    assert captured[0].metadata == {"topic": "operations"}

    row.lock_version = 4
    with pytest.raises(HTTPException) as error:
        await documents.update_document(
            document_id,
            DocumentUpdate(expected_lock_version=3, content="Stale update"),
            scoped_principal(tenant_id, project_id, "documents:write"),
            FakeSession([row]),
        )
    assert error.value.status_code == 409


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
