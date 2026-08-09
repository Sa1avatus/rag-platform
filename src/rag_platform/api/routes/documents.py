import json
import uuid
from contextlib import suppress
from pathlib import PurePath
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import DocumentCreate, DocumentRead, UploadRead
from rag_platform.core.auth import Principal, principal
from rag_platform.core.config import get_settings
from rag_platform.db.models import Chunk, Document, DocumentBlob, DocumentVersion, Status
from rag_platform.db.session import get_session
from rag_platform.services.blobs import object_key, put, remove
from rag_platform.services.documents import enqueue_deletion, enqueue_indexing, ingest
from rag_platform.services.extraction import extract
from rag_platform.services.source_safety import UnsafeSourceError

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("/upload", response_model=UploadRead, status_code=202)
async def upload(
    project_id: Annotated[uuid.UUID, Form()],
    collection: Annotated[str, Form()],
    external_document_id: Annotated[str, Form(min_length=1, max_length=300)],
    file: Annotated[UploadFile, File()],
    document_type: Annotated[str, Form()] = "file",
    language: Annotated[str, Form()] = "und",
    version: Annotated[int, Form(ge=1)] = 1,
    metadata: Annotated[str, Form()] = "{}",
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> UploadRead:
    who.authorize(project_id, [collection], "documents:write")
    filename = PurePath(file.filename or "").name
    content = await file.read(get_settings().document_max_bytes + 1)
    await file.close()
    try:
        parsed_metadata = json.loads(metadata)
        if not isinstance(parsed_metadata, dict):
            raise ValueError("metadata must be a JSON object")
        metadata_value = cast(dict[str, Any], parsed_metadata)
        extracted = extract(filename, content, file.content_type)
    except (json.JSONDecodeError, UnicodeError, UnsafeSourceError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc

    upload_id = uuid.uuid4()
    key = object_key(who.tenant_id, project_id, upload_id, filename)
    try:
        await put(key, content, file.content_type or "application/octet-stream")
    except Exception as exc:
        raise HTTPException(503, "source object storage is unavailable") from exc

    created: list[DocumentRead] = []
    try:
        for index, item in enumerate(extracted):
            item_external_id = external_document_id
            if len(extracted) > 1:
                item_external_id = f"{external_document_id}:{index}:{item.filename}"
            row = await ingest(
                session,
                who,
                DocumentCreate(
                    project_id=project_id,
                    collection=collection,
                    external_document_id=item_external_id,
                    document_type=document_type,
                    title=item.filename,
                    content=item.content,
                    language=language,
                    version=version,
                    metadata=metadata_value,
                ),
            )
            blob = await session.scalar(
                select(DocumentBlob).where(DocumentBlob.version_id == row.id)
            )
            if blob is None:
                session.add(
                    DocumentBlob(
                        version_id=row.id,
                        object_key=key,
                        mime_type=file.content_type or "application/octet-stream",
                        size_bytes=len(content),
                    )
                )
                await session.commit()
            created.append(_document_read(row))
    except Exception:
        if not created:
            with suppress(Exception):
                await remove(key)
        raise
    return UploadRead(documents=created, source_object_key=key)


def _document_read(version: DocumentVersion) -> DocumentRead:
    return DocumentRead(
        id=version.document_id,
        external_document_id=version.external_document_id,
        version=version.version,
        status=version.status.value,
        content_hash=version.content_hash,
    )


@router.post("", response_model=DocumentRead, status_code=202)
async def create(
    data: DocumentCreate,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> DocumentRead:
    try:
        version = await ingest(session, who, data)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _document_read(version)


@router.get("")
async def list_documents(
    project_id: uuid.UUID,
    collection: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    collections = [collection] if collection else sorted(who.collections)
    who.authorize(project_id, collections, "documents:read")
    statement = (
        select(Document)
        .where(
            Document.tenant_id == who.tenant_id,
            Document.project_id == project_id,
            Document.collection.in_(collections),
            Document.deleted_at.is_(None),
        )
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await session.scalars(statement)).all()
    return [
        {
            "id": row.id,
            "project_id": row.project_id,
            "collection": row.collection,
            "external_document_id": row.external_document_id,
            "current_version": row.current_version,
            "metadata": row.metadata_,
        }
        for row in rows
    ]


@router.get("/{document_id}")
async def get(
    document_id: uuid.UUID,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == who.tenant_id,
            Document.project_id.in_(who.project_ids),
            Document.deleted_at.is_(None),
        )
    )
    if row is None:
        raise HTTPException(404, "document not found")
    who.authorize(row.project_id, [row.collection], "documents:read")
    return {
        "id": row.id,
        "external_document_id": row.external_document_id,
        "current_version": row.current_version,
    }


@router.get("/{document_id}/chunks")
async def document_chunks(
    document_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    document = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == who.tenant_id,
            Document.project_id.in_(who.project_ids),
            Document.deleted_at.is_(None),
        )
    )
    if document is None:
        raise HTTPException(404, "document not found")
    who.authorize(document.project_id, [document.collection], "documents:read")
    rows = (
        await session.scalars(
            select(Chunk)
            .where(
                Chunk.document_id == document.id,
                Chunk.tenant_id == who.tenant_id,
                Chunk.project_id == document.project_id,
                Chunk.collection == document.collection,
            )
            .order_by(Chunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return [
        {
            "id": row.id,
            "parent_chunk_id": row.parent_chunk_id,
            "chunk_index": row.chunk_index,
            "chunk_type": row.chunk_type,
            "content": row.content,
            "token_count": row.token_count,
            "language": row.language,
            "content_hash": row.content_hash,
            "metadata": row.metadata_,
            "embedding_model": row.embedding_model,
            "embedding_dimension": row.embedding_dimension,
        }
        for row in rows
    ]


@router.delete("/{document_id}", status_code=204)
async def delete(
    document_id: uuid.UUID,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> None:
    row = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == who.tenant_id,
        )
    )
    if row is None:
        raise HTTPException(404, "document not found")
    who.authorize(row.project_id, [row.collection], "documents:delete")
    await enqueue_deletion(session, row)


@router.post("/{document_id}/reindex", status_code=202)
async def reindex(
    document_id: uuid.UUID,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    document = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == who.tenant_id,
            Document.deleted_at.is_(None),
        )
    )
    if document is None:
        raise HTTPException(404, "document not found")
    who.authorize(document.project_id, [document.collection], "admin:reindex")
    version = await session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document.id,
            DocumentVersion.version == document.current_version,
        )
    )
    if version is None:
        raise HTTPException(409, "current document version is missing")
    version.status = Status.queued
    version.error = None
    job = await enqueue_indexing(session, version)
    await session.commit()
    return {"job_id": job.id, "status": "queued"}
