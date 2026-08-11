from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import DocumentCreate
from rag_platform.core.auth import Principal
from rag_platform.core.config import get_settings
from rag_platform.core.metrics import DOCUMENTS_RECEIVED
from rag_platform.db.models import (
    Collection,
    Document,
    DocumentVersion,
    IndexingJob,
    OutboxEvent,
    Status,
)
from rag_platform.services.versioning import (
    content_hash,
    normalize_content,
    stable_document_id,
    stable_version_id,
)


async def ingest(session: AsyncSession, who: Principal, data: DocumentCreate) -> DocumentVersion:
    who.authorize(data.project_id, [data.collection], "documents:write")
    collection = await session.scalar(
        select(Collection).where(
            Collection.tenant_id == who.tenant_id,
            Collection.project_id == data.project_id,
            Collection.name == data.collection,
        )
    )
    if collection is None:
        raise ValueError("collection not found")
    normalized_content = normalize_content(data.content)
    if not normalized_content:
        raise ValueError("content is empty after normalization")
    digest = content_hash(normalized_content)
    existing = await session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.tenant_id == who.tenant_id,
            DocumentVersion.project_id == data.project_id,
            DocumentVersion.collection == data.collection,
            DocumentVersion.external_document_id == data.external_document_id,
            DocumentVersion.version == data.version,
        )
    )
    if existing:
        if (
            existing.content_hash != digest
            and normalize_content(existing.content) != normalized_content
        ):
            raise ValueError("version already exists with different content")
        return existing
    document = await session.scalar(
        select(Document).where(
            Document.tenant_id == who.tenant_id,
            Document.project_id == data.project_id,
            Document.collection == data.collection,
            Document.external_document_id == data.external_document_id,
        )
    )
    if document is None:
        document = Document(
            id=stable_document_id(
                who.tenant_id,
                data.project_id,
                data.collection,
                data.external_document_id,
            ),
            tenant_id=who.tenant_id,
            project_id=data.project_id,
            collection=data.collection,
            external_document_id=data.external_document_id,
            current_version=data.version,
            metadata_=data.metadata,
        )
        session.add(document)
        await session.flush()
    elif data.version > document.current_version:
        await session.execute(
            update(DocumentVersion)
            .where(DocumentVersion.document_id == document.id)
            .values(is_current=False)
        )
        document.current_version = data.version
        document.lock_version += 1
        document.metadata_ = data.metadata
    settings = get_settings()
    version = DocumentVersion(
        id=stable_version_id(document.id, data.version, digest),
        document_id=document.id,
        tenant_id=who.tenant_id,
        project_id=data.project_id,
        collection=data.collection,
        external_document_id=data.external_document_id,
        document_type=data.document_type,
        title=data.title,
        content=normalized_content,
        content_hash=digest,
        language=data.language,
        version=data.version,
        is_current=data.version == document.current_version,
        metadata_=data.metadata,
        parser_version=settings.parser_version,
        chunker_version=settings.chunker_version,
        embedding_model=settings.embedding_model,
        embedding_revision=settings.embedding_revision,
        index_version=settings.index_version,
        status=Status.queued,
    )
    session.add(version)
    await session.flush()
    await enqueue_indexing(session, version)
    await session.commit()
    DOCUMENTS_RECEIVED.inc()
    return version


async def enqueue_indexing(
    session: AsyncSession,
    version: DocumentVersion,
) -> IndexingJob:
    job = IndexingJob(
        tenant_id=version.tenant_id,
        project_id=version.project_id,
        payload={
            "version_id": str(version.id),
            "status": "queued",
            "attempt": 0,
        },
    )
    session.add(job)
    await session.flush()
    session.add(
        OutboxEvent(
            tenant_id=version.tenant_id,
            project_id=version.project_id,
            payload={
                "type": "document.index",
                "version_id": str(version.id),
                "job_id": str(job.id),
                "attempts": 0,
            },
        )
    )
    return job


async def enqueue_deletion(
    session: AsyncSession,
    document: Document,
) -> IndexingJob:
    document.deleted_at = datetime.now(UTC)
    await session.execute(
        update(DocumentVersion)
        .where(DocumentVersion.document_id == document.id)
        .values(status=Status.deleted, is_current=False)
    )
    job = IndexingJob(
        tenant_id=document.tenant_id,
        project_id=document.project_id,
        payload={
            "document_id": str(document.id),
            "status": "queued",
            "job_type": "document.delete",
            "attempt": 0,
        },
    )
    session.add(job)
    await session.flush()
    session.add(
        OutboxEvent(
            tenant_id=document.tenant_id,
            project_id=document.project_id,
            payload={
                "type": "document.delete",
                "document_id": str(document.id),
                "job_id": str(job.id),
                "attempts": 0,
            },
        )
    )
    await session.commit()
    return job
