import hashlib
import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import DocumentCreate
from rag_platform.core.auth import Principal
from rag_platform.db.models import (
    Document,
    DocumentVersion,
    IndexingJob,
    OutboxEvent,
    Status,
)


async def ingest(session: AsyncSession, who: Principal, data: DocumentCreate) -> DocumentVersion:
    who.authorize(data.project_id, [data.collection], "documents:write")
    digest = hashlib.sha256(data.content.encode()).hexdigest()
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
        if existing.content_hash != digest:
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
            tenant_id=who.tenant_id,
            project_id=data.project_id,
            collection=data.collection,
            external_document_id=data.external_document_id,
            current_version=data.version,
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
    version = DocumentVersion(
        document_id=document.id,
        tenant_id=who.tenant_id,
        project_id=data.project_id,
        collection=data.collection,
        external_document_id=data.external_document_id,
        document_type=data.document_type,
        title=data.title,
        content=data.content,
        content_hash=digest,
        language=data.language,
        version=data.version,
        is_current=data.version == document.current_version,
        metadata_=data.metadata,
        status=Status.queued,
    )
    session.add(version)
    await session.flush()
    job = IndexingJob(
        tenant_id=who.tenant_id,
        project_id=data.project_id,
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
            tenant_id=who.tenant_id,
            project_id=data.project_id,
            payload={
                "type": "document.index",
                "version_id": str(version.id),
                "job_id": str(job.id),
                "attempts": 0,
            },
        )
    )
    await session.commit()
    return version
