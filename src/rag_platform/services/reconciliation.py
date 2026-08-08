import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.core.metrics import INDEXING_QUEUE_SIZE
from rag_platform.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    IndexingJob,
    Status,
)
from rag_platform.services.documents import enqueue_deletion, enqueue_indexing
from rag_platform.services.reconciliation_contract import active_targets


async def reconcile(
    session: AsyncSession,
    project_id: uuid.UUID | None = None,
) -> dict[str, int]:
    jobs = (await session.scalars(select(IndexingJob))).all()
    INDEXING_QUEUE_SIZE.set(sum(1 for job in jobs if job.payload.get("status") == "queued"))
    active_version_ids, active_document_ids = active_targets([job.payload for job in jobs])

    version_statement = select(DocumentVersion).where(
        DocumentVersion.status == Status.partially_indexed,
        DocumentVersion.is_current.is_(True),
    )
    if project_id:
        version_statement = version_statement.where(DocumentVersion.project_id == project_id)
    versions = (await session.scalars(version_statement)).all()
    requeued = 0
    for version in versions:
        if version.id in active_version_ids:
            continue
        version.status = Status.queued
        await enqueue_indexing(session, version)
        requeued += 1

    deleted_statement = (
        select(Document)
        .join(Chunk, Chunk.document_id == Document.id)
        .where(Document.deleted_at.is_not(None))
        .distinct()
    )
    if project_id:
        deleted_statement = deleted_statement.where(Document.project_id == project_id)
    deleted_documents = (await session.scalars(deleted_statement)).all()
    deletion_requeued = 0
    for document in deleted_documents:
        if document.id in active_document_ids:
            continue
        await enqueue_deletion(session, document)
        deletion_requeued += 1
    await session.commit()
    return {
        "indexing_requeued": requeued,
        "deletion_requeued": deletion_requeued,
    }
