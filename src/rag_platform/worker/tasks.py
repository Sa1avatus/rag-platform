import asyncio
import hashlib
import json
import re
import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime

from celery import shared_task
from redis import Redis
from sqlalchemy import delete, select

from rag_platform.core.config import get_settings
from rag_platform.core.metrics import DOCUMENTS_FAILED, DOCUMENTS_INDEXED
from rag_platform.db.models import (
    Chunk,
    ChunkEmbedding,
    Document,
    DocumentVersion,
    IndexingJob,
    Status,
)
from rag_platform.db.session import Session, engine
from rag_platform.services.opensearch import (
    OpenSearchUnavailable,
    delete_document_chunks,
    index_chunks,
)
from rag_platform.services.readiness import MODEL_READY_KEY
from rag_platform.services.reconciliation import reconcile
from rag_platform.worker.embeddings import dimension, embed
from rag_platform.worker.evaluation import evaluate_run
from rag_platform.worker.outbox import publish_pending


def run_async[T](awaitable: Awaitable[T]) -> T:
    async def isolated() -> T:
        try:
            return await awaitable
        finally:
            await engine.dispose()

    return asyncio.run(isolated())


def chunks(
    text: str,
    target_words: int = 330,
    overlap_words: int = 45,
) -> list[str]:
    words = re.split(r"\s+", text.strip())
    step = max(1, target_words - overlap_words)
    return [
        " ".join(words[start : start + target_words])
        for start in range(0, len(words), step)
        if words[start : start + target_words]
    ]


async def index_version(version_id: uuid.UUID, job_id: uuid.UUID) -> None:
    async with Session() as session:
        version = await session.scalar(
            select(DocumentVersion).where(DocumentVersion.id == version_id).with_for_update()
        )
        job = await session.get(IndexingJob, job_id)
        if version is None or job is None:
            return
        if job.payload.get("status") == "canceled":
            return
        if version.status == Status.indexed:
            job.payload = {**job.payload, "status": "completed", "idempotent": True}
            await session.commit()
            return
        version.status = Status.processing
        job.payload = {
            **job.payload,
            "status": "running",
            "attempt": int(job.payload.get("attempt", 0)) + 1,
            "started_at": datetime.now(UTC).isoformat(),
        }
        await session.commit()
        try:
            parts = chunks(version.content)
            vectors = embed(parts)
            embedding_dimension = dimension()
            if vectors and len(vectors[0]) != embedding_dimension:
                raise RuntimeError("embedding dimension mismatch")
            await session.execute(delete(Chunk).where(Chunk.document_version_id == version.id))
            for index, (content, vector) in enumerate(zip(parts, vectors, strict=True)):
                digest = hashlib.sha256(content.encode()).hexdigest()
                chunk = Chunk(
                    document_id=version.document_id,
                    document_version_id=version.id,
                    tenant_id=version.tenant_id,
                    project_id=version.project_id,
                    collection=version.collection,
                    chunk_index=index,
                    content=content,
                    token_count=len(content.split()),
                    language=version.language,
                    content_hash=digest,
                    metadata_=version.metadata_,
                    embedding_model=get_settings().embedding_model,
                    embedding_dimension=embedding_dimension,
                )
                session.add(chunk)
                await session.flush()
                session.add(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        model=get_settings().embedding_model,
                        embedding=vector,
                    )
                )
            version.status = Status.partially_indexed
            job.payload = {
                **job.payload,
                "status": "running",
                "stage": "bm25",
                "chunks": len(parts),
            }
            await session.commit()
            search_documents = [
                {
                    "tenant_id": str(version.tenant_id),
                    "project_id": str(version.project_id),
                    "collection": version.collection,
                    "document_id": str(chunk.document_id),
                    "chunk_id": str(chunk.id),
                    "content": chunk.content,
                    "language": chunk.language,
                    "metadata": chunk.metadata_,
                }
                for chunk in (
                    await session.scalars(
                        select(Chunk).where(Chunk.document_version_id == version.id)
                    )
                ).all()
            ]
            try:
                await index_chunks(search_documents)
            except OpenSearchUnavailable as exc:
                version = await session.get(DocumentVersion, version_id)
                job = await session.get(IndexingJob, job_id)
                if version:
                    version.error = str(exc)
                if job:
                    job.payload = {
                        **job.payload,
                        "status": "completed_degraded",
                        "finished_at": datetime.now(UTC).isoformat(),
                        "warning": str(exc),
                    }
                await session.commit()
                return
            version = await session.get(DocumentVersion, version_id)
            job = await session.get(IndexingJob, job_id)
            if version:
                version.status = Status.indexed
                version.error = None
            if job:
                job.payload = {
                    **job.payload,
                    "status": "completed",
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            await session.commit()
            DOCUMENTS_INDEXED.inc()
        except Exception as exc:
            await session.rollback()
            version = await session.get(DocumentVersion, version_id)
            job = await session.get(IndexingJob, job_id)
            if version:
                version.status = Status.failed
                version.error = str(exc)[:4000]
            if job:
                job.payload = {
                    **job.payload,
                    "status": "failed",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "error": str(exc)[:4000],
                }
            await session.commit()
            DOCUMENTS_FAILED.inc()
            raise


async def delete_derivatives(document_id: uuid.UUID, job_id: uuid.UUID) -> None:
    async with Session() as session:
        document = await session.get(Document, document_id)
        job = await session.get(IndexingJob, job_id)
        if document is None or job is None:
            return
        if job.payload.get("status") == "canceled":
            return
        job.payload = {
            **job.payload,
            "status": "running",
            "attempt": int(job.payload.get("attempt", 0)) + 1,
            "started_at": datetime.now(UTC).isoformat(),
        }
        await session.commit()
        try:
            await delete_document_chunks(
                document.tenant_id,
                document.project_id,
                document.id,
            )
            await session.execute(delete(Chunk).where(Chunk.document_id == document.id))
            job = await session.get(IndexingJob, job_id)
            if job:
                job.payload = {
                    **job.payload,
                    "status": "completed",
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            await session.commit()
        except Exception as exc:
            await session.rollback()
            job = await session.get(IndexingJob, job_id)
            if job:
                job.payload = {
                    **job.payload,
                    "status": "failed",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "error": str(exc)[:4000],
                }
                await session.commit()
            raise


@shared_task(
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def index_document(self: object, version_id: str, job_id: str) -> None:
    run_async(index_version(uuid.UUID(version_id), uuid.UUID(job_id)))


@shared_task(
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def delete_document_derivatives(
    self: object,
    document_id: str,
    job_id: str,
) -> None:
    run_async(delete_derivatives(uuid.UUID(document_id), uuid.UUID(job_id)))


@shared_task(
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=10,
)
def dispatch_outbox() -> int:
    return run_async(publish_pending())


@shared_task
def embed_query(query: str) -> list[float]:
    return embed([query])[0]


@shared_task
def model_readiness_heartbeat() -> dict[str, object]:
    settings = get_settings()
    detected = dimension()
    payload: dict[str, object] = {
        "model": settings.embedding_model,
        "dimension": detected,
        "device": settings.embedding_device,
    }
    cache = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        cache.set(MODEL_READY_KEY, json.dumps(payload), ex=60)
    finally:
        cache.close()
    return payload


@shared_task
def reconcile_indexes() -> dict[str, int]:
    async def run() -> dict[str, int]:
        async with Session() as session:
            return await reconcile(session)

    return run_async(run())


@shared_task
def run_evaluation_task(run_id: str) -> None:
    run_async(evaluate_run(uuid.UUID(run_id)))
