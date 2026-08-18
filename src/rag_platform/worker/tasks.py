import asyncio
import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime

from celery import shared_task
from sqlalchemy import delete, select

from rag_platform.core.config import get_settings
from rag_platform.core.embedding_registry import get_active_model
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
from rag_platform.services.chunking import ChunkingConfig, chunk_text
from rag_platform.services.opensearch import (
    OpenSearchUnavailable,
    delete_document_chunks,
    index_chunks,
)
from rag_platform.services.reconciliation import reconcile
from rag_platform.services.versioning import content_hash, stable_chunk_id
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
    return [
        draft.content
        for draft in chunk_text(
            text,
            "recursive",
            ChunkingConfig(target_words, overlap_words, 1),
        )
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
        # Idempotent: skip only if already indexed WITH THE CURRENT MODEL.
        cfg = get_active_model()
        if version.status == Status.indexed and version.embedding_model == cfg.model_name:
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
            settings = get_settings()

            # Model-aware chunking: respect max_input_tokens.
            # Approximate: 1 token ≈ 0.75 words for multilingual models.
            max_words = int(cfg.max_input_tokens * 0.75)
            target = min(settings.chunk_size_words, max_words)
            overlap = min(settings.chunk_overlap_words, target // 3)

            drafts = chunk_text(
                version.content,
                settings.chunk_strategy,
                ChunkingConfig(target, overlap, settings.chunk_min_words),
            )
            parts = [draft.content for draft in drafts]
            vectors = embed(parts, cfg=cfg)
            embedding_dimension = dimension(cfg)
            if vectors and len(vectors[0]) != 1024:  # padded dimension
                raise RuntimeError("embedding padded dimension mismatch")
            # Delete only embeddings for the current model (preserve other models).
            await session.execute(
                delete(ChunkEmbedding).where(
                    ChunkEmbedding.model == cfg.model_name,
                    ChunkEmbedding.chunk_id.in_(
                        select(Chunk.id).where(
                            Chunk.document_version_id == version.id
                        )
                    ),
                )
            )
            version.parser_version = settings.parser_version
            version.chunker_version = settings.chunker_version
            version.embedding_model = cfg.model_name
            version.embedding_revision = settings.embedding_revision
            version.index_version = cfg.index_version
            for draft, vector in zip(drafts, vectors, strict=True):
                digest = content_hash(draft.content)
                chunk_id = stable_chunk_id(
                    version.id,
                    settings.chunker_version,
                    draft.chunk_index,
                    digest,
                )
                # Reuse existing chunk if present (model-independent text).
                existing = await session.get(Chunk, chunk_id)
                if existing is None:
                    chunk = Chunk(
                        id=chunk_id,
                        document_id=version.document_id,
                        document_version_id=version.id,
                        tenant_id=version.tenant_id,
                        project_id=version.project_id,
                        owner_user_id=version.owner_user_id,
                        collection=version.collection,
                        chunk_index=draft.chunk_index,
                        content=draft.content,
                        token_count=len(draft.content.split()),
                        language=version.language,
                        content_hash=digest,
                        metadata_=version.metadata_,
                        source_type=version.document_type,
                        source_id=version.external_document_id,
                        section_title=draft.section_title,
                        start_offset=draft.start_offset,
                        end_offset=draft.end_offset,
                        chunker_version=settings.chunker_version,
                        index_version=cfg.index_version,
                        embedding_model=cfg.model_name,
                        embedding_dimension=embedding_dimension,
                    )
                    session.add(chunk)
                    await session.flush()
                else:
                    chunk = existing
                session.add(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        model=cfg.model_name,
                        model_revision=settings.embedding_revision,
                        backend=settings.embedding_backend,
                        normalization=cfg.normalization,
                        embedding_dimension=embedding_dimension,
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
                    "owner_user_id": str(version.owner_user_id),
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
                document.owner_user_id,
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
def embed_query_task(query: str) -> list[float]:
    """Embed a single query with active model's query_prefix."""
    from rag_platform.worker.embeddings import embed_query

    return embed_query(query)


@shared_task
def reconcile_indexes() -> dict[str, int]:
    async def run() -> dict[str, int]:
        async with Session() as session:
            return await reconcile(session)

    return run_async(run())


@shared_task
def run_evaluation_task(run_id: str) -> None:
    run_async(evaluate_run(uuid.UUID(run_id)))
