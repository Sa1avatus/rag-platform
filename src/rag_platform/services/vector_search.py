import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import Chunk, ChunkEmbedding, Document, DocumentVersion


async def vector_search(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    collections: list[str],
    filters: dict[str, Any],
    query_embedding: list[float],
    model: str,
    top_k: int,
) -> list[tuple[Chunk, str, float]]:
    distance = ChunkEmbedding.embedding.cosine_distance(query_embedding)
    statement = (
        select(Chunk, Document.external_document_id, distance.label("distance"))
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.tenant_id == tenant_id,
            Chunk.project_id == project_id,
            Chunk.collection.in_(collections),
            ChunkEmbedding.model == model,
            DocumentVersion.is_current.is_(True),
            Document.deleted_at.is_(None),
        )
        .order_by(distance)
        .limit(top_k)
    )
    for key, value in filters.items():
        normalized = str(value).lower() if isinstance(value, bool) else str(value)
        statement = statement.where(Chunk.metadata_[key].astext == normalized)
    rows = (await session.execute(statement)).all()
    return [
        (chunk, external_id, max(0.0, 1.0 - float(raw_distance)))
        for chunk, external_id, raw_distance in rows
    ]
