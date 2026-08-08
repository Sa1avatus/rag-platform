import time
import uuid
from typing import Any

import httpx
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import SearchRequest
from rag_platform.core.auth import Principal
from rag_platform.core.config import get_settings
from rag_platform.db.models import Chunk, Document, RetrievalRequest
from rag_platform.services.fusion import reciprocal_rank_fusion
from rag_platform.services.opensearch import OpenSearchUnavailable, bm25_search
from rag_platform.services.query_embeddings import embed_query
from rag_platform.services.vector_search import vector_search

ChunkRow = tuple[Chunk, str]


def rrf(
    vector: list[uuid.UUID],
    lexical: list[uuid.UUID],
    k: int = 60,
) -> dict[uuid.UUID, float]:
    return reciprocal_rank_fusion([vector, lexical], k)


def scoped_statement(
    who: Principal,
    data: SearchRequest,
) -> Select[tuple[Chunk, str]]:
    statement = (
        select(Chunk, Document.external_document_id)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.tenant_id == who.tenant_id,
            Chunk.project_id == data.project_id,
            Chunk.collection.in_(data.collections),
            Document.deleted_at.is_(None),
        )
    )
    for key, value in data.filters.items():
        normalized = str(value).lower() if isinstance(value, bool) else str(value)
        statement = statement.where(Chunk.metadata_[key].astext == normalized)
    return statement


async def search(
    session: AsyncSession,
    who: Principal,
    data: SearchRequest,
) -> tuple[uuid.UUID, list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    who.authorize(data.project_id, data.collections, "retrieval:search")
    statement = scoped_statement(who, data)
    settings = get_settings()
    query_vector = await embed_query(data.query)
    vector_hits = await vector_search(
        session,
        who.tenant_id,
        data.project_id,
        data.collections,
        data.filters,
        query_vector,
        settings.embedding_model,
        data.vector_top_k,
    )
    vector_rows = [(chunk, external_id) for chunk, external_id, _ in vector_hits]
    vector_ids = [chunk.id for chunk, _, _ in vector_hits]
    vector_scores = {chunk.id: score for chunk, _, score in vector_hits}

    opensearch_degraded = False
    bm25_scores: dict[uuid.UUID, float] = {}
    try:
        lexical_hits = await bm25_search(
            who.tenant_id,
            data.project_id,
            data.collections,
            data.query,
            data.filters,
            data.bm25_top_k,
        )
        lexical_ids = [chunk_id for chunk_id, _ in lexical_hits]
        bm25_scores = dict(lexical_hits)
    except OpenSearchUnavailable:
        lexical_ids = []
        opensearch_degraded = True

    rows_by_id = {row[0].id: row for row in vector_rows}
    missing_ids = [chunk_id for chunk_id in lexical_ids if chunk_id not in rows_by_id]
    if missing_ids:
        lexical_rows = (
            await session.execute(statement.where(Chunk.id.in_(missing_ids)))
        ).all()
        rows_by_id.update({row[0].id: row for row in lexical_rows})
    lexical_ids = [chunk_id for chunk_id in lexical_ids if chunk_id in rows_by_id]

    fusion_scores = reciprocal_rank_fusion([vector_ids, lexical_ids])
    ranked_ids = sorted(
        fusion_scores,
        key=fusion_scores.__getitem__,
        reverse=True,
    )[: data.fusion_top_k]
    ranked = [rows_by_id[chunk_id] for chunk_id in ranked_ids]

    reranker_degraded = False
    reranker_used = False
    if data.use_reranker and settings.reranker_enabled and ranked:
        try:
            async with httpx.AsyncClient(
                timeout=settings.reranker_timeout_seconds
            ) as reranker:
                response = await reranker.post(
                    f"{settings.reranker_base_url}/v1/rerank",
                    json={
                        "query": data.query,
                        "documents": [row[0].content for row in ranked],
                        "top_k": data.rerank_top_k,
                    },
                )
                response.raise_for_status()
                order = response.json().get("results", [])
                ranked = [
                    ranked[item["index"]]
                    for item in order
                    if item["index"] < len(ranked)
                ]
                reranker_used = True
        except (httpx.HTTPError, KeyError, TypeError):
            reranker_degraded = True

    results = [
        {
            "document_id": str(chunk.document_id),
            "external_document_id": external_id,
            "chunk_id": str(chunk.id),
            "content": chunk.content,
            "metadata": chunk.metadata_,
            "vector_score": vector_scores.get(chunk.id),
            "bm25_score": bm25_scores.get(chunk.id),
            "fusion_score": fusion_scores[chunk.id],
            "final_score": fusion_scores[chunk.id],
        }
        for chunk, external_id in ranked[: data.rerank_top_k]
    ]
    request_id = uuid.uuid4()
    trace = {
        "embedding_model": settings.embedding_model,
        "retrieval_strategy": "hybrid_rrf",
        "opensearch_degraded": opensearch_degraded,
        "reranker_used": reranker_used,
        "reranker_degraded": reranker_degraded,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    session.add(
        RetrievalRequest(
            id=request_id,
            tenant_id=who.tenant_id,
            project_id=data.project_id,
            payload={
                "query": data.query,
                "collections": data.collections,
                "results": results,
                "trace": trace,
            },
        )
    )
    await session.commit()
    return request_id, results, trace
