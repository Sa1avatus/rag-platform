import time
import uuid
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession
from structlog.contextvars import get_contextvars

from rag_platform.api.schemas import SearchRequest
from rag_platform.core.auth import Principal
from rag_platform.core.config import get_settings
from rag_platform.core.metrics import (
    BM25_SEARCH_DURATION,
    DUPLICATE_CHUNKS_REMOVED,
    FUSION_DURATION,
    RERANKER_DEGRADED,
    RERANKER_DURATION,
    RERANKER_ERRORS,
    RERANKER_REQUESTS,
    RETRIEVAL_DURATION,
    RETRIEVAL_EMPTY,
    RETRIEVAL_FAILURES,
    RETRIEVAL_REQUESTS,
    RETRIEVAL_RESULTS_COUNT,
    VECTOR_SEARCH_DURATION,
)
from rag_platform.db.models import Chunk, Document, DocumentVersion, RetrievalRequest
from rag_platform.services.fusion import reciprocal_rank_fusion
from rag_platform.services.opensearch import OpenSearchUnavailable, bm25_search
from rag_platform.services.query_embeddings import embed_query
from rag_platform.services.reranker import (
    RerankerClient,
    RerankerDocument,
    RerankerUnavailable,
)
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
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.tenant_id == who.tenant_id,
            Chunk.project_id == data.project_id,
            Chunk.collection.in_(data.collections),
            DocumentVersion.is_current.is_(True),
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
    query_vector: list[float] | None = None,
) -> tuple[uuid.UUID, list[dict[str, Any]], dict[str, Any]]:
    RETRIEVAL_REQUESTS.labels(data.mode).inc()
    started = time.perf_counter()
    try:
        return await _search(session, who, data, query_vector)
    except Exception:
        RETRIEVAL_FAILURES.labels(data.mode).inc()
        raise
    finally:
        RETRIEVAL_DURATION.labels(data.mode).observe(time.perf_counter() - started)


async def _search(
    session: AsyncSession,
    who: Principal,
    data: SearchRequest,
    query_vector: list[float] | None = None,
) -> tuple[uuid.UUID, list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    request_id = uuid.uuid4()
    context = get_contextvars()
    correlation_id = str(context.get("correlation_id", request_id))
    who.authorize(data.project_id, data.collections, "retrieval:search")
    statement = scoped_statement(who, data)
    settings = get_settings()
    stage_latency_ms: dict[str, float] = {}

    async def run_vector_search() -> list[tuple[Chunk, str, float]]:
        nonlocal query_vector
        if query_vector is None:
            query_vector = await embed_query(data.query)
        vector_started = time.perf_counter()
        try:
            return await vector_search(
                session,
                who.tenant_id,
                data.project_id,
                data.collections,
                data.filters,
                query_vector,
                settings.embedding_model,
                data.vector_top_k,
            )
        finally:
            duration = time.perf_counter() - vector_started
            VECTOR_SEARCH_DURATION.observe(duration)
            stage_latency_ms["dense"] = round(duration * 1000, 2)

    async def run_lexical_search() -> list[tuple[uuid.UUID, float]]:
        lexical_started = time.perf_counter()
        try:
            return await bm25_search(
                who.tenant_id,
                data.project_id,
                data.collections,
                data.query,
                data.filters,
                data.bm25_top_k,
            )
        finally:
            duration = time.perf_counter() - lexical_started
            BM25_SEARCH_DURATION.observe(duration)
            stage_latency_ms["lexical"] = round(duration * 1000, 2)

    vector_hits: list[tuple[Chunk, str, float]] = []
    if data.mode in {"dense", "hybrid"}:
        vector_hits = await run_vector_search()
    vector_rows = [(chunk, external_id) for chunk, external_id, _ in vector_hits]
    vector_ids = [chunk.id for chunk, _, _ in vector_hits]
    vector_scores = {chunk.id: score for chunk, _, score in vector_hits}

    opensearch_degraded = False
    bm25_scores: dict[uuid.UUID, float] = {}
    lexical_hits: list[tuple[uuid.UUID, float]] = []
    effective_mode: str = data.mode
    if data.mode in {"lexical", "hybrid"}:
        try:
            lexical_hits = await run_lexical_search()
        except OpenSearchUnavailable:
            opensearch_degraded = True
            if data.mode == "lexical":
                vector_hits = await run_vector_search()
                vector_rows = [(chunk, external_id) for chunk, external_id, _ in vector_hits]
                vector_ids = [chunk.id for chunk, _, _ in vector_hits]
                vector_scores = {chunk.id: score for chunk, _, score in vector_hits}
                effective_mode = "dense_fallback"
    lexical_ids = [chunk_id for chunk_id, _ in lexical_hits]
    bm25_scores = dict(lexical_hits)

    rows_by_id = {row[0].id: row for row in vector_rows}
    missing_ids = [chunk_id for chunk_id in lexical_ids if chunk_id not in rows_by_id]
    if missing_ids:
        lexical_rows = (await session.execute(statement.where(Chunk.id.in_(missing_ids)))).all()
        rows_by_id.update({row[0].id: row for row in lexical_rows})
    lexical_ids = [chunk_id for chunk_id in lexical_ids if chunk_id in rows_by_id]

    duplicate_count = len(vector_ids) + len(lexical_ids) - len(set(vector_ids) | set(lexical_ids))
    if duplicate_count:
        DUPLICATE_CHUNKS_REMOVED.inc(duplicate_count)
    fusion_started = time.perf_counter()
    try:
        rankings = [ranking for ranking in (vector_ids, lexical_ids) if ranking]
        fusion_scores = reciprocal_rank_fusion(rankings)
    finally:
        duration = time.perf_counter() - fusion_started
        FUSION_DURATION.observe(duration)
        stage_latency_ms["fusion"] = round(duration * 1000, 2)
    ranked_ids = sorted(
        fusion_scores,
        key=fusion_scores.__getitem__,
        reverse=True,
    )[: data.fusion_top_k]
    ranked = [rows_by_id[chunk_id] for chunk_id in ranked_ids]

    reranker_degraded = False
    reranker_used = False
    reranker_scores: dict[uuid.UUID, float] = {}
    reranker_metadata: dict[str, Any] = {}
    reranker_input_ids = [str(row[0].id) for row in ranked]
    if data.use_reranker and settings.reranker_enabled and ranked:
        reranker_started = time.perf_counter()
        try:
            RERANKER_REQUESTS.inc()
            reranker_rows_by_id = {str(row[0].id): row for row in ranked}
            async with RerankerClient.from_settings(settings) as reranker:
                response = await reranker.rerank(
                    query=data.query,
                    documents=[
                        RerankerDocument(
                            id=str(row[0].id),
                            text=row[0].content,
                            metadata={
                                "document_id": str(row[0].document_id),
                                "source_type": row[0].source_type,
                                "source_id": row[0].source_id,
                            },
                        )
                        for row in ranked
                    ],
                    top_n=data.rerank_top_k,
                    request_id=request_id,
                    correlation_id=correlation_id,
                )
            ranked = [reranker_rows_by_id[item.id] for item in response.results]
            reranker_scores = {
                uuid.UUID(item.id): item.ranking_score for item in response.results
            }
            reranker_metadata = {
                "model": response.model,
                "model_revision": response.model_revision,
                "device": response.device,
                "usage": response.usage,
            }
            reranker_used = True
        except RerankerUnavailable:
            reranker_degraded = True
            RERANKER_ERRORS.inc()
            RERANKER_DEGRADED.inc()
        finally:
            RERANKER_DURATION.observe(time.perf_counter() - reranker_started)

    results = []
    for final_rank, (chunk, external_id) in enumerate(
        ranked[: data.rerank_top_k],
        start=1,
    ):
        retrieval_sources = []
        if chunk.id in vector_scores:
            retrieval_sources.append("dense")
        if chunk.id in bm25_scores:
            retrieval_sources.append("lexical")
        results.append(
            {
                "document_id": str(chunk.document_id),
                "external_document_id": external_id,
                "chunk_id": str(chunk.id),
                "content": chunk.content,
                "metadata": chunk.metadata_,
                "source_type": chunk.source_type,
                "source_id": chunk.source_id,
                "section_title": chunk.section_title,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "index_version": chunk.index_version,
                "vector_score": vector_scores.get(chunk.id),
                "bm25_score": bm25_scores.get(chunk.id),
                "fusion_score": fusion_scores[chunk.id],
                "reranker_score": reranker_scores.get(chunk.id),
                "final_score": reranker_scores.get(chunk.id, fusion_scores[chunk.id]),
                "retrieval_sources": retrieval_sources,
                "final_rank": final_rank,
            }
        )
    trace = {
        "original_query": data.query,
        "normalized_query": data.query.strip(),
        "filters": data.filters,
        "embedding_model": settings.embedding_model,
        "requested_mode": data.mode,
        "effective_mode": effective_mode,
        "retrieval_strategy": f"{effective_mode}_rrf",
        "opensearch_degraded": opensearch_degraded,
        "reranker_used": reranker_used,
        "reranker_degraded": reranker_degraded,
        "reranker": reranker_metadata,
        "dense_candidates": [
            {"chunk_id": str(chunk_id), "rank": rank, "score": vector_scores[chunk_id]}
            for rank, chunk_id in enumerate(vector_ids, start=1)
        ],
        "lexical_candidates": [
            {"chunk_id": str(chunk_id), "rank": rank, "score": bm25_scores[chunk_id]}
            for rank, chunk_id in enumerate(lexical_ids, start=1)
        ],
        "fusion_candidates": [
            {
                "chunk_id": str(chunk_id),
                "document_id": str(rows_by_id[chunk_id][0].document_id),
                "rank": rank,
                "score": fusion_scores[chunk_id],
            }
            for rank, chunk_id in enumerate(ranked_ids, start=1)
        ],
        "reranker_candidates": reranker_input_ids,
        "final_chunks": [item["chunk_id"] for item in results],
        "stage_latency_ms": stage_latency_ms,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    RETRIEVAL_RESULTS_COUNT.observe(len(results))
    if not results:
        RETRIEVAL_EMPTY.inc()
    session.add(
        RetrievalRequest(
            id=request_id,
            tenant_id=who.tenant_id,
            project_id=data.project_id,
            payload={
                "query": data.query,
                "collections": data.collections,
                "filters": data.filters,
                "configuration": {
                    "mode": data.mode,
                    "vector_top_k": data.vector_top_k,
                    "bm25_top_k": data.bm25_top_k,
                    "fusion_top_k": data.fusion_top_k,
                    "rerank_top_k": data.rerank_top_k,
                    "use_reranker": data.use_reranker,
                    "embedding_model": settings.embedding_model,
                },
                "results": results,
                "trace": trace,
            },
        )
    )
    await session.commit()
    return request_id, results, trace
