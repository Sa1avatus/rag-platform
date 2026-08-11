import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import ContextResponse, SearchRequest, SearchResult
from rag_platform.core.auth import Principal, principal
from rag_platform.db.models import RetrievalRequest
from rag_platform.db.session import get_session
from rag_platform.services.context_selection import select_context
from rag_platform.services.retrieval import search

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResult)
async def search_route(
    data: SearchRequest,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> SearchResult:
    request_id, results, trace = await search(session, who, data)
    return SearchResult(
        request_id=request_id,
        results=results,
        trace=trace if data.include_trace else None,
    )


@router.post("/context", response_model=ContextResponse)
async def context(
    data: SearchRequest,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> ContextResponse:
    request_id, results, _ = await search(session, who, data)
    selection = select_context(
        results,
        max_chunks=data.max_context_chunks,
        max_estimated_tokens=data.max_context_tokens,
        per_document_limit=data.per_document_limit,
    )
    trace_row = await session.get(RetrievalRequest, request_id)
    if trace_row is not None:
        trace_row.payload = {
            **trace_row.payload,
            "selected_context": [item["chunk_id"] for item in selection.items],
            "context_selection": {
                "estimated_tokens": selection.estimated_tokens,
                "duplicates_suppressed": selection.duplicates_suppressed,
                "document_limit_suppressed": selection.document_limit_suppressed,
                "token_budget_suppressed": selection.token_budget_suppressed,
            },
        }
        await session.commit()
    return ContextResponse(
        request_id=request_id,
        sources=selection.items,
        context="\n\n".join(str(item["content"]) for item in selection.items),
    )


@router.get("/requests/{request_id}")
async def request(
    request_id: uuid.UUID,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.scalar(
        select(RetrievalRequest).where(
            RetrievalRequest.id == request_id,
            RetrievalRequest.tenant_id == who.tenant_id,
        )
    )
    if row is None:
        raise HTTPException(404, "request not found")
    return row.payload
