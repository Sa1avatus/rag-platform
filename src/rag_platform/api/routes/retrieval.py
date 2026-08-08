import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import ContextResponse, SearchRequest, SearchResult
from rag_platform.core.auth import Principal, principal
from rag_platform.db.models import RetrievalRequest
from rag_platform.db.session import get_session
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
    return ContextResponse(
        request_id=request_id,
        sources=results,
        context="\n\n".join(str(item["content"]) for item in results),
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
