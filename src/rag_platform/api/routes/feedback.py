from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import FeedbackCreate
from rag_platform.core.auth import Principal, principal
from rag_platform.db.models import (
    Chunk,
    RetrievalFeedback,
    RetrievalRequest,
)
from rag_platform.db.session import get_session

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


@router.post("", status_code=201)
async def create_feedback(
    data: FeedbackCreate,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    request = await session.scalar(
        select(RetrievalRequest).where(
            RetrievalRequest.id == data.request_id,
            RetrievalRequest.tenant_id == who.tenant_id,
            RetrievalRequest.project_id == data.project_id,
        )
    )
    chunk = await session.scalar(
        select(Chunk).where(
            Chunk.id == data.chunk_id,
            Chunk.tenant_id == who.tenant_id,
            Chunk.project_id == data.project_id,
        )
    )
    if request is None or chunk is None:
        raise HTTPException(404, "retrieval request or chunk not found")
    who.authorize(data.project_id, [chunk.collection], "feedback:write")
    result_ids = {
        result.get("chunk_id")
        for result in request.payload.get("results", [])
        if isinstance(result, dict)
    }
    if str(chunk.id) not in result_ids:
        raise HTTPException(409, "chunk was not returned by this retrieval request")
    feedback = RetrievalFeedback(
        tenant_id=who.tenant_id,
        project_id=data.project_id,
        payload={
            "request_id": str(data.request_id),
            "chunk_id": str(data.chunk_id),
            "collection": chunk.collection,
            "relevant": data.relevant,
            "relevance_grade": data.relevance_grade,
            "comment": data.comment,
            "retrieval_configuration": request.payload.get("configuration", {}),
        },
    )
    session.add(feedback)
    await session.commit()
    return {"id": feedback.id, **feedback.payload}
