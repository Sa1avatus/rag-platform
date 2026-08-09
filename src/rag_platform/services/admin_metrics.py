import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import (
    AuditLog,
    Document,
    IndexingJob,
    RetrievalFeedback,
    RetrievalRequest,
)

METRIC_MODELS = {
    "documents": Document,
    "retrieval_requests": RetrievalRequest,
    "feedback": RetrievalFeedback,
    "audit_events": AuditLog,
    "indexing_jobs": IndexingJob,
    "indexing_errors": IndexingJob,
}


def metric_statement(
    metric: str,
    *,
    project_id: uuid.UUID | None,
    collection: str | None,
    start: datetime,
    end: datetime,
    step: str,
) -> Select[tuple[datetime, int]]:
    model: Any = METRIC_MODELS[metric]
    bucket = func.date_trunc(step, model.created_at).label("bucket")
    statement = (
        select(bucket, func.count().label("value"))
        .where(model.created_at >= start, model.created_at <= end)
        .group_by(bucket)
        .order_by(bucket)
    )
    if project_id is not None:
        statement = statement.where(model.project_id == project_id)
    if collection is not None:
        if model is Document:
            statement = statement.where(Document.collection == collection)
        else:
            statement = statement.where(model.payload["collections"].contains([collection]))
    if metric == "indexing_errors":
        statement = statement.where(model.payload["status"].astext.in_(["failed", "dead_letter"]))
    return statement


async def metric_timeseries(
    session: AsyncSession,
    metric: str,
    *,
    project_id: uuid.UUID | None,
    collection: str | None,
    start: datetime,
    end: datetime,
    step: str,
) -> list[dict[str, object]]:
    rows = (
        await session.execute(
            metric_statement(
                metric,
                project_id=project_id,
                collection=collection,
                start=start,
                end=end,
                step=step,
            )
        )
    ).all()
    return [{"timestamp": row.bucket, "value": row.value} for row in rows]
