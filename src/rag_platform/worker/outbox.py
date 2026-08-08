from datetime import UTC, datetime
from typing import Any

from celery import current_app
from sqlalchemy import select

from rag_platform.db.models import OutboxEvent
from rag_platform.db.session import Session


def index_message(payload: dict[str, Any]) -> tuple[str, list[str]]:
    if payload.get("type") != "document.index":
        raise ValueError("unsupported outbox event type")
    version_id = payload.get("version_id")
    job_id = payload.get("job_id")
    if not isinstance(version_id, str) or not isinstance(job_id, str):
        raise ValueError("outbox event is missing identifiers")
    return "rag_platform.worker.tasks.index_document", [version_id, job_id]


async def publish_pending(limit: int = 100) -> int:
    published = 0
    async with Session() as session:
        rows = (
            await session.scalars(
                select(OutboxEvent)
                .where(OutboxEvent.payload["published_at"].astext.is_(None))
                .order_by(OutboxEvent.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        ).all()
        for row in rows:
            task_name, args = index_message(row.payload)
            current_app.send_task(task_name, args=args, task_id=str(row.id))
            row.payload = {
                **row.payload,
                "attempts": int(row.payload.get("attempts", 0)) + 1,
                "published_at": datetime.now(UTC).isoformat(),
            }
            published += 1
        await session.commit()
    return published
