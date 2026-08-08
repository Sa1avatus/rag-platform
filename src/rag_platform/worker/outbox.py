from datetime import UTC, datetime

from celery import current_app
from sqlalchemy import select

from rag_platform.db.models import OutboxEvent
from rag_platform.db.session import Session
from rag_platform.services.outbox_contract import event_message


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
            task_name, args = event_message(row.payload)
            current_app.send_task(task_name, args=args, task_id=str(row.id))
            row.payload = {
                **row.payload,
                "attempts": int(row.payload.get("attempts", 0)) + 1,
                "published_at": datetime.now(UTC).isoformat(),
            }
            published += 1
        await session.commit()
    return published
