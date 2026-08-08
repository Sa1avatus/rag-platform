import json
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import text

from rag_platform.core.config import get_settings
from rag_platform.db.session import Session

MODEL_READY_KEY = "rag:worker:model_ready"


async def readiness_status() -> tuple[bool, dict[str, Any]]:
    components: dict[str, Any] = {}
    database_ready = False
    redis_ready = False
    model_ready = False
    try:
        async with Session() as session:
            await session.execute(text("SELECT 1"))
        database_ready = True
    except Exception as exc:
        components["postgresql"] = {"status": "down", "error": type(exc).__name__}
    else:
        components["postgresql"] = {"status": "up"}

    cache = Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        heartbeat = await cache.get(MODEL_READY_KEY)
        redis_ready = True
        if heartbeat:
            details = json.loads(heartbeat)
            model_ready = details.get("dimension") == get_settings().embedding_dimension
            components["embedding_model"] = {
                "status": "ready" if model_ready else "incompatible",
                **details,
            }
        else:
            components["embedding_model"] = {"status": "not_ready"}
    except Exception as exc:
        components["redis"] = {"status": "down", "error": type(exc).__name__}
        components["embedding_model"] = {"status": "unknown"}
    else:
        components["redis"] = {"status": "up"}
    finally:
        await cache.aclose()
    return database_ready and redis_ready and model_ready, components
