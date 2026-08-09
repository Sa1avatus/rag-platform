from typing import Any

from rag_platform.core.config import get_settings
from rag_platform.services.readiness import readiness_status


async def embedding_profile() -> dict[str, Any]:
    ready, components = await readiness_status()
    model = components.get("embedding_model", {})
    settings = get_settings()
    actual_dimension = model.get("dimension")
    compatible = (
        isinstance(actual_dimension, int)
        and actual_dimension == settings.embedding_dimension
        and model.get("model") == settings.embedding_model
    )
    return {
        "status": "ready" if ready and compatible else model.get("status", "not_ready"),
        "model": model.get("model", settings.embedding_model),
        "device": model.get("device"),
        "dimension": actual_dimension,
        "expected_dimension": settings.embedding_dimension,
        "compatible": compatible,
    }
