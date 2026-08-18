from typing import Any

from rag_platform.core.config import get_settings
from rag_platform.core.embedding_registry import get_active_model, registry
from rag_platform.services.readiness import readiness_status


async def embedding_profile() -> dict[str, Any]:
    ready, components = await readiness_status()
    model = components.get("embedding_model", {})
    settings = get_settings()
    cfg = get_active_model()
    actual_dimension = model.get("dimension")
    compatible = (
        isinstance(actual_dimension, int)
        and actual_dimension == cfg.dimension
        and model.get("model") == cfg.model_name
    )
    return {
        "status": "ready" if ready and compatible else model.get("status", "not_ready"),
        "model_id": cfg.id,
        "model": cfg.model_name,
        "backend": settings.embedding_backend,
        "revision": settings.embedding_revision,
        "normalization": cfg.normalization,
        "device": model.get("device"),
        "dimension": actual_dimension,
        "expected_dimension": cfg.dimension,
        "max_input_tokens": cfg.max_input_tokens,
        "index_version": cfg.index_version,
        "chunker_version": settings.chunker_version,
        "compatible": compatible,
    }


async def all_models_status() -> list[dict[str, Any]]:
    """Return status for every registered embedding model."""
    active_id = get_active_model().id
    result = []
    for model_id, cfg in registry.items():
        is_active = model_id == active_id
        result.append({
            "model_id": cfg.id,
            "display_name": cfg.display_name,
            "model_name": cfg.model_name,
            "dimension": cfg.dimension,
            "max_input_tokens": cfg.max_input_tokens,
            "device": cfg.device,
            "model_type": cfg.model_type,
            "multilingual": cfg.multilingual,
            "query_prefix": cfg.query_prefix,
            "passage_prefix": cfg.passage_prefix,
            "normalization": cfg.normalization,
            "similarity_metric": cfg.similarity_metric,
            "index_version": cfg.index_version,
            "enabled": cfg.enabled,
            "is_active": is_active,
        })
    return result
