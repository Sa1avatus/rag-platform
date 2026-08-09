from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import RuntimeSetting

SETTING_DEFINITIONS: dict[str, dict[str, Any]] = {
    "default_vector_top_k": {
        "default": 30,
        "minimum": 1,
        "maximum": 200,
        "description": "Default vector candidates per search.",
    },
    "default_bm25_top_k": {
        "default": 30,
        "minimum": 1,
        "maximum": 200,
        "description": "Default lexical candidates per search.",
    },
    "default_fusion_top_k": {
        "default": 20,
        "minimum": 1,
        "maximum": 100,
        "description": "Default fused candidates retained.",
    },
    "default_rerank_top_k": {
        "default": 5,
        "minimum": 1,
        "maximum": 50,
        "description": "Default results retained after reranking.",
    },
    "reranker_enabled": {"default": True, "description": "Enable reranking by default."},
    "query_normalization_enabled": {
        "default": True,
        "description": "Normalize retrieval queries by default.",
    },
    "query_expansion_enabled": {
        "default": False,
        "description": "Enable query expansion by default.",
    },
    "parent_content_enabled": {
        "default": True,
        "description": "Include parent content by default.",
    },
    "indexing_concurrency": {
        "default": 2,
        "minimum": 1,
        "maximum": 64,
        "restart": True,
        "description": "Indexing worker concurrency.",
    },
    "embedding_batch_size": {
        "default": 16,
        "minimum": 1,
        "maximum": 256,
        "restart": True,
        "description": "Embedding worker batch size.",
    },
    "document_max_bytes": {
        "default": 26_214_400,
        "minimum": 1_048_576,
        "maximum": 104_857_600,
        "restart": True,
        "description": "Maximum accepted document size in bytes.",
    },
    "trace_retention_days": {
        "default": 30,
        "minimum": 1,
        "maximum": 3650,
        "description": "Retrieval trace retention in days.",
    },
    "completed_job_retention_days": {
        "default": 30,
        "minimum": 1,
        "maximum": 3650,
        "description": "Completed indexing job retention in days.",
    },
}


async def read_runtime_settings(session: AsyncSession) -> dict[str, Any]:
    rows = (await session.scalars(select(RuntimeSetting))).all()
    saved = {row.key: row.value for row in rows}
    return {
        key: saved.get(key, definition["default"])
        for key, definition in SETTING_DEFINITIONS.items()
    }


async def runtime_settings_response(session: AsyncSession) -> dict[str, object]:
    values = await read_runtime_settings(session)
    return {
        "settings": [
            {
                "key": key,
                "value": values[key],
                "default": definition["default"],
                "description": definition["description"],
                "minimum": definition.get("minimum"),
                "maximum": definition.get("maximum"),
                "restart_required": definition.get("restart", False),
                "reindex_required": definition.get("reindex", False),
            }
            for key, definition in SETTING_DEFINITIONS.items()
        ]
    }


async def apply_runtime_settings(session: AsyncSession, updates: dict[str, Any]) -> None:
    rows = (await session.scalars(select(RuntimeSetting))).all()
    existing = {row.key: row for row in rows}
    for key, value in updates.items():
        row = existing.get(key)
        if row is None:
            session.add(RuntimeSetting(key=key, value=value))
        else:
            row.value = value
