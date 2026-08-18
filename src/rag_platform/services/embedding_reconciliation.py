"""Multi-model embedding reconciliation.

Finds chunks missing embeddings for any registered model and fills them
incrementally.  Designed to run as a periodic Celery beat task so that
newly-ingested chunks (which only get the *active* model's embedding
during ``index_version``) eventually receive embeddings for ALL models.

Non-destructive: never recalculates existing embeddings.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
import onnxruntime as ort
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rag_platform.core.embedding_registry import (
    EmbeddingModelConfig,
    registry,
)
from rag_platform.db.models import Chunk, ChunkEmbedding
from rag_platform.db.session import Session
from rag_platform.worker.embeddings import _load_model

log = logging.getLogger(__name__)

# How many chunks to embed per reconciliation tick (per model).
_BATCH_SIZE = 32
# Maximum total chunks to process per tick across all models.
_MAX_TOTAL = 200


async def find_missing_embeddings(
    model_name: str,
    limit: int = _BATCH_SIZE,
) -> list[tuple[uuid.UUID, str]]:
    """Return ``(chunk_id, content)`` for chunks lacking *model_name*."""
    async with Session() as session:
        rows = (
            await session.execute(
                select(Chunk.id, Chunk.content)
                .outerjoin(
                    ChunkEmbedding,
                    (ChunkEmbedding.chunk_id == Chunk.id)
                    & (ChunkEmbedding.model == model_name),
                )
                .where(ChunkEmbedding.id.is_(None))
                .order_by(Chunk.id)
                .limit(limit)
            )
        ).all()
    return [(row[0], row[1]) for row in rows]


async def insert_embeddings(
    chunk_ids: list[uuid.UUID],
    vectors: list[list[float]],
    cfg: EmbeddingModelConfig,
) -> int:
    """Bulk-insert embeddings.  Returns count inserted."""
    if not chunk_ids:
        return 0
    async with Session() as session:
        for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
            await session.execute(
                pg_insert(ChunkEmbedding)
                .values(
                    id=uuid.uuid4(),
                    chunk_id=chunk_id,
                    model=cfg.model_name,
                    model_revision="default",
                    backend="onnxruntime",
                    normalization=cfg.normalization,
                    embedding_dimension=cfg.dimension,
                    embedding=vector,
                )
                .on_conflict_do_nothing(
                    constraint="uq_chunk_embeddings_identity",
                )
            )
        await session.commit()
    return len(chunk_ids)


def embed_texts(
    texts: list[str],
    cfg: EmbeddingModelConfig,
    session: ort.InferenceSession,
    tokenizer: object,
) -> list[list[float]]:
    """Embed *texts* using the given model components."""
    if cfg.passage_prefix:
        texts = [cfg.passage_prefix + t for t in texts]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg.max_input_tokens,
        return_tensors="np",
    )
    valid = {inp.name for inp in session.get_inputs()}
    ort_inputs = {k: v for k, v in inputs.items() if k in valid}
    if "token_type_ids" in valid and "token_type_ids" not in ort_inputs:
        ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
    outputs = session.run(None, ort_inputs)
    token_embeddings = outputs[0]
    attention_mask = inputs["attention_mask"]
    mask = np.expand_dims(attention_mask, axis=-1)
    summed = np.sum(token_embeddings * mask, axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    embeddings = pooled / norms
    raw = embeddings.tolist()
    return [cfg.pad_vector(v) for v in raw]


async def reconcile_model(cfg: EmbeddingModelConfig) -> int:
    """Fill missing embeddings for one model.  Returns count inserted."""
    missing = await find_missing_embeddings(cfg.model_name, limit=_BATCH_SIZE)
    if not missing:
        return 0
    chunk_ids = [cid for cid, _ in missing]
    texts = [text for _, text in missing]
    log.info(
        "Reconciling %s: %d missing chunks", cfg.model_name, len(chunk_ids),
    )
    session, tokenizer, _device = _load_model(cfg)
    vectors = embed_texts(texts, cfg, session, tokenizer)
    inserted = await insert_embeddings(chunk_ids, vectors, cfg)
    log.info("Reconciled %s: %d embeddings inserted", cfg.model_name, inserted)
    return inserted


async def reconcile_all_models() -> dict[str, int]:
    """Fill missing embeddings for every registered model.

    Returns ``{model_name: inserted_count}``.
    """
    results: dict[str, int] = {}
    total = 0
    for _model_id, cfg in registry.items():
        if not cfg.enabled:
            continue
        if total >= _MAX_TOTAL:
            break
        inserted = await reconcile_model(cfg)
        results[cfg.model_name] = inserted
        total += inserted
    return results
