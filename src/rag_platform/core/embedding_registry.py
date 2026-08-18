"""Embedding model registry.

Central configuration for all supported embedding models.  Every model
parameter (dimension, max tokens, preprocessing, index name) lives here —
no global constants scattered across the codebase.

Usage::

    from rag_platform.core.embedding_registry import get_active_model, registry

    cfg = get_active_model()           # from ACTIVE_EMBEDDING_MODEL env
    cfg = registry["bge-m3"]           # by id
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "EmbeddingModelConfig",
    "registry",
    "get_active_model",
    "get_model_by_id",
    "MAX_VECTOR_DIMENSION",
]

# Largest embedding dimension across all registered models.
# pgvector column is Vector(MAX_VECTOR_DIMENSION); shorter vectors are
# zero-padded before storage / query.
MAX_VECTOR_DIMENSION = 1024


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Immutable configuration for a single embedding model."""

    # ── identity ────────────────────────────────────────────────────────
    id: str  # short stable id, e.g. "bge-m3"
    display_name: str  # human-readable, e.g. "BAAI/bge-m3"
    model_name: str  # HuggingFace repo, e.g. "BAAI/bge-m3"
    onnx_subfolder: str = ""  # subfolder in HF repo containing ONNX files

    # ── dimensions ──────────────────────────────────────────────────────
    dimension: int = 1024  # actual embedding dimension
    max_input_tokens: int = 8192  # model's context window in tokens

    # ── runtime ─────────────────────────────────────────────────────────
    device: str = "auto"  # "auto", "cpu", "cuda"
    model_type: str = "dense"  # "dense" (future: "sparse", "colbert")
    multilingual: bool = True

    # ── preprocessing ───────────────────────────────────────────────────
    query_prefix: str = ""  # prepended to queries (E5 needs "query: ")
    passage_prefix: str = ""  # prepended to passages (E5 needs "passage: ")
    normalization: str = "l2"  # "l2" or "none"
    similarity_metric: str = "cosine"

    # ── index ───────────────────────────────────────────────────────────
    index_version: str = ""  # auto-generated if empty

    # ── feature flags ───────────────────────────────────────────────────
    enabled: bool = True

    # ── derived (set in __post_init__) ──────────────────────────────────
    _padding: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Auto-generate index version from model id + dimension.
        if not self.index_version:
            object.__setattr__(
                self, "index_version", f"rag-chunks-{self.id}-v1"
            )
        # How many zeros to append so the vector fills MAX_VECTOR_DIMENSION.
        object.__setattr__(
            self, "_padding", MAX_VECTOR_DIMENSION - self.dimension
        )

    # ── vector helpers ──────────────────────────────────────────────────
    @property
    def storage_dimension(self) -> int:
        """Physical vector size used in pgvector (MAX_VECTOR_DIMENSION)."""
        from rag_platform.core.embedding_registry import MAX_VECTOR_DIMENSION

        return MAX_VECTOR_DIMENSION

    def pad_vector(self, vector: list[float]) -> list[float]:
        """Zero-pad *vector* to ``MAX_VECTOR_DIMENSION``.

        Raises ``ValueError`` if the vector is already larger than the
        storage dimension (would require truncation which is never safe).
        """
        if len(vector) > self.storage_dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} exceeds storage dimension "
                f"{self.storage_dimension} — cannot pad"
            )
        diff = self.storage_dimension - len(vector)
        if diff > 0:
            return vector + [0.0] * diff
        return vector

    @property
    def needs_query_prefix(self) -> bool:
        return bool(self.query_prefix)

    @property
    def needs_passage_prefix(self) -> bool:
        return bool(self.passage_prefix)


# ── Model registry ──────────────────────────────────────────────────────

registry: dict[str, EmbeddingModelConfig] = {
    "bge-m3": EmbeddingModelConfig(
        id="bge-m3",
        display_name="BAAI/bge-m3",
        model_name="BAAI/bge-m3",
        onnx_subfolder="onnx",
        dimension=1024,
        max_input_tokens=8192,
        device="auto",
        model_type="dense",
        multilingual=True,
        query_prefix="",
        passage_prefix="",
        normalization="l2",
        similarity_metric="cosine",
        enabled=True,
    ),
    "multilingual-e5-small": EmbeddingModelConfig(
        id="multilingual-e5-small",
        display_name="intfloat/multilingual-e5-small",
        model_name="intfloat/multilingual-e5-small",
        onnx_subfolder="onnx",
        dimension=384,
        max_input_tokens=512,
        device="auto",
        model_type="dense",
        multilingual=True,
        query_prefix="query: ",
        passage_prefix="passage: ",
        normalization="l2",
        similarity_metric="cosine",
        enabled=True,
    ),
}


def get_model_by_id(model_id: str) -> EmbeddingModelConfig:
    """Return config for *model_id* or raise ``KeyError``."""
    try:
        return registry[model_id]
    except KeyError:
        available = ", ".join(sorted(registry))
        raise KeyError(
            f"Unknown embedding model {model_id!r}. "
            f"Available: {available}"
        ) from None


def get_active_model() -> EmbeddingModelConfig:
    """Return the model config for the current ``ACTIVE_EMBEDDING_MODEL``.

    Checks Redis first (for runtime switching), falls back to env var.
    No lru_cache — Redis read is fast and we need fresh values after switching.
    """
    model_id = _read_active_model_from_redis()
    if model_id:
        return get_model_by_id(model_id)
    from rag_platform.core.config import get_settings

    return get_model_by_id(get_settings().active_embedding_model)


def _read_active_model_from_redis() -> str | None:
    """Read the active model id from Redis (non-blocking, best-effort)."""
    try:
        import json as _json

        from redis import Redis as SyncRedis

        from rag_platform.core.config import get_settings

        cache = SyncRedis.from_url(get_settings().redis_url, decode_responses=True)
        try:
            raw = cache.get("rag:active_embedding_model")
            if raw:
                data = _json.loads(raw)
                return data.get("model_id")
        finally:
            cache.close()
    except Exception:
        pass
    return None


def set_active_model_in_redis(model_id: str, cfg: EmbeddingModelConfig) -> None:
    """Write the active model to Redis so all containers pick it up."""
    import json as _json

    from redis import Redis as SyncRedis

    from rag_platform.core.config import get_settings

    cache = SyncRedis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        cache.set(
            "rag:active_embedding_model",
            _json.dumps({"model_id": model_id}),
            ex=86400 * 30,  # 30 days
        )
    finally:
        cache.close()
