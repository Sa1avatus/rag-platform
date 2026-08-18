"""Tests for embedding model registry, padding, model isolation, and heartbeat."""


import pytest

from rag_platform.core.embedding_registry import (
    MAX_VECTOR_DIMENSION,
    get_model_by_id,
    registry,
)

# ── padding tests ───────────────────────────────────────────────────────

class TestPadVector:
    """Test the zero-padding adapter for pgvector storage."""

    def test_bge_no_padding_needed(self) -> None:
        """BGE-M3 1024-dim → storage 1024 = no-op."""
        cfg = registry["bge-m3"]
        vec = [0.1] * 1024
        padded = cfg.pad_vector(vec)
        assert len(padded) == 1024
        assert padded == vec

    def test_e5_padded_to_1024(self) -> None:
        """E5-small 384-dim → padded to 1024 with zeros."""
        cfg = registry["multilingual-e5-small"]
        vec = [0.5] * 384
        padded = cfg.pad_vector(vec)
        assert len(padded) == MAX_VECTOR_DIMENSION
        assert padded[:384] == vec
        assert all(v == 0.0 for v in padded[384:])

    def test_padding_preserves_cosine_similarity(self) -> None:
        """Two 384-dim vectors padded to 1024 have same cosine as originals."""
        import math

        cfg = registry["multilingual-e5-small"]
        a = [1.0] * 384
        b = [2.0] * 384
        # Cosine of originals (identical directions = 1.0)
        dot_orig = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        cos_orig = dot_orig / (norm_a * norm_b)
        # Cosine of padded
        a_pad = cfg.pad_vector(a)
        b_pad = cfg.pad_vector(b)
        dot_pad = sum(x * y for x, y in zip(a_pad, b_pad, strict=True))
        norm_ap = math.sqrt(sum(x * x for x in a_pad))
        norm_bp = math.sqrt(sum(x * x for x in b_pad))
        cos_pad = dot_pad / (norm_ap * norm_bp)
        assert abs(cos_orig - cos_pad) < 1e-10

    def test_invalid_down_padding_raises(self) -> None:
        """A vector larger than storage dimension raises ValueError."""
        cfg = registry["multilingual-e5-small"]
        oversized = [0.1] * 2048
        with pytest.raises(ValueError, match="exceeds storage dimension"):
            cfg.pad_vector(oversized)

    def test_storage_dimension_property(self) -> None:
        """Both models report the same storage dimension."""
        assert registry["bge-m3"].storage_dimension == 1024
        assert registry["multilingual-e5-small"].storage_dimension == 1024

    def test_model_dimension_not_storage(self) -> None:
        """Model dimension is the REAL embedding size, not the storage size."""
        assert registry["bge-m3"].dimension == 1024
        assert registry["multilingual-e5-small"].dimension == 384


# ── registry tests ──────────────────────────────────────────────────────

class TestModelRegistry:
    def test_bge_m3_config(self) -> None:
        cfg = get_model_by_id("bge-m3")
        assert cfg.model_name == "BAAI/bge-m3"
        assert cfg.dimension == 1024
        assert cfg.max_input_tokens == 8192
        assert cfg.query_prefix == ""
        assert cfg.passage_prefix == ""
        assert cfg.index_version == "rag-chunks-bge-m3-v1"

    def test_e5_small_config(self) -> None:
        cfg = get_model_by_id("multilingual-e5-small")
        assert cfg.model_name == "intfloat/multilingual-e5-small"
        assert cfg.dimension == 384
        assert cfg.max_input_tokens == 512
        assert cfg.query_prefix == "query: "
        assert cfg.passage_prefix == "passage: "
        assert cfg.index_version == "rag-chunks-multilingual-e5-small-v1"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown embedding model"):
            get_model_by_id("nonexistent-model")

    def test_index_versions_differ(self) -> None:
        """Each model has its own index version — no collisions."""
        bge = get_model_by_id("bge-m3")
        e5 = get_model_by_id("multilingual-e5-small")
        assert bge.index_version != e5.index_version

    def test_bge_e5_dimension_mismatch(self) -> None:
        """BGE 1024 and E5 384 must not be confused."""
        bge = get_model_by_id("bge-m3")
        e5 = get_model_by_id("multilingual-e5-small")
        assert bge.dimension != e5.dimension


# ── model isolation tests ───────────────────────────────────────────────

class TestModelIsolation:
    def test_unique_constraint_allows_both_models(self) -> None:
        """ChunkEmbedding unique (chunk_id, model, revision) allows both."""
        # This is a schema test — verify the constraint name exists.
        from rag_platform.db.models import ChunkEmbedding

        constraint_names = [
            c.name for c in ChunkEmbedding.__table__.constraints
        ]
        assert "uq_chunk_embeddings_identity" in constraint_names

    def test_cache_key_differs_by_model(self) -> None:
        """Cache keys for the same query text differ between models."""
        from rag_platform.core.config import Settings

        # Create settings with bge-m3
        s_bge = Settings(
            active_embedding_model="bge-m3",
            admin_token="test-token-12345678",
            api_key_pepper="test-pepper-123456",
        )
        # Create settings with e5
        s_e5 = Settings(
            active_embedding_model="multilingual-e5-small",
            admin_token="test-token-12345678",
            api_key_pepper="test-pepper-123456",
        )
        from rag_platform.services.cache import query_embedding_cache_key

        key_bge = query_embedding_cache_key("test query", s_bge, model_id="bge-m3")
        key_e5 = query_embedding_cache_key(
            "test query", s_e5, model_id="multilingual-e5-small"
        )
        assert key_bge != key_e5


# ── heartbeat tests (import guard — only test structure) ────────────────

class TestHeartbeatStructure:
    def test_heartbeat_loop_exists(self) -> None:
        """The heartbeat loop function exists and is importable."""
        from rag_platform.worker.embeddings import _heartbeat_loop

        assert callable(_heartbeat_loop)

    def test_model_ready_key_format(self) -> None:
        """MODEL_READY_KEY is a non-empty string."""
        from rag_platform.services.readiness import MODEL_READY_KEY

        assert isinstance(MODEL_READY_KEY, str)
        assert len(MODEL_READY_KEY) > 0
