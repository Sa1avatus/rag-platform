import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from celery.signals import worker_process_init
from huggingface_hub import snapshot_download
from redis import Redis
from transformers import AutoTokenizer

from rag_platform.core.config import get_settings
from rag_platform.core.embedding_registry import (
    EmbeddingModelConfig,
    get_active_model,
)
from rag_platform.core.metrics import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DURATION,
    EMBEDDING_FAILURES,
    EMBEDDING_REQUESTS,
)
from rag_platform.services.embedding_contract import validate_embedding_dimension
from rag_platform.services.readiness import MODEL_READY_KEY


def _resolve_device(requested: str) -> str:
    """Resolve embedding device.  ``auto`` picks CUDA if *nvidia-smi* reports
    ≥1 GB free memory, otherwise falls back to CPU."""
    if requested != "auto":
        return requested
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            free_mb = int(result.stdout.strip().split("\n")[0])
            if free_mb >= 1024:
                return "cuda"
    except Exception:
        pass
    return "cpu"


def _onnx_provider(device: str) -> str:
    if device == "cuda":
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


# ── per-model session + tokenizer cache ─────────────────────────────────

_SESSION_CACHE: dict[str, tuple] = {}


def _load_model(cfg: EmbeddingModelConfig) -> tuple:
    """Load (session, tokenizer, resolved_device) for *cfg*."""
    if cfg.id in _SESSION_CACHE:
        return _SESSION_CACHE[cfg.id]

    device = _resolve_device(cfg.device)
    provider = _onnx_provider(device)

    # Download the ONNX subfolder so model.onnx_data is co-located.
    local_dir = snapshot_download(
        cfg.model_name,
        allow_patterns=[f"{cfg.onnx_subfolder}/*"] if cfg.onnx_subfolder else None,
    )
    if cfg.onnx_subfolder:
        onnx_dir = Path(local_dir) / cfg.onnx_subfolder
    else:
        onnx_dir = Path(local_dir)

    onnx_file = next(onnx_dir.glob("*.onnx"))
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))

    available = set(ort.get_available_providers())
    use_provider = provider if provider in available else "CPUExecutionProvider"
    session = ort.InferenceSession(str(onnx_file), providers=[use_provider])

    result = (session, tokenizer, device)
    _SESSION_CACHE[cfg.id] = result
    return result


def _clear_session_cache() -> None:
    """Force reload on next call (used after model switch)."""
    _SESSION_CACHE.clear()


# ── public API (backward-compatible) ────────────────────────────────────

def model() -> ort.InferenceSession:
    """Return the active model's ONNX session."""
    cfg = get_active_model()
    return _load_model(cfg)[0]


def embed(texts: list[str], *, cfg: EmbeddingModelConfig | None = None) -> list[list[float]]:
    """Embed *texts* using the active (or explicitly passed) model.

    If the model has ``passage_prefix``, it is prepended automatically.
    Vectors are zero-padded to ``MAX_VECTOR_DIMENSION`` for pgvector.
    """
    if cfg is None:
        cfg = get_active_model()

    EMBEDDING_REQUESTS.inc()
    EMBEDDING_BATCH_SIZE.observe(len(texts))
    started = time.perf_counter()
    try:
        session, tokenizer, _ = _load_model(cfg)

        # Apply passage prefix if the model requires it.
        if cfg.passage_prefix:
            texts = [cfg.passage_prefix + t for t in texts]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_input_tokens,
            return_tensors="np",
        )
        valid = {i.name for i in session.get_inputs()}
        ort_inputs = {k: v for k, v in inputs.items() if k in valid}
        # Some models (E5, XLM-R) need token_type_ids; add zeros if missing.
        if "token_type_ids" in valid and "token_type_ids" not in ort_inputs:
            ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
        outputs = session.run(None, ort_inputs)
        token_embeddings: np.ndarray = outputs[0]

        # Mean-pool over token embeddings, respecting the attention mask.
        attention_mask: np.ndarray = inputs["attention_mask"]
        mask = np.expand_dims(attention_mask, axis=-1)
        summed = np.sum(token_embeddings * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts

        # L2-normalize.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        embeddings = pooled / norms

        # Convert to list and zero-pad to MAX_VECTOR_DIMENSION.
        raw = embeddings.tolist()
        return [cfg.pad_vector(v) for v in raw]
    except Exception:
        EMBEDDING_FAILURES.inc()
        raise
    finally:
        EMBEDDING_DURATION.observe(time.perf_counter() - started)


def embed_query(query: str, *, cfg: EmbeddingModelConfig | None = None) -> list[float]:
    """Embed a single query string with query_prefix if needed."""
    if cfg is None:
        cfg = get_active_model()

    EMBEDDING_REQUESTS.inc()
    started = time.perf_counter()
    try:
        session, tokenizer, _ = _load_model(cfg)

        text = (cfg.query_prefix + query) if cfg.query_prefix else query
        inputs = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=cfg.max_input_tokens,
            return_tensors="np",
        )
        valid = {i.name for i in session.get_inputs()}
        ort_inputs = {k: v for k, v in inputs.items() if k in valid}
        if "token_type_ids" in valid and "token_type_ids" not in ort_inputs:
            ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
        outputs = session.run(None, ort_inputs)
        token_embeddings: np.ndarray = outputs[0]

        attention_mask: np.ndarray = inputs["attention_mask"]
        mask = np.expand_dims(attention_mask, axis=-1)
        summed = np.sum(token_embeddings * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        vec = (pooled / norms)[0].tolist()
        return cfg.pad_vector(vec)
    except Exception:
        EMBEDDING_FAILURES.inc()
        raise
    finally:
        EMBEDDING_DURATION.observe(time.perf_counter() - started)


def dimension(cfg: EmbeddingModelConfig | None = None) -> int:
    """Return the actual (unpadded) embedding dimension of the active model."""
    if cfg is None:
        cfg = get_active_model()
    return cfg.dimension


# ── worker startup contract ─────────────────────────────────────────────

@worker_process_init.connect
def validate_model_contract(**kwargs: object) -> None:
    cfg = get_active_model()
    detected = dimension(cfg)
    validate_embedding_dimension(detected, cfg.dimension)
    device = _resolve_device(cfg.device)
    cache = Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        cache.set(
            MODEL_READY_KEY,
            json.dumps(
                {
                    "model": cfg.model_name,
                    "model_id": cfg.id,
                    "dimension": detected,
                    "device": device,
                    "index_version": cfg.index_version,
                }
            ),
            ex=60,
        )
    finally:
        cache.close()
