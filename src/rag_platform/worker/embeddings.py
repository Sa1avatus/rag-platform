import json
import time
from functools import lru_cache

from celery.signals import worker_process_init
from redis import Redis
from sentence_transformers import SentenceTransformer

from rag_platform.core.config import get_settings
from rag_platform.core.metrics import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DURATION,
    EMBEDDING_FAILURES,
    EMBEDDING_REQUESTS,
)
from rag_platform.services.embedding_contract import validate_embedding_dimension
from rag_platform.services.readiness import MODEL_READY_KEY


def _resolve_device(requested: str) -> str:
    """Resolve embedding device. 'auto' picks CUDA if available and has
    enough free memory (>1 GB), otherwise falls back to CPU."""
    if requested != "auto":
        return requested
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        free, total = torch.cuda.mem_info(0)
        free_gb = free / (1024**3)
        if free_gb < 1.0:
            return "cpu"
        return "cuda"
    except Exception:
        return "cpu"


@lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    settings = get_settings()
    device = _resolve_device(settings.embedding_device)
    return SentenceTransformer(settings.embedding_model, device=device)


def embed(texts: list[str]) -> list[list[float]]:
    EMBEDDING_REQUESTS.inc()
    EMBEDDING_BATCH_SIZE.observe(len(texts))
    started = time.perf_counter()
    try:
        vectors = model().encode(
            texts,
            batch_size=get_settings().embedding_batch_size,
            normalize_embeddings=True,
        )
    except Exception:
        EMBEDDING_FAILURES.inc()
        raise
    finally:
        EMBEDDING_DURATION.observe(time.perf_counter() - started)
    return [[float(component) for component in vector] for vector in vectors.tolist()]


def dimension() -> int:
    return model().get_sentence_embedding_dimension() or 0


@worker_process_init.connect
def validate_model_contract(**kwargs: object) -> None:
    detected = dimension()
    settings = get_settings()
    validate_embedding_dimension(detected, settings.embedding_dimension)
    device = _resolve_device(settings.embedding_device)
    cache = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        cache.set(
            MODEL_READY_KEY,
            json.dumps(
                {
                    "model": settings.embedding_model,
                    "dimension": detected,
                    "device": device,
                }
            ),
            ex=60,
        )
    finally:
        cache.close()
