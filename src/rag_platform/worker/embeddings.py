import json
from functools import lru_cache

from celery.signals import worker_process_init
from redis import Redis
from sentence_transformers import SentenceTransformer

from rag_platform.core.config import get_settings
from rag_platform.services.embedding_contract import validate_embedding_dimension
from rag_platform.services.readiness import MODEL_READY_KEY


@lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model, device=settings.embedding_device)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = model().encode(
        texts,
        batch_size=get_settings().embedding_batch_size,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def dimension() -> int:
    return model().get_sentence_embedding_dimension() or 0


@worker_process_init.connect
def validate_model_contract(**kwargs: object) -> None:
    detected = dimension()
    settings = get_settings()
    validate_embedding_dimension(detected, settings.embedding_dimension)
    cache = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        cache.set(
            MODEL_READY_KEY,
            json.dumps(
                {
                    "model": settings.embedding_model,
                    "dimension": detected,
                    "device": settings.embedding_device,
                }
            ),
            ex=60,
        )
    finally:
        cache.close()
