from functools import lru_cache

from sentence_transformers import SentenceTransformer

from rag_platform.core.config import get_settings


@lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model, device=settings.embedding_device)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = model().encode(texts, batch_size=get_settings().embedding_batch_size, normalize_embeddings=True)
    return vectors.tolist()


def dimension() -> int:
    return model().get_sentence_embedding_dimension() or 0
