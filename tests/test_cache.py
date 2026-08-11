from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from rag_platform.services import cache


class FakeRedis:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = values or {}
        self.closed = False

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, **kwargs: object) -> None:
        self.values[key] = value

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)

    async def unlink(self, key: str) -> int:
        return await self.delete(key)

    async def scan_iter(self, *, match: str, count: int) -> AsyncIterator[str]:
        prefix = match.removesuffix("*")
        for key in list(self.values):
            if key.startswith(prefix):
                yield key

    async def aclose(self) -> None:
        self.closed = True


def settings() -> SimpleNamespace:
    return SimpleNamespace(
        cache_namespace="rag:cache:test",
        embedding_backend="sentence-transformers",
        embedding_dimension=1024,
        embedding_model="BAAI/bge-m3",
        embedding_normalization="l2",
        embedding_revision="revision-1",
        query_embedding_cache_enabled=True,
        query_embedding_cache_ttl_seconds=3600,
        redis_url="redis://unused",
    )


@pytest.mark.asyncio
async def test_query_embedding_cache_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(cache, "get_settings", settings)
    monkeypatch.setattr(cache.Redis, "from_url", lambda *args, **kwargs: redis)

    await cache.set_query_embedding("  deployment guide  ", [0.1, 0.2])
    assert await cache.get_query_embedding("deployment guide") == [0.1, 0.2]
    assert redis.closed is True


@pytest.mark.asyncio
async def test_clear_rag_cache_preserves_other_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis(
        {
            "rag:cache:test:query-embedding:one": "[0.1]",
            "rag:cache:test:query-embedding:two": "[0.2]",
            "celery-task-meta:unrelated": "keep",
        }
    )
    monkeypatch.setattr(cache, "get_settings", settings)
    monkeypatch.setattr(cache.Redis, "from_url", lambda *args, **kwargs: redis)

    assert await cache.clear_rag_cache() == 2
    assert redis.values == {"celery-task-meta:unrelated": "keep"}
