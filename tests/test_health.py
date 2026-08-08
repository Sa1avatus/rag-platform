import json
from typing import Any

import pytest

from rag_platform.services import health, readiness


class FakeSession:
    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, statement: object) -> None:
        return None

    async def scalar(self, statement: object) -> str:
        return "17.0"


class FakeRedis:
    def __init__(self, heartbeat: str | None = None) -> None:
        self.heartbeat = heartbeat
        self.closed = False

    async def get(self, key: str) -> str | None:
        return self.heartbeat

    async def info(self, section: str) -> dict[str, str]:
        return {"redis_version": "7.4"}

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_readiness_requires_compatible_worker_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat = json.dumps({"model": "BAAI/bge-m3", "dimension": 1024, "device": "cpu"})
    cache = FakeRedis(heartbeat)
    monkeypatch.setattr(readiness, "Session", FakeSession)
    monkeypatch.setattr(readiness.Redis, "from_url", lambda *args, **kwargs: cache)

    ready, components = await readiness.readiness_status()

    assert ready is True
    assert components["postgresql"]["status"] == "up"
    assert components["embedding_model"]["status"] == "ready"
    assert cache.closed is True


@pytest.mark.asyncio
async def test_readiness_reports_missing_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = FakeRedis()
    monkeypatch.setattr(readiness, "Session", FakeSession)
    monkeypatch.setattr(readiness.Redis, "from_url", lambda *args, **kwargs: cache)

    ready, components = await readiness.readiness_status()

    assert ready is False
    assert components["embedding_model"] == {"status": "not_ready"}


@pytest.mark.asyncio
async def test_system_health_aggregates_degraded_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def up() -> dict[str, Any]:
        return {"status": "up"}

    async def degraded() -> dict[str, Any]:
        return {"status": "not_ready"}

    for name in ("_postgresql", "_redis", "_opensearch", "_minio", "_reranker"):
        monkeypatch.setattr(health, name, up)
    monkeypatch.setattr(health, "_worker", degraded)

    result = await health.system_health()

    assert result["status"] == "degraded"
    assert len(result["components"]) == 7


@pytest.mark.asyncio
async def test_timed_probe_converts_failure_to_down() -> None:
    async def fail() -> dict[str, Any]:
        raise TimeoutError

    result = await health._timed_probe("dependency", fail)

    assert result["name"] == "dependency"
    assert result["status"] == "down"
    assert result["error"] == "TimeoutError"


@pytest.mark.asyncio
async def test_dependency_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "Session", FakeSession)
    assert (await health._postgresql())["status"] == "up"

    cache = FakeRedis()
    monkeypatch.setattr(health.Redis, "from_url", lambda *args, **kwargs: cache)
    redis_result = await health._redis()
    assert redis_result == {"status": "up", "version": "7.4"}
    assert cache.closed is True

    monkeypatch.setattr(
        health,
        "minio_client",
        lambda: type("Minio", (), {"list_buckets": lambda self: ["sources"]})(),
    )
    assert (await health._minio())["bucket_count"] == 1


@pytest.mark.asyncio
async def test_opensearch_and_worker_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    class SearchClient:
        async def info(self) -> dict[str, object]:
            return {"version": {"number": "2.19.1"}}

        class Cluster:
            async def health(self) -> dict[str, str]:
                return {"status": "yellow"}

        cluster = Cluster()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(health, "opensearch_client", SearchClient)
    result = await health._opensearch()
    assert result["status"] == "up"
    assert result["version"] == "2.19.1"

    async def ready() -> tuple[bool, dict[str, object]]:
        return True, {"embedding_model": {"model": "bge", "dimension": 1024, "device": "cpu"}}

    monkeypatch.setattr(health, "readiness_status", ready)
    assert (await health._worker())["status"] == "up"


@pytest.mark.asyncio
async def test_disabled_reranker_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health,
        "get_settings",
        lambda: type("Settings", (), {"reranker_enabled": False})(),
    )
    assert await health._reranker() == {"status": "disabled"}
