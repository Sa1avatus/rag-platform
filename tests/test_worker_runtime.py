import json
import uuid
from types import SimpleNamespace

import pytest

from rag_platform.db.models import OutboxEvent
from rag_platform.worker import embeddings, outbox


class FakeVectors:
    def tolist(self) -> list[list[float]]:
        return [[1, 2.5], [3, 4]]


class FakeModel:
    def encode(self, texts: list[str], **kwargs: object) -> FakeVectors:
        return FakeVectors()

    def get_sentence_embedding_dimension(self) -> int:
        return 1024


class FakeCache:
    def __init__(self) -> None:
        self.value = ""
        self.closed = False

    def set(self, key: str, value: str, ex: int) -> None:
        self.value = value

    def close(self) -> None:
        self.closed = True


def test_embedding_runtime_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        embedding_batch_size=16,
        embedding_dimension=1024,
        embedding_model="BAAI/bge-m3",
        embedding_device="cpu",
        redis_url="redis://test",
    )
    cache = FakeCache()
    monkeypatch.setattr(embeddings, "model", FakeModel)
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    monkeypatch.setattr(embeddings.Redis, "from_url", lambda *args, **kwargs: cache)

    assert embeddings.embed(["one", "two"]) == [[1.0, 2.5], [3.0, 4.0]]
    assert embeddings.dimension() == 1024
    embeddings.validate_model_contract()
    assert json.loads(cache.value)["dimension"] == 1024
    assert cache.closed is True


class Rows:
    def __init__(self, rows: list[OutboxEvent]) -> None:
        self.rows = rows

    def all(self) -> list[OutboxEvent]:
        return self.rows


class FakeSession:
    def __init__(self, rows: list[OutboxEvent]) -> None:
        self.rows = rows
        self.committed = False

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def scalars(self, statement: object) -> Rows:
        return Rows(self.rows)

    async def commit(self) -> None:
        self.committed = True


@pytest.mark.asyncio
async def test_publish_pending_marks_and_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    event = OutboxEvent(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        payload={
            "type": "document.index",
            "version_id": str(uuid.uuid4()),
            "job_id": str(uuid.uuid4()),
            "attempts": 1,
        },
    )
    session = FakeSession([event])
    sent: list[tuple[str, list[str], str]] = []
    monkeypatch.setattr(outbox, "Session", lambda: session)
    monkeypatch.setattr(
        outbox.current_app,
        "send_task",
        lambda name, args, task_id: sent.append((name, args, task_id)),
    )

    assert await outbox.publish_pending() == 1
    assert event.payload["attempts"] == 2
    assert "published_at" in event.payload
    assert sent[0][2] == str(event.id)
    assert session.committed is True
