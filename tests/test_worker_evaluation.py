import uuid
from typing import Any

import pytest

from rag_platform.db.models import EvaluationCase, EvaluationDataset, EvaluationRun
from rag_platform.worker import evaluation


class Rows:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def all(self) -> list[object]:
        return self.rows


class FakeSession:
    def __init__(self, gets: list[Any], cases: list[EvaluationCase] | None = None) -> None:
        self.gets = gets
        self.cases = cases or []
        self.added: list[object] = []
        self.commits = 0
        self.rolled_back = False

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, model: object, identifier: uuid.UUID) -> Any:
        return self.gets.pop(0)

    async def execute(self, statement: object) -> None:
        return None

    async def scalars(self, statement: object) -> Rows:
        return Rows(self.cases)

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rolled_back = True


@pytest.mark.asyncio
async def test_evaluate_run_completes_and_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, project_id, dataset_id, run_id = (
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
    )
    run = EvaluationRun(
        id=run_id,
        tenant_id=tenant_id,
        project_id=project_id,
        payload={
            "status": "queued",
            "dataset_id": str(dataset_id),
            "configuration": {"use_reranker": False},
        },
    )
    dataset = EvaluationDataset(
        id=dataset_id,
        tenant_id=tenant_id,
        project_id=project_id,
        payload={"collections": ["manuals"]},
    )
    case = EvaluationCase(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        payload={
            "dataset_id": str(dataset_id),
            "query": "deployment",
            "expected_document_ids": ["guide-1"],
            "relevance_grades": {"guide-1": 3},
            "forbidden_results": ["forbidden"],
        },
    )
    session = FakeSession([run, dataset, run], [case])
    monkeypatch.setattr(evaluation, "Session", lambda: session)
    monkeypatch.setattr(evaluation, "embed", lambda texts: [[0.1, 0.2]])

    async def search(
        *args: object, **kwargs: object
    ) -> tuple[uuid.UUID, list[dict[str, object]], dict[str, object]]:
        return (
            uuid.uuid4(),
            [{"document_id": "guide-1", "chunk_id": "chunk-1"}],
            {"latency_ms": 12.5},
        )

    monkeypatch.setattr(evaluation, "search", search)
    await evaluation.evaluate_run(run_id)

    assert run.payload["status"] == "completed"
    assert run.payload["case_count"] == 1
    assert run.payload["metrics"]["Recall@1"] == 1.0
    assert len(session.added) == 1
    assert session.commits == 3


@pytest.mark.asyncio
async def test_evaluate_run_marks_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id, dataset_id = uuid.uuid4(), uuid.uuid4()
    run = EvaluationRun(
        id=run_id,
        tenant_id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        payload={"status": "queued", "dataset_id": str(dataset_id)},
    )
    session = FakeSession([run, None, run])
    monkeypatch.setattr(evaluation, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="dataset is missing"):
        await evaluation.evaluate_run(run_id)

    assert session.rolled_back is True
    assert run.payload["status"] == "failed"


@pytest.mark.asyncio
async def test_evaluate_run_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, dataset_id, project_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    run = EvaluationRun(
        id=run_id,
        tenant_id=uuid.uuid4(),
        project_id=project_id,
        payload={
            "status": "queued",
            "dataset_id": str(dataset_id),
            "configuration": "invalid",
        },
    )
    dataset = EvaluationDataset(
        id=dataset_id,
        tenant_id=run.tenant_id,
        project_id=project_id,
        payload={"collections": ["manuals"]},
    )
    session = FakeSession([run, dataset, run])
    monkeypatch.setattr(evaluation, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="configuration is invalid"):
        await evaluation.evaluate_run(run_id)

    assert run.payload["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("run", [None, EvaluationRun(payload={"status": "completed"})])
async def test_evaluate_run_ignores_missing_or_completed(
    monkeypatch: pytest.MonkeyPatch, run: EvaluationRun | None
) -> None:
    session = FakeSession([run])
    monkeypatch.setattr(evaluation, "Session", lambda: session)
    await evaluation.evaluate_run(uuid.uuid4())
    assert session.commits == 0


def test_evaluation_value_helpers() -> None:
    assert evaluation._string_list([1, "two"]) == ["1", "two"]
    assert evaluation._string_list("not-list") == []
    assert evaluation._dict({"enabled": True}) == {"enabled": True}
    assert evaluation._dict([]) == {}
    assert evaluation._grade_dict({1: "3"}) == {"1": 3}
    assert evaluation._grade_dict([]) == {}
    assert evaluation._mean([1.0, 3.0]) == 2.0
    assert evaluation._mean([]) == 0.0
    assert evaluation._percentile([1.0, 2.0, 10.0], 0.95) == 10.0
    assert evaluation._percentile([], 0.95) == 0.0
