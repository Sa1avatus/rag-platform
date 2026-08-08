import uuid
from typing import Any

import pytest
from fastapi import HTTPException

from rag_platform.api.routes import evaluations, feedback
from rag_platform.api.schemas import (
    EvaluationCaseInput,
    EvaluationDatasetCreate,
    EvaluationRunCreate,
    FeedbackCreate,
)
from rag_platform.core.auth import Principal
from rag_platform.db.models import Chunk, EvaluationDataset, EvaluationResult, RetrievalRequest


class ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def all(self) -> list[object]:
        return self.rows


class FakeSession:
    def __init__(self, values: list[Any] | None = None, rows: list[object] | None = None) -> None:
        self.values = list(values or [])
        self.rows = rows or []
        self.added: list[Any] = []
        self.commits = 0

    def add(self, value: Any) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        for value in self.added:
            if getattr(value, "id", None) is None:
                value.id = uuid.uuid4()

    async def commit(self) -> None:
        self.commits += 1

    async def scalar(self, statement: object) -> Any:
        return self.values.pop(0)

    async def scalars(self, statement: object) -> ScalarRows:
        return ScalarRows(self.rows)


def principal(tenant_id: uuid.UUID, project_id: uuid.UUID) -> Principal:
    return Principal(
        tenant_id,
        frozenset({project_id}),
        frozenset({"manuals"}),
        frozenset({"admin:evaluate", "feedback:write"}),
    )


@pytest.mark.asyncio
async def test_evaluation_dataset_run_and_result() -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    who = principal(tenant_id, project_id)
    dataset_session = FakeSession()
    dataset_result = await evaluations.create_dataset(
        EvaluationDatasetCreate(
            project_id=project_id,
            name="baseline",
            collections=["manuals"],
            cases=[EvaluationCaseInput(query="deployment", expected_document_ids=["guide-1"])],
        ),
        who,
        dataset_session,
    )
    assert dataset_result["case_count"] == 1
    assert len(dataset_session.added) == 2

    dataset = dataset_session.added[0]
    run_session = FakeSession([dataset])
    run_result = await evaluations.create_run(
        EvaluationRunCreate(dataset_id=dataset.id), who, run_session
    )
    assert run_result["status"] == "queued"
    assert len(run_session.added) == 2

    run = run_session.added[0]
    result = EvaluationResult(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        project_id=project_id,
        payload={"run_id": str(run.id), "recall_at_k": 1.0},
    )
    returned = await evaluations.get_run(run.id, who, FakeSession([run], [result]))
    assert returned["results"][0]["recall_at_k"] == 1.0


@pytest.mark.asyncio
async def test_evaluation_rejects_missing_or_malformed_dataset() -> None:
    tenant_id, project_id = uuid.uuid4(), uuid.uuid4()
    who = principal(tenant_id, project_id)
    data = EvaluationRunCreate(dataset_id=uuid.uuid4())
    with pytest.raises(HTTPException) as missing:
        await evaluations.create_run(data, who, FakeSession([None]))
    assert missing.value.status_code == 404

    malformed = EvaluationDataset(
        id=data.dataset_id,
        tenant_id=tenant_id,
        project_id=project_id,
        payload={"collections": "manuals"},
    )
    with pytest.raises(HTTPException) as invalid:
        await evaluations.create_run(data, who, FakeSession([malformed]))
    assert invalid.value.status_code == 409

    with pytest.raises(HTTPException) as missing_run:
        await evaluations.get_run(uuid.uuid4(), who, FakeSession([None]))
    assert missing_run.value.status_code == 404


@pytest.mark.asyncio
async def test_feedback_requires_chunk_from_retrieval_results() -> None:
    tenant_id, project_id, request_id, chunk_id = (
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
    )
    who = principal(tenant_id, project_id)
    retrieval_request = RetrievalRequest(
        id=request_id,
        tenant_id=tenant_id,
        project_id=project_id,
        payload={"results": [{"chunk_id": str(chunk_id)}], "configuration": {"top_k": 5}},
    )
    chunk = Chunk(
        id=chunk_id,
        tenant_id=tenant_id,
        project_id=project_id,
        collection="manuals",
    )
    data = FeedbackCreate(
        project_id=project_id,
        request_id=request_id,
        chunk_id=chunk_id,
        relevant=True,
        relevance_grade=3,
        comment="useful",
    )
    session = FakeSession([retrieval_request, chunk])
    result = await feedback.create_feedback(data, who, session)
    assert result["relevant"] is True
    assert result["retrieval_configuration"] == {"top_k": 5}

    other = Chunk(id=uuid.uuid4(), tenant_id=tenant_id, project_id=project_id, collection="manuals")
    with pytest.raises(HTTPException) as not_returned:
        await feedback.create_feedback(data, who, FakeSession([retrieval_request, other]))
    assert not_returned.value.status_code == 409

    with pytest.raises(HTTPException) as missing:
        await feedback.create_feedback(data, who, FakeSession([None, chunk]))
    assert missing.value.status_code == 404
