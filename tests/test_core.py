import uuid

import pytest
from fastapi import HTTPException

from rag_platform.core.auth import Principal
from rag_platform.services.retrieval import rrf
from rag_platform.worker.tasks import chunks


def test_rrf_combines_and_rewards_shared_results() -> None:
    shared, vector_only, lexical_only = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    scores = rrf([shared, vector_only], [shared, lexical_only])
    assert scores[shared] > scores[vector_only]
    assert scores[shared] > scores[lexical_only]


def test_chunker_is_deterministic_and_overlaps() -> None:
    text = " ".join(str(i) for i in range(30))
    result = chunks(text, target_words=10, overlap_words=2)
    assert result == chunks(text, target_words=10, overlap_words=2)
    assert result[0].split()[-2:] == result[1].split()[:2]


def test_principal_rejects_cross_tenant_scope() -> None:
    allowed = uuid.uuid4()
    who = Principal(
        uuid.uuid4(), frozenset({allowed}), frozenset({"public"}), frozenset({"retrieval:search"})
    )
    with pytest.raises(HTTPException) as error:
        who.authorize(uuid.uuid4(), ["public"], "retrieval:search")
    assert error.value.status_code == 403


def test_principal_requires_permission_and_collection() -> None:
    project = uuid.uuid4()
    who = Principal(
        uuid.uuid4(), frozenset({project}), frozenset({"public"}), frozenset({"documents:read"})
    )
    with pytest.raises(HTTPException):
        who.authorize(project, ["private"], "documents:read")
    with pytest.raises(HTTPException):
        who.authorize(project, ["public"], "documents:delete")
