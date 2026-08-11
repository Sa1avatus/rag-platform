import uuid

from rag_platform.services.versioning import (
    content_hash,
    normalize_content,
    stable_chunk_id,
    stable_document_id,
    stable_version_id,
)


def test_content_normalization_is_deterministic() -> None:
    first = normalize_content("Cafe\u0301  \r\nsecond\x00 line  \r\n")
    second = normalize_content("Café\nsecond line")

    assert first == second
    assert content_hash(first) == content_hash(second)


def test_domain_ids_are_stable_and_version_aware() -> None:
    tenant_id = uuid.uuid4()
    project_id = uuid.uuid4()
    document_id = stable_document_id(tenant_id, project_id, "jobs", "job-123")

    assert document_id == stable_document_id(tenant_id, project_id, "jobs", "job-123")
    assert document_id != stable_document_id(tenant_id, project_id, "jobs", "job-456")

    digest = content_hash("content")
    version_id = stable_version_id(document_id, 1, digest)
    assert version_id == stable_version_id(document_id, 1, digest)
    assert version_id != stable_version_id(document_id, 2, digest)

    chunk_id = stable_chunk_id(version_id, "word-window-v1", 0, digest)
    assert chunk_id == stable_chunk_id(version_id, "word-window-v1", 0, digest)
    assert chunk_id != stable_chunk_id(version_id, "word-window-v2", 0, digest)
