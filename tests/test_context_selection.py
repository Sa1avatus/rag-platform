from typing import Any

from rag_platform.services.context_selection import select_context


def result(chunk: str, document: str, content: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk,
        "document_id": document,
        "source_type": "resume",
        "source_id": "resume-1",
        "content": content,
    }


def test_context_selection_applies_document_duplicate_and_token_limits() -> None:
    results = [
        result("a", "doc-1", "Python FastAPI production systems"),
        result("b", "doc-1", "Python FastAPI production systems and APIs"),
        result("c", "doc-1", "Distinct database operations"),
        result("d", "doc-2", "A" * 100),
        result("e", "doc-3", "Short context"),
    ]

    selection = select_context(
        results,
        max_chunks=5,
        max_estimated_tokens=20,
        per_document_limit=1,
        near_duplicate_threshold=0.75,
    )

    assert [item["chunk_id"] for item in selection.items] == ["a", "e"]
    assert selection.document_limit_suppressed == 2
    assert selection.token_budget_suppressed == 1
    assert selection.estimated_tokens <= 20


def test_context_selection_suppresses_near_duplicates_across_documents() -> None:
    selection = select_context(
        [
            result("a", "doc-1", "Python FastAPI machine learning"),
            result("b", "doc-2", "Python FastAPI machine learning role"),
        ],
        max_chunks=5,
        max_estimated_tokens=100,
        per_document_limit=2,
        near_duplicate_threshold=0.75,
    )

    assert [item["chunk_id"] for item in selection.items] == ["a"]
    assert selection.duplicates_suppressed == 1
