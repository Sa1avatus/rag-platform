from types import SimpleNamespace

import pytest

from rag_platform.services import embedding_admin


@pytest.mark.asyncio
async def test_embedding_profile_reports_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    async def readiness() -> tuple[bool, dict[str, object]]:
        return True, {
            "embedding_model": {
                "status": "ready",
                "model": "BAAI/bge-m3",
                "dimension": 1024,
                "device": "cpu",
            }
        }

    monkeypatch.setattr(embedding_admin, "readiness_status", readiness)
    monkeypatch.setattr(
        embedding_admin,
        "get_settings",
        lambda: SimpleNamespace(embedding_model="BAAI/bge-m3", embedding_dimension=1024),
    )
    result = await embedding_admin.embedding_profile()
    assert result["status"] == "ready"
    assert result["compatible"] is True
    assert result["device"] == "cpu"


@pytest.mark.asyncio
async def test_embedding_profile_reports_incompatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    async def readiness() -> tuple[bool, dict[str, object]]:
        return False, {
            "embedding_model": {
                "status": "incompatible",
                "model": "BAAI/bge-m3",
                "dimension": 768,
                "device": "cuda",
            }
        }

    monkeypatch.setattr(embedding_admin, "readiness_status", readiness)
    monkeypatch.setattr(
        embedding_admin,
        "get_settings",
        lambda: SimpleNamespace(embedding_model="BAAI/bge-m3", embedding_dimension=1024),
    )
    result = await embedding_admin.embedding_profile()
    assert result["status"] == "incompatible"
    assert result["compatible"] is False
    assert result["expected_dimension"] == 1024
