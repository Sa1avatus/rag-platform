import pytest
from pydantic import ValidationError

from rag_platform.api.schemas import RuntimeSettingsUpdate
from rag_platform.db.models import RuntimeSetting
from rag_platform.services.runtime_settings import (
    apply_runtime_settings,
    read_runtime_settings,
    runtime_settings_response,
)


class ScalarRows:
    def __init__(self, rows: list[RuntimeSetting]) -> None:
        self.rows = rows

    def all(self) -> list[RuntimeSetting]:
        return self.rows


class Session:
    def __init__(self, rows: list[RuntimeSetting] | None = None) -> None:
        self.rows = rows or []
        self.added: list[RuntimeSetting] = []

    async def scalars(self, statement: object) -> ScalarRows:
        return ScalarRows(self.rows)

    def add(self, row: RuntimeSetting) -> None:
        self.added.append(row)


@pytest.mark.asyncio
async def test_runtime_settings_merge_saved_values_with_catalog() -> None:
    session = Session([RuntimeSetting(key="default_vector_top_k", value=42)])
    values = await read_runtime_settings(session)  # type: ignore[arg-type]
    assert values["default_vector_top_k"] == 42
    assert values["default_bm25_top_k"] == 30

    response = await runtime_settings_response(session)  # type: ignore[arg-type]
    vector = next(item for item in response["settings"] if item["key"] == "default_vector_top_k")  # type: ignore[union-attr]
    assert vector["value"] == 42
    assert vector["description"]
    assert vector["minimum"] == 1


@pytest.mark.asyncio
async def test_apply_runtime_settings_updates_and_inserts() -> None:
    existing = RuntimeSetting(key="reranker_enabled", value=True)
    session = Session([existing])
    await apply_runtime_settings(  # type: ignore[arg-type]
        session,
        {"reranker_enabled": False, "default_vector_top_k": 50},
    )
    assert existing.value is False
    assert [(row.key, row.value) for row in session.added] == [("default_vector_top_k", 50)]


def test_runtime_settings_reject_unknown_or_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        RuntimeSettingsUpdate.model_validate({"unknown_setting": True})
    with pytest.raises(ValidationError):
        RuntimeSettingsUpdate(default_vector_top_k=0)
