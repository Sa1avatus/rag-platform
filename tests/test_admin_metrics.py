import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.dialects import postgresql

from rag_platform.services.admin_metrics import metric_statement, metric_timeseries


def test_metric_statement_scopes_error_counts() -> None:
    project_id = uuid.uuid4()
    statement = metric_statement(
        "indexing_errors",
        project_id=project_id,
        collection="manuals",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        step="hour",
    )
    compiled = str(statement.compile(dialect=postgresql.dialect()))
    assert "indexing_jobs.project_id" in compiled
    assert "indexing_jobs.payload" in compiled
    assert "date_trunc" in compiled


@pytest.mark.asyncio
async def test_metric_timeseries_serializes_rows() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)

    class Rows:
        def all(self) -> list[object]:
            return [type("MetricRow", (), {"bucket": timestamp, "value": 3})()]

    class Session:
        async def execute(self, statement: object) -> Rows:
            return Rows()

    points = await metric_timeseries(
        Session(),  # type: ignore[arg-type]
        "documents",
        project_id=None,
        collection=None,
        start=timestamp,
        end=datetime(2026, 1, 2, tzinfo=UTC),
        step="day",
    )
    assert points == [{"timestamp": timestamp, "value": 3}]
