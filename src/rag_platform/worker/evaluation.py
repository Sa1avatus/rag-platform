import math
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select

from rag_platform.api.schemas import SearchRequest
from rag_platform.core.auth import Principal
from rag_platform.core.metrics import EVALUATION_FAILURES, EVALUATION_RUNS
from rag_platform.db.models import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
)
from rag_platform.db.session import Session
from rag_platform.services.evaluation_metrics import aggregate_metrics, case_metrics
from rag_platform.services.retrieval import search
from rag_platform.worker.embeddings import embed


async def evaluate_run(run_id: uuid.UUID) -> None:
    async with Session() as session:
        run = await session.get(EvaluationRun, run_id)
        if run is None or run.payload.get("status") == "completed":
            return
        run.payload = {
            **run.payload,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
        }
        await session.commit()
        EVALUATION_RUNS.inc()
        try:
            await session.execute(
                delete(EvaluationResult).where(
                    EvaluationResult.payload["run_id"].astext == str(run_id)
                )
            )
            await session.commit()
            dataset_id = uuid.UUID(str(run.payload["dataset_id"]))
            dataset = await session.get(EvaluationDataset, dataset_id)
            if dataset is None or dataset.project_id is None:
                raise RuntimeError("evaluation dataset is missing")
            collections = _string_list(dataset.payload.get("collections"))
            cases = (
                await session.scalars(
                    select(EvaluationCase).where(
                        EvaluationCase.payload["dataset_id"].astext == str(dataset.id)
                    )
                )
            ).all()
            principal = Principal(
                dataset.tenant_id,
                uuid.UUID(int=0),
                frozenset({dataset.project_id}),
                frozenset(collections),
                frozenset({"retrieval:search", "admin:evaluate"}),
            )
            configuration = run.payload.get("configuration", {})
            if not isinstance(configuration, dict):
                raise RuntimeError("evaluation configuration is invalid")
            case_metric_rows_before: list[dict[str, float]] = []
            case_metric_rows_after: list[dict[str, float]] = []
            case_metric_deltas: list[dict[str, float]] = []
            latencies: list[float] = []
            forbidden_violations = 0
            for case in cases:
                query = str(case.payload.get("query", ""))
                request = SearchRequest(
                    project_id=dataset.project_id,
                    collections=collections,
                    query=query,
                    filters=_dict(case.payload.get("filters")),
                    **configuration,
                )
                query_vector = embed([query])[0]
                _, results, trace = await search(
                    session,
                    principal,
                    request,
                    query_vector=query_vector,
                )
                expected_chunks = _string_list(case.payload.get("expected_chunk_ids"))
                identifier_field = "chunk_id" if expected_chunks else "document_id"
                retrieved_after = [str(result[identifier_field]) for result in results]
                fusion_candidates = trace.get("fusion_candidates", [])
                retrieved_before = [
                    str(candidate[identifier_field])
                    for candidate in fusion_candidates[: request.rerank_top_k]
                    if isinstance(candidate, dict) and identifier_field in candidate
                ]
                expected = expected_chunks or _string_list(
                    case.payload.get("expected_document_ids")
                )
                grades = {
                    **{item_id: 1 for item_id in expected},
                    **_grade_dict(case.payload.get("relevance_grades")),
                }
                metrics_before = case_metrics(retrieved_before, grades)
                metrics_after = case_metrics(retrieved_after, grades)
                reranker_delta = _delta_metrics(metrics_before, metrics_after)
                forbidden = set(_string_list(case.payload.get("forbidden_results")))
                violations = len(set(retrieved_after) & forbidden)
                forbidden_violations += violations
                case_metric_rows_before.append(metrics_before)
                case_metric_rows_after.append(metrics_after)
                case_metric_deltas.append(reranker_delta)
                latencies.append(float(trace["latency_ms"]))
                session.add(
                    EvaluationResult(
                        tenant_id=run.tenant_id,
                        project_id=run.project_id,
                        payload={
                            "run_id": str(run.id),
                            "case_id": str(case.id),
                            "retrieved_ids": retrieved_after,
                            "retrieved_ids_before_reranking": retrieved_before,
                            "metrics": metrics_after,
                            "metrics_before_reranking": metrics_before,
                            "metrics_after_reranking": metrics_after,
                            "reranker_delta": reranker_delta,
                            "forbidden_violations": violations,
                            "latency_ms": trace["latency_ms"],
                        },
                    )
                )
            aggregate_before = aggregate_metrics(case_metric_rows_before)
            aggregate_after = aggregate_metrics(case_metric_rows_after)
            aggregate_delta = aggregate_metrics(case_metric_deltas)
            aggregate_after["average_latency_ms"] = _mean(latencies)
            aggregate_after["p95_latency_ms"] = _percentile(latencies, 0.95)
            aggregate_after["forbidden_violations"] = float(forbidden_violations)
            run = await session.get(EvaluationRun, run_id)
            if run:
                run.payload = {
                    **run.payload,
                    "status": "completed",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "metrics": aggregate_after,
                    "metrics_before_reranking": aggregate_before,
                    "metrics_after_reranking": aggregate_after,
                    "reranker_uplift": aggregate_delta,
                    "case_count": len(cases),
                }
            await session.commit()
        except Exception as exc:
            EVALUATION_FAILURES.inc()
            await session.rollback()
            run = await session.get(EvaluationRun, run_id)
            if run:
                run.payload = {
                    **run.payload,
                    "status": "failed",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "error": str(exc)[:4000],
                }
                await session.commit()
            raise


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _grade_dict(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(grade) for key, grade in value.items()}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _delta_metrics(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    return {name: after.get(name, 0.0) - before.get(name, 0.0) for name in after}
