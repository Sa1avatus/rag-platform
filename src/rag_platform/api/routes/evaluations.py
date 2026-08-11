import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import EvaluationDatasetCreate, EvaluationRunCreate
from rag_platform.core.auth import Principal, principal
from rag_platform.core.config import get_settings
from rag_platform.db.models import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    OutboxEvent,
)
from rag_platform.db.session import get_session
from rag_platform.services.evaluation_metrics import pin_retrieval_configuration

router = APIRouter(prefix="/v1/evaluations", tags=["evaluations"])


@router.post("/datasets", status_code=201)
async def create_dataset(
    data: EvaluationDatasetCreate,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    who.authorize(data.project_id, data.collections, "admin:evaluate")
    dataset = EvaluationDataset(
        tenant_id=who.tenant_id,
        project_id=data.project_id,
        payload={
            "name": data.name,
            "version": data.version,
            "collections": data.collections,
            "case_count": len(data.cases),
        },
    )
    session.add(dataset)
    await session.flush()
    for case in data.cases:
        session.add(
            EvaluationCase(
                tenant_id=who.tenant_id,
                project_id=data.project_id,
                payload={"dataset_id": str(dataset.id), **case.model_dump()},
            )
        )
    await session.commit()
    return {"id": dataset.id, **dataset.payload}


@router.post("/run", status_code=202)
async def create_run(
    data: EvaluationRunCreate,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    dataset = await session.scalar(
        select(EvaluationDataset).where(
            EvaluationDataset.id == data.dataset_id,
            EvaluationDataset.tenant_id == who.tenant_id,
        )
    )
    if dataset is None or dataset.project_id is None:
        raise HTTPException(404, "evaluation dataset not found")
    collections = dataset.payload.get("collections", [])
    if not isinstance(collections, list):
        raise HTTPException(409, "evaluation dataset collections are invalid")
    who.authorize(dataset.project_id, collections, "admin:evaluate")
    run = EvaluationRun(
        tenant_id=who.tenant_id,
        project_id=dataset.project_id,
        payload={
            "dataset_id": str(dataset.id),
            "status": "queued",
            "configuration": pin_retrieval_configuration(
                data.model_dump(exclude={"dataset_id"}),
                get_settings(),
            ),
        },
    )
    session.add(run)
    await session.flush()
    session.add(
        OutboxEvent(
            tenant_id=who.tenant_id,
            project_id=dataset.project_id,
            payload={
                "type": "evaluation.run",
                "run_id": str(run.id),
                "attempts": 0,
            },
        )
    )
    await session.commit()
    return {"id": run.id, "status": "queued"}


@router.get("/runs/{run_id}")
async def get_run(
    run_id: uuid.UUID,
    who: Principal = Depends(principal),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    run = await session.scalar(
        select(EvaluationRun).where(
            EvaluationRun.id == run_id,
            EvaluationRun.tenant_id == who.tenant_id,
        )
    )
    if run is None:
        raise HTTPException(404, "evaluation run not found")
    results = (
        await session.scalars(
            select(EvaluationResult).where(EvaluationResult.payload["run_id"].astext == str(run.id))
        )
    ).all()
    return {
        "id": run.id,
        **run.payload,
        "results": [{"id": result.id, **result.payload} for result in results],
    }
