import secrets
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import (
    ApiKeyCreate,
    CollectionCreate,
    CollectionUpdate,
    ConfigurationComparisonRequest,
    EmbeddingReindexRequest,
    EvaluationRunCreate,
    ProjectCreate,
    ProjectUpdate,
    RuntimeSettingsUpdate,
    SearchRequest,
    TenantCreate,
)
from rag_platform.core.auth import Principal, admin, hash_key
from rag_platform.db.models import (
    ApiKey,
    AuditLog,
    Chunk,
    Collection,
    Document,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    IndexingJob,
    OutboxEvent,
    Project,
    RetrievalRequest,
    Tenant,
)
from rag_platform.db.session import get_session
from rag_platform.services.admin_metrics import metric_timeseries
from rag_platform.services.embedding_admin import embedding_profile
from rag_platform.services.health import system_health
from rag_platform.services.reconciliation import reconcile, reindex_collection, reindex_embeddings
from rag_platform.services.reranker import reranker_status, test_reranker_connection
from rag_platform.services.resources import system_resources
from rag_platform.services.retrieval import search
from rag_platform.services.runtime_settings import apply_runtime_settings, runtime_settings_response

router = APIRouter(
    prefix="/v1/admin",
    tags=["admin"],
    dependencies=[Depends(admin)],
)


def _audit(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    project_id: uuid.UUID | None,
    action: str,
    resource_type: str,
    resource_id: uuid.UUID,
) -> None:
    session.add(
        AuditLog(
            tenant_id=tenant_id,
            project_id=project_id,
            payload={
                "action": action,
                "resource_type": resource_type,
                "resource_id": str(resource_id),
            },
        )
    )


@router.post("/tenants", status_code=201)
async def create_tenant(
    data: TenantCreate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = Tenant(name=data.name)
    session.add(row)
    await session.flush()
    _audit(
        session,
        tenant_id=row.id,
        project_id=None,
        action="tenant.create",
        resource_type="tenant",
        resource_id=row.id,
    )
    await session.commit()
    return {"id": row.id, "name": row.name}


@router.post("/projects", status_code=201)
async def create_project(
    data: ProjectCreate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = Project(**data.model_dump())
    session.add(row)
    await session.flush()
    _audit(
        session,
        tenant_id=row.tenant_id,
        project_id=row.id,
        action="project.create",
        resource_type="project",
        resource_id=row.id,
    )
    await session.commit()
    return {"id": row.id, "slug": row.slug, "name": row.name}


@router.get("/projects")
async def projects(
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    rows = (await session.scalars(select(Project).order_by(Project.created_at.desc()))).all()
    return [
        {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "slug": row.slug,
            "name": row.name,
            "enabled": row.enabled,
        }
        for row in rows
    ]


@router.get("/documents")
async def admin_documents(
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    collection: str | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    filters = [
        Document.tenant_id == tenant_id,
        Document.project_id == project_id,
        Document.deleted_at.is_(None),
    ]
    if collection:
        filters.append(Document.collection == collection)
    rows = (
        await session.scalars(
            select(Document)
            .where(*filters)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return [
        {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "project_id": row.project_id,
            "collection": row.collection,
            "external_document_id": row.external_document_id,
            "current_version": row.current_version,
            "lock_version": row.lock_version,
            "metadata": row.metadata_,
        }
        for row in rows
    ]


@router.get("/documents/{document_id}/chunks")
async def admin_document_chunks(
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    collection: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    document = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
            Document.project_id == project_id,
            Document.collection == collection,
            Document.deleted_at.is_(None),
        )
    )
    if document is None:
        raise HTTPException(404, "document not found")
    rows = (
        await session.scalars(
            select(Chunk)
            .where(
                Chunk.document_id == document_id,
                Chunk.tenant_id == tenant_id,
                Chunk.project_id == project_id,
                Chunk.collection == collection,
            )
            .order_by(Chunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return [
        {
            "id": row.id,
            "chunk_index": row.chunk_index,
            "chunk_type": row.chunk_type,
            "content": row.content,
            "token_count": row.token_count,
            "language": row.language,
        }
        for row in rows
    ]


@router.get("/evaluation/datasets")
async def admin_evaluation_datasets(
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    rows = (
        await session.scalars(
            select(EvaluationDataset)
            .where(
                EvaluationDataset.tenant_id == tenant_id,
                EvaluationDataset.project_id == project_id,
            )
            .order_by(EvaluationDataset.created_at.desc())
        )
    ).all()
    return [
        {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "project_id": row.project_id,
            **row.payload,
        }
        for row in rows
    ]


@router.get("/evaluation/runs")
async def admin_evaluation_runs(
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    rows = (
        await session.scalars(
            select(EvaluationRun)
            .where(
                EvaluationRun.tenant_id == tenant_id,
                EvaluationRun.project_id == project_id,
            )
            .order_by(EvaluationRun.created_at.desc())
        )
    ).all()
    return [
        {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "project_id": row.project_id,
            **row.payload,
        }
        for row in rows
    ]


@router.get("/evaluation/runs/{run_id}")
async def admin_evaluation_run(
    run_id: uuid.UUID,
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    run = await session.scalar(
        select(EvaluationRun).where(
            EvaluationRun.id == run_id,
            EvaluationRun.tenant_id == tenant_id,
            EvaluationRun.project_id == project_id,
        )
    )
    if run is None:
        raise HTTPException(404, "evaluation run not found")
    results = (
        await session.scalars(
            select(EvaluationResult).where(
                EvaluationResult.tenant_id == tenant_id,
                EvaluationResult.project_id == project_id,
                EvaluationResult.payload["run_id"].astext == str(run.id),
            )
        )
    ).all()
    return {
        "id": run.id,
        **run.payload,
        "results": [{"id": row.id, **row.payload} for row in results],
    }


@router.post("/evaluation/runs", status_code=202)
async def admin_create_evaluation_run(
    data: EvaluationRunCreate,
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    dataset = await session.scalar(
        select(EvaluationDataset).where(
            EvaluationDataset.id == data.dataset_id,
            EvaluationDataset.tenant_id == tenant_id,
            EvaluationDataset.project_id == project_id,
        )
    )
    if dataset is None:
        raise HTTPException(404, "evaluation dataset not found")
    run = EvaluationRun(
        tenant_id=tenant_id,
        project_id=project_id,
        payload={
            "dataset_id": str(dataset.id),
            "status": "queued",
            "configuration": data.model_dump(exclude={"dataset_id"}),
        },
    )
    session.add(run)
    await session.flush()
    session.add(
        OutboxEvent(
            tenant_id=tenant_id,
            project_id=project_id,
            payload={"type": "evaluation.run", "run_id": str(run.id), "attempts": 0},
        )
    )
    _audit(
        session,
        tenant_id=tenant_id,
        project_id=project_id,
        action="evaluation.run",
        resource_type="evaluation_run",
        resource_id=run.id,
    )
    await session.commit()
    return {"id": run.id, "status": "queued"}


@router.get("/projects/{project_id}")
async def project(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.get(Project, project_id)
    if row is None:
        raise HTTPException(404, "project not found")
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "slug": row.slug,
        "name": row.name,
        "description": row.description,
        "enabled": row.enabled,
    }


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: uuid.UUID,
    data: ProjectUpdate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.get(Project, project_id)
    if row is None:
        raise HTTPException(404, "project not found")
    for field, value in data.model_dump(exclude_none=True).items():
        setattr(row, field, value)
    _audit(
        session,
        tenant_id=row.tenant_id,
        project_id=row.id,
        action="project.update",
        resource_type="project",
        resource_id=row.id,
    )
    await session.commit()
    return await project(project_id, session)


@router.post("/projects/{project_id}/reconcile", status_code=202)
async def reconcile_project(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    result = await reconcile(session, project.id)
    _audit(
        session,
        tenant_id=project.tenant_id,
        project_id=project.id,
        action="project.reconcile",
        resource_type="project",
        resource_id=project.id,
    )
    await session.commit()
    return result


@router.post("/api-keys", status_code=201)
async def create_key(
    data: ApiKeyCreate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    raw = f"rag_{secrets.token_urlsafe(32)}"
    row = ApiKey(
        tenant_id=data.tenant_id,
        prefix=raw[:12],
        key_hash=hash_key(raw),
        allowed_project_ids=data.allowed_project_ids,
        allowed_collections=data.allowed_collections,
        permissions=data.permissions,
    )
    session.add(row)
    await session.flush()
    _audit(
        session,
        tenant_id=row.tenant_id,
        project_id=None,
        action="api_key.create",
        resource_type="api_key",
        resource_id=row.id,
    )
    await session.commit()
    return {
        "id": str(row.id),
        "api_key": raw,
        "warning": "This value is shown only once.",
    }


@router.get("/api-keys")
async def api_keys(
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    rows = (await session.scalars(select(ApiKey).order_by(ApiKey.created_at.desc()))).all()
    return [
        {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "prefix": row.prefix,
            "allowed_project_ids": row.allowed_project_ids,
            "allowed_collections": row.allowed_collections,
            "permissions": row.permissions,
            "revoked": row.revoked,
            "created_at": row.created_at,
        }
        for row in rows
    ]


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    row = await session.get(ApiKey, key_id)
    if row is None:
        raise HTTPException(404, "API key not found")
    if not row.revoked:
        row.revoked = True
        _audit(
            session,
            tenant_id=row.tenant_id,
            project_id=None,
            action="api_key.revoke",
            resource_type="api_key",
            resource_id=row.id,
        )
        await session.commit()


@router.post("/collections", status_code=201)
async def create_collection(
    data: CollectionCreate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = Collection(**data.model_dump())
    session.add(row)
    await session.flush()
    _audit(
        session,
        tenant_id=row.tenant_id,
        project_id=row.project_id,
        action="collection.create",
        resource_type="collection",
        resource_id=row.id,
    )
    await session.commit()
    return {"id": row.id, "name": row.name, "settings": row.settings}


@router.get("/collections")
async def collections(
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    rows = (await session.scalars(select(Collection))).all()
    return [
        {
            "id": row.id,
            "project_id": row.project_id,
            "name": row.name,
            "settings": row.settings,
        }
        for row in rows
    ]


@router.get("/collections/{collection_id}")
async def collection(
    collection_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.get(Collection, collection_id)
    if row is None:
        raise HTTPException(404, "collection not found")
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "project_id": row.project_id,
        "name": row.name,
        "description": row.description,
        "settings": row.settings,
    }


@router.patch("/collections/{collection_id}")
async def update_collection(
    collection_id: uuid.UUID,
    data: CollectionUpdate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.get(Collection, collection_id)
    if row is None:
        raise HTTPException(404, "collection not found")
    for field, value in data.model_dump(exclude_none=True).items():
        setattr(row, field, value)
    _audit(
        session,
        tenant_id=row.tenant_id,
        project_id=row.project_id,
        action="collection.update",
        resource_type="collection",
        resource_id=row.id,
    )
    await session.commit()
    return await collection(collection_id, session)


@router.post("/collections/{collection_id}/reindex", status_code=202)
async def reindex_admin_collection(
    collection_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    row = await session.get(Collection, collection_id)
    if row is None:
        raise HTTPException(404, "collection not found")
    result = await reindex_collection(session, row.tenant_id, row.project_id, row.name)
    _audit(
        session,
        tenant_id=row.tenant_id,
        project_id=row.project_id,
        action="collection.reindex",
        resource_type="collection",
        resource_id=row.id,
    )
    await session.commit()
    return result


@router.post("/collections/{collection_id}/compare-configurations")
async def compare_collection_configurations(
    collection_id: uuid.UUID,
    data: ConfigurationComparisonRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.get(Collection, collection_id)
    if row is None:
        raise HTTPException(404, "collection not found")
    who = Principal(
        row.tenant_id,
        frozenset({row.project_id}),
        frozenset({row.name}),
        frozenset({"retrieval:search"}),
    )
    comparisons: dict[str, dict[str, object]] = {}
    result_ids: dict[str, set[str]] = {}
    for label, configuration in (("baseline", data.baseline), ("candidate", data.candidate)):
        request = SearchRequest(
            project_id=row.project_id,
            collections=[row.name],
            query=data.query,
            filters=data.filters,
            include_trace=True,
            **configuration.model_dump(),
        )
        request_id, results, trace = await search(session, who, request)
        comparisons[label] = {"request_id": request_id, "results": results, "trace": trace}
        result_ids[label] = {str(result["chunk_id"]) for result in results if "chunk_id" in result}
    baseline_ids = result_ids["baseline"]
    candidate_ids = result_ids["candidate"]
    return {
        **comparisons,
        "overlap": {
            "shared_chunks": len(baseline_ids & candidate_ids),
            "baseline_only": len(baseline_ids - candidate_ids),
            "candidate_only": len(candidate_ids - baseline_ids),
        },
    }


@router.get("/indexing/jobs")
async def indexing_jobs(
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    statement = select(IndexingJob).order_by(IndexingJob.created_at.desc()).limit(limit)
    if status:
        statement = statement.where(IndexingJob.payload["status"].astext == status)
    rows = (await session.scalars(statement)).all()
    return [_indexing_job_read(row) for row in rows]


def _indexing_job_read(row: IndexingJob) -> dict[str, object]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "project_id": row.project_id,
        **row.payload,
    }


def _queue_job_retry(session: AsyncSession, job: IndexingJob) -> bool:
    job_type = job.payload.get("job_type", "document.index")
    identifier_field = "document_id" if job_type == "document.delete" else "version_id"
    identifier = job.payload.get(identifier_field)
    if not isinstance(identifier, str):
        return False
    job.payload = {**job.payload, "status": "queued", "error": None}
    session.add(
        OutboxEvent(
            tenant_id=job.tenant_id,
            project_id=job.project_id,
            payload={
                "type": job_type,
                identifier_field: identifier,
                "job_id": str(job.id),
                "attempts": 0,
            },
        )
    )
    _audit(
        session,
        tenant_id=job.tenant_id,
        project_id=job.project_id,
        action="indexing_job.retry",
        resource_type="indexing_job",
        resource_id=job.id,
    )
    return True


@router.get("/indexing/jobs/{job_id}")
async def indexing_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.get(IndexingJob, job_id)
    if row is None:
        raise HTTPException(404, "indexing job not found")
    return _indexing_job_read(row)


@router.post("/indexing/jobs/{job_id}/retry", status_code=202)
async def retry_indexing_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    job = await session.scalar(
        select(IndexingJob).where(IndexingJob.id == job_id).with_for_update()
    )
    if job is None:
        raise HTTPException(404, "indexing job not found")
    if job.payload.get("status") not in {"failed", "dead_letter"}:
        raise HTTPException(409, "only failed or dead-letter jobs can be retried")
    if not _queue_job_retry(session, job):
        raise HTTPException(409, "indexing job has no target identifier")
    await session.commit()
    return {"id": job.id, "status": "queued"}


@router.post("/indexing/jobs/retry-filtered", status_code=202)
async def retry_filtered_indexing_jobs(
    status: str = Query(default="failed", pattern="^(failed|dead_letter)$"),
    project_id: uuid.UUID | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    statement = (
        select(IndexingJob)
        .where(IndexingJob.payload["status"].astext == status)
        .order_by(IndexingJob.created_at)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    if project_id:
        statement = statement.where(IndexingJob.project_id == project_id)
    jobs = (await session.scalars(statement)).all()
    retried = sum(_queue_job_retry(session, job) for job in jobs)
    await session.commit()
    return {"matched": len(jobs), "retried": retried, "skipped_invalid": len(jobs) - retried}


@router.post("/indexing/jobs/{job_id}/cancel")
async def cancel_indexing_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    job = await session.scalar(
        select(IndexingJob).where(IndexingJob.id == job_id).with_for_update()
    )
    if job is None:
        raise HTTPException(404, "indexing job not found")
    if job.payload.get("status") != "queued":
        raise HTTPException(409, "only queued jobs can be canceled")
    job.payload = {**job.payload, "status": "canceled"}
    _audit(
        session,
        tenant_id=job.tenant_id,
        project_id=job.project_id,
        action="indexing_job.cancel",
        resource_type="indexing_job",
        resource_id=job.id,
    )
    await session.commit()
    return {"id": job.id, "status": "canceled"}


@router.get("/dashboard")
async def dashboard(
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    documents = await session.scalar(select(func.count()).select_from(Document)) or 0
    chunks = await session.scalar(select(func.count()).select_from(Chunk)) or 0
    return {"documents": documents, "chunks": chunks, "embeddings": chunks}


@router.get("/metrics/timeseries")
async def metrics_timeseries(
    metric: str = Query(
        pattern="^(documents|retrieval_requests|feedback|audit_events|indexing_jobs|indexing_errors)$"
    ),
    project_id: uuid.UUID | None = None,
    collection: str | None = Query(default=None, min_length=1, max_length=100),
    from_: datetime | None = Query(default=None, alias="from"),
    to: datetime | None = None,
    step: str = Query(default="hour", pattern="^(hour|day)$"),
    aggregation: str = Query(default="count", pattern="^count$"),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    end = to or datetime.now(UTC)
    start = from_ or end - timedelta(days=1)
    if start >= end:
        raise HTTPException(422, "from must be earlier than to")
    if end - start > timedelta(days=90):
        raise HTTPException(422, "time range cannot exceed 90 days")
    points = await metric_timeseries(
        session,
        metric,
        project_id=project_id,
        collection=collection,
        start=start,
        end=end,
        step=step,
    )
    return {
        "metric": metric,
        "aggregation": aggregation,
        "step": step,
        "from": start,
        "to": end,
        "points": points,
    }


@router.get("/audit-log")
async def audit_log(
    action: str | None = None,
    tenant_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    statement = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    if action:
        statement = statement.where(AuditLog.payload["action"].astext == action)
    if tenant_id:
        statement = statement.where(AuditLog.tenant_id == tenant_id)
    if project_id:
        statement = statement.where(AuditLog.project_id == project_id)
    rows = (await session.scalars(statement)).all()
    return [
        {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "project_id": row.project_id,
            "created_at": row.created_at,
            **row.payload,
        }
        for row in rows
    ]


@router.get("/system/health")
async def health() -> dict[str, object]:
    return await system_health()


@router.get("/system/resources")
async def resources() -> dict[str, object]:
    return system_resources()


@router.get("/settings")
async def settings(
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    return await runtime_settings_response(session)


@router.patch("/settings")
async def update_settings(
    data: RuntimeSettingsUpdate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    updates = data.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(422, "at least one setting is required")
    await apply_runtime_settings(session, updates)
    global_scope = uuid.UUID(int=0)
    _audit(
        session,
        tenant_id=global_scope,
        project_id=None,
        action="runtime_settings.update",
        resource_type="runtime_settings",
        resource_id=global_scope,
    )
    await session.commit()
    return await runtime_settings_response(session)


@router.get("/reranker/status")
async def reranker() -> dict[str, object]:
    return await reranker_status()


@router.post("/reranker/test")
async def test_reranker() -> dict[str, object]:
    return await test_reranker_connection()


@router.get("/models/embeddings")
async def embeddings() -> dict[str, object]:
    return await embedding_profile()


@router.post("/models/embeddings/check")
async def check_embeddings() -> dict[str, object]:
    return await embedding_profile()


@router.post("/models/embeddings/reindex", status_code=202)
async def reindex_embedding_model(
    data: EmbeddingReindexRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    if not data.confirm:
        raise HTTPException(409, "explicit confirmation is required")
    profile = await embedding_profile()
    if profile.get("compatible") is not True:
        raise HTTPException(409, "embedding model is not compatible")
    return await reindex_embeddings(session)


@router.post("/retrieval/search")
async def admin_search(
    data: SearchRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    project = await session.scalar(select(Project).where(Project.id == data.project_id))
    if project is None:
        raise HTTPException(404, "project not found")
    who = Principal(
        project.tenant_id,
        frozenset({project.id}),
        frozenset(data.collections),
        frozenset({"retrieval:search"}),
    )
    request_id, results, trace = await search(session, who, data)
    return {"request_id": request_id, "results": results, "trace": trace}


def _retrieval_trace_read(row: RetrievalRequest) -> dict[str, object]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "project_id": row.project_id,
        "created_at": row.created_at,
        **row.payload,
    }


@router.get("/retrieval/traces")
async def retrieval_traces(
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    rows = (
        await session.scalars(
            select(RetrievalRequest).order_by(RetrievalRequest.created_at.desc()).limit(limit)
        )
    ).all()
    return [_retrieval_trace_read(row) for row in rows]


@router.get("/retrieval/traces/{request_id}")
async def retrieval_trace(
    request_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.get(RetrievalRequest, request_id)
    if row is None:
        raise HTTPException(404, "retrieval trace not found")
    return _retrieval_trace_read(row)


@router.post("/retrieval/traces/{request_id}/repeat", status_code=202)
async def repeat_retrieval_trace(
    request_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.get(RetrievalRequest, request_id)
    if row is None or row.project_id is None:
        raise HTTPException(404, "retrieval trace not found")
    configuration = row.payload.get("configuration", {})
    if not isinstance(configuration, dict):
        raise HTTPException(409, "retrieval trace configuration is invalid")
    collections = row.payload.get("collections", [])
    if not isinstance(collections, list) or not all(isinstance(item, str) for item in collections):
        raise HTTPException(409, "retrieval trace collections are invalid")
    query = row.payload.get("query")
    if not isinstance(query, str):
        raise HTTPException(409, "retrieval trace query is invalid")
    who = Principal(
        row.tenant_id,
        frozenset({row.project_id}),
        frozenset(collections),
        frozenset({"retrieval:search"}),
    )
    data = SearchRequest(
        project_id=row.project_id,
        collections=collections,
        query=query,
        filters=row.payload.get("filters", {}),
        vector_top_k=int(configuration.get("vector_top_k", 30)),
        bm25_top_k=int(configuration.get("bm25_top_k", 30)),
        fusion_top_k=int(configuration.get("fusion_top_k", 20)),
        rerank_top_k=int(configuration.get("rerank_top_k", 5)),
        use_reranker=bool(configuration.get("use_reranker", True)),
        include_trace=True,
    )
    repeated_id, results, trace = await search(session, who, data)
    return {"request_id": repeated_id, "results": results, "trace": trace}
