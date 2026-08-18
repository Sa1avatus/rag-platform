import secrets
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from prometheus_client import REGISTRY
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import (
    ApiKeyCreate,
    ApiKeyUpdate,
    CacheClearRequest,
    CollectionCreate,
    CollectionUpdate,
    ConfigurationComparisonRequest,
    DocumentActionRequest,
    EmbeddingReindexRequest,
    EvaluationRunComparisonRequest,
    EvaluationRunCreate,
    ProjectCreate,
    ProjectUpdate,
    RuntimeSettingsUpdate,
    SearchRequest,
    TenantCreate,
)
from rag_platform.core.auth import Principal, admin, hash_key
from rag_platform.core.config import get_settings
from rag_platform.db.models import (
    ApiKey,
    AuditLog,
    Chunk,
    Collection,
    Document,
    DocumentVersion,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    IndexingJob,
    OutboxEvent,
    Project,
    RetrievalFeedback,
    RetrievalRequest,
    Status,
    Tenant,
)
from rag_platform.db.session import get_session
from rag_platform.services.admin_metrics import metric_timeseries
from rag_platform.services.cache import CacheUnavailable, clear_rag_cache
from rag_platform.services.documents import enqueue_deletion, enqueue_indexing
from rag_platform.services.embedding_admin import all_models_status, embedding_profile
from rag_platform.services.evaluation_metrics import pin_retrieval_configuration
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


@router.get("/tenants")
async def list_tenants(
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    rows = (await session.scalars(select(Tenant).order_by(Tenant.name))).all()
    return [{"id": row.id, "name": row.name} for row in rows]


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
    tenant = await session.scalar(select(Tenant).where(Tenant.id == data.tenant_id))
    if tenant is None:
        raise HTTPException(404, "tenant not found")
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
    owner_user_id: uuid.UUID | None = None,
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
    if owner_user_id is not None:
        filters.append(Document.owner_user_id == owner_user_id)
    rows = (
        await session.scalars(
            select(Document)
            .where(*filters)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    doc_ids = [row.id for row in rows]
    # Chunk counts per document
    chunk_counts: dict[uuid.UUID, int] = {}
    if doc_ids:
        count_rows = (
            await session.execute(
                select(Chunk.document_id, func.count(Chunk.id))
                .where(Chunk.document_id.in_(doc_ids))
                .group_by(Chunk.document_id)
            )
        ).all()
        chunk_counts = {row[0]: row[1] for row in count_rows}
    # Latest version content per document
    version_map: dict[uuid.UUID, DocumentVersion] = {}
    if doc_ids:
        latest_versions = (
            await session.scalars(
                select(DocumentVersion)
                .where(DocumentVersion.document_id.in_(doc_ids))
                .order_by(DocumentVersion.document_id, DocumentVersion.version.desc())
            )
        ).all()
        for v in latest_versions:
            if v.document_id not in version_map:
                version_map[v.document_id] = v
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
            "chunk_count": chunk_counts.get(row.id, 0),
            "title": (v := version_map.get(row.id)) and v.title or "",
            "content": (v := version_map.get(row.id)) and v.content[:2000] or "",
            "document_type": (v := version_map.get(row.id)) and v.document_type or "",
        }
        for row in rows
    ]


@router.get("/documents/{document_id}/chunks")
async def admin_document_chunks(
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    collection: str,
    owner_user_id: uuid.UUID | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    doc_filters = [
        Document.id == document_id,
        Document.tenant_id == tenant_id,
        Document.project_id == project_id,
        Document.collection == collection,
        Document.deleted_at.is_(None),
    ]
    if owner_user_id is not None:
        doc_filters.append(Document.owner_user_id == owner_user_id)
    document = await session.scalar(
        select(Document).where(*doc_filters)
    )
    if document is None:
        raise HTTPException(404, "document not found")
    rows = (
        await session.scalars(
            select(Chunk)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .where(
                Chunk.document_id == document_id,
                Chunk.tenant_id == tenant_id,
                Chunk.project_id == project_id,
                Chunk.collection == collection,
                DocumentVersion.is_current.is_(True),
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
            "source_type": row.source_type,
            "source_id": row.source_id,
            "section_title": row.section_title,
            "start_offset": row.start_offset,
            "end_offset": row.end_offset,
            "chunker_version": row.chunker_version,
            "index_version": row.index_version,
        }
        for row in rows
    ]


@router.post("/documents/{document_id}/reindex", status_code=202)
async def admin_reindex_document(
    document_id: uuid.UUID,
    data: DocumentActionRequest,
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    collection: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    if not data.confirm:
        raise HTTPException(409, "explicit confirmation is required")
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
    version = await session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document.id,
            DocumentVersion.tenant_id == tenant_id,
            DocumentVersion.project_id == project_id,
            DocumentVersion.collection == collection,
            DocumentVersion.is_current.is_(True),
        )
    )
    if version is None:
        raise HTTPException(409, "current document version is missing")
    version.status = Status.queued
    version.error = None
    job = await enqueue_indexing(session, version)
    _audit(
        session,
        tenant_id=tenant_id,
        project_id=project_id,
        action="document.reindex",
        resource_type="document",
        resource_id=document.id,
    )
    await session.commit()
    return {"job_id": job.id, "status": "queued"}


@router.post("/documents/{document_id}/delete", status_code=202)
async def admin_delete_document(
    document_id: uuid.UUID,
    data: DocumentActionRequest,
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    collection: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    if not data.confirm:
        raise HTTPException(409, "explicit confirmation is required")
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
    job = await enqueue_deletion(session, document)
    _audit(
        session,
        tenant_id=tenant_id,
        project_id=project_id,
        action="document.delete",
        resource_type="document",
        resource_id=document.id,
    )
    await session.commit()
    return {"job_id": job.id, "status": "queued"}


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


def _comparison_metrics(payload: dict[str, object]) -> dict[str, float]:
    raw = payload.get("metrics_after_reranking")
    if not isinstance(raw, dict):
        raw = payload.get("metrics_before_reranking")
    if not isinstance(raw, dict):
        return {}
    return {
        str(name): float(value)
        for name, value in raw.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


@router.post("/evaluation/compare")
async def admin_compare_evaluation_runs(
    data: EvaluationRunComparisonRequest,
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    async def scoped_run(run_id: uuid.UUID) -> EvaluationRun:
        run = await session.scalar(
            select(EvaluationRun).where(
                EvaluationRun.id == run_id,
                EvaluationRun.tenant_id == tenant_id,
                EvaluationRun.project_id == project_id,
            )
        )
        if run is None:
            raise HTTPException(404, "evaluation run not found")
        if run.payload.get("status") != "completed":
            raise HTTPException(409, "evaluation runs must be completed before comparison")
        return run

    baseline = await scoped_run(data.baseline_run_id)
    candidate = await scoped_run(data.candidate_run_id)
    baseline_metrics = _comparison_metrics(baseline.payload)
    candidate_metrics = _comparison_metrics(candidate.payload)
    metric_names = sorted(baseline_metrics.keys() | candidate_metrics.keys())
    return {
        "baseline": {"id": baseline.id, "configuration": baseline.payload.get("configuration", {})},
        "candidate": {
            "id": candidate.id,
            "configuration": candidate.payload.get("configuration", {}),
        },
        "comparison": [
            {
                "metric": name,
                "baseline": baseline_metrics.get(name, 0.0),
                "candidate": candidate_metrics.get(name, 0.0),
                "delta": candidate_metrics.get(name, 0.0) - baseline_metrics.get(name, 0.0),
            }
            for name in metric_names
        ],
    }


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


@router.get("/feedback")
async def admin_feedback(
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    relevant: bool | None = None,
    collection: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    filters = [
        RetrievalFeedback.tenant_id == tenant_id,
        RetrievalFeedback.project_id == project_id,
    ]
    if relevant is not None:
        filters.append(RetrievalFeedback.payload["relevant"].astext == str(relevant).lower())
    if collection:
        filters.append(RetrievalFeedback.payload["collection"].astext == collection)
    rows = (
        await session.scalars(
            select(RetrievalFeedback)
            .where(*filters)
            .order_by(RetrievalFeedback.created_at.desc())
            .limit(limit)
        )
    ).all()
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


@router.patch("/api-keys/{key_id}")
async def update_api_key(
    key_id: uuid.UUID,
    data: ApiKeyUpdate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = await session.get(ApiKey, key_id)
    if row is None:
        raise HTTPException(404, "API key not found")
    if row.revoked:
        raise HTTPException(409, "Revoked API key cannot be updated")
    allowed_collections = list(dict.fromkeys(data.allowed_collections))
    collections = (
        await session.scalars(
            select(Collection).where(
                Collection.tenant_id == row.tenant_id,
                Collection.project_id.in_(row.allowed_project_ids),
                Collection.name.in_(allowed_collections),
            )
        )
    ).all()
    registered_names = {collection.name for collection in collections}
    unknown_names = sorted(set(allowed_collections) - registered_names)
    if unknown_names:
        raise HTTPException(
            422,
            f"Collections are not registered in an allowed project: {', '.join(unknown_names)}",
        )
    row.allowed_collections = allowed_collections
    _audit(
        session,
        tenant_id=row.tenant_id,
        project_id=None,
        action="api_key.authorization.update",
        resource_type="api_key",
        resource_id=row.id,
    )
    await session.commit()
    return {
        "id": row.id,
        "prefix": row.prefix,
        "allowed_project_ids": row.allowed_project_ids,
        "allowed_collections": row.allowed_collections,
        "permissions": row.permissions,
        "revoked": row.revoked,
    }


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
        uuid.UUID(int=0),
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


def _metric_value(name: str, labels: dict[str, str] | None = None) -> float:
    return float(REGISTRY.get_sample_value(name, labels) or 0.0)


@router.get("/dashboard")
async def dashboard(
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    documents = (
        await session.scalar(
            select(func.count()).select_from(Document).where(Document.deleted_at.is_(None))
        )
        or 0
    )
    chunks = (
        await session.scalar(
            select(func.count())
            .select_from(Chunk)
            .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                DocumentVersion.is_current.is_(True),
                Document.deleted_at.is_(None),
            )
        )
        or 0
    )
    recent_failures = (
        await session.scalar(
            select(func.count())
            .select_from(IndexingJob)
            .where(
                IndexingJob.payload["status"].astext.in_(["failed", "dead_letter"]),
                IndexingJob.created_at >= datetime.now(UTC) - timedelta(hours=24),
            )
        )
        or 0
    )
    retrieval_count = sum(
        _metric_value("rag_retrieval_latency_seconds_count", {"mode": mode})
        for mode in ("lexical", "dense", "hybrid")
    )
    retrieval_sum = sum(
        _metric_value("rag_retrieval_latency_seconds_sum", {"mode": mode})
        for mode in ("lexical", "dense", "hybrid")
    )
    reranker_count = _metric_value("rag_reranker_latency_seconds_count")
    reranker_sum = _metric_value("rag_reranker_latency_seconds_sum")
    cache_labels = {"cache": "query_embedding"}
    cache_hits = _metric_value("rag_cache_hits_total", cache_labels)
    cache_misses = _metric_value("rag_cache_misses_total", cache_labels)
    health_snapshot, model = await system_health(), await embedding_profile()
    settings = get_settings()
    return {
        "documents": documents,
        "chunks": chunks,
        "recent_indexing_failures": recent_failures,
        "active_index": settings.index_version,
        "embedding": model,
        "health": health_snapshot,
        "retrieval_latency_ms": (
            round(retrieval_sum / retrieval_count * 1000, 2) if retrieval_count else None
        ),
        "reranker_latency_ms": (
            round(reranker_sum / reranker_count * 1000, 2) if reranker_count else None
        ),
        "cache_hit_rate": (
            round(cache_hits / (cache_hits + cache_misses), 4)
            if cache_hits + cache_misses
            else None
        ),
    }


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


@router.post("/cache/clear")
async def clear_cache(
    data: CacheClearRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    if not data.confirm:
        raise HTTPException(409, "explicit confirmation is required")
    try:
        deleted = await clear_rag_cache()
    except CacheUnavailable as exc:
        raise HTTPException(503, "RAG cache is unavailable") from exc
    global_scope = uuid.UUID(int=0)
    _audit(
        session,
        tenant_id=global_scope,
        project_id=None,
        action="cache.clear",
        resource_type="cache",
        resource_id=global_scope,
    )
    await session.commit()
    return {"status": "cleared", "deleted_keys": deleted}


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
    """Return all registered embedding models with their status."""
    models = await all_models_status()
    active = await embedding_profile()
    return {"active": active, "models": models}


@router.post("/models/embeddings/check")
async def check_embeddings() -> dict[str, object]:
    return await embedding_profile()


@router.post("/models/embeddings/activate", status_code=200)
async def activate_embedding_model(
    model_id: str = Body(..., embed=True),
) -> dict[str, object]:
    """Switch the active embedding model."""
    from rag_platform.core.embedding_registry import get_model_by_id

    try:
        cfg = get_model_by_id(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    # Validate: model must be enabled.
    if not cfg.enabled:
        raise HTTPException(409, f"Model {model_id} is disabled")
    # Update settings in-memory (runtime switch).
    settings = get_settings()
    settings.active_embedding_model = cfg.id
    settings.embedding_model = cfg.model_name
    settings.embedding_dimension = cfg.dimension
    settings.embedding_normalization = cfg.normalization
    settings.index_version = cfg.index_version
    # Persist to Redis so all containers pick it up.
    from rag_platform.core.embedding_registry import set_active_model_in_redis

    set_active_model_in_redis(cfg.id, cfg)
    return {
        "activated": cfg.id,
        "model": cfg.model_name,
        "dimension": cfg.dimension,
        "index_version": cfg.index_version,
    }


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
        uuid.UUID(int=0),
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
        uuid.UUID(int=0),
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
