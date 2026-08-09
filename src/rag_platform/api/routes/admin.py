import secrets
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.api.schemas import (
    ApiKeyCreate,
    CollectionCreate,
    CollectionUpdate,
    ProjectCreate,
    ProjectUpdate,
    SearchRequest,
    TenantCreate,
)
from rag_platform.core.auth import Principal, admin, hash_key
from rag_platform.db.models import (
    ApiKey,
    Chunk,
    Collection,
    Document,
    IndexingJob,
    OutboxEvent,
    Project,
    RetrievalRequest,
    Tenant,
)
from rag_platform.db.session import get_session
from rag_platform.services.health import system_health
from rag_platform.services.reconciliation import reconcile, reindex_collection
from rag_platform.services.reranker import reranker_status, test_reranker_connection
from rag_platform.services.retrieval import search

router = APIRouter(
    prefix="/v1/admin",
    tags=["admin"],
    dependencies=[Depends(admin)],
)


@router.post("/tenants", status_code=201)
async def create_tenant(
    data: TenantCreate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = Tenant(name=data.name)
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return {"id": row.id, "name": row.name}


@router.post("/projects", status_code=201)
async def create_project(
    data: ProjectCreate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = Project(**data.model_dump())
    session.add(row)
    await session.commit()
    await session.refresh(row)
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
    return await reconcile(session, project.id)


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
        await session.commit()


@router.post("/collections", status_code=201)
async def create_collection(
    data: CollectionCreate,
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    row = Collection(**data.model_dump())
    session.add(row)
    await session.commit()
    await session.refresh(row)
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
    return await reindex_collection(session, row.tenant_id, row.project_id, row.name)


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
    job_type = job.payload.get("job_type", "document.index")
    identifier_field = "document_id" if job_type == "document.delete" else "version_id"
    identifier = job.payload.get(identifier_field)
    if not isinstance(identifier, str):
        raise HTTPException(409, "indexing job has no target identifier")
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
    await session.commit()
    return {"id": job.id, "status": "queued"}


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
    await session.commit()
    return {"id": job.id, "status": "canceled"}


@router.get("/dashboard")
async def dashboard(
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    documents = await session.scalar(select(func.count()).select_from(Document)) or 0
    chunks = await session.scalar(select(func.count()).select_from(Chunk)) or 0
    return {"documents": documents, "chunks": chunks, "embeddings": chunks}


@router.get("/system/health")
async def health() -> dict[str, object]:
    return await system_health()


@router.get("/settings")
async def settings() -> dict[str, object]:
    return {
        "default_vector_top_k": 30,
        "default_bm25_top_k": 30,
        "default_fusion_top_k": 20,
        "reranker_enabled": True,
    }


@router.get("/reranker/status")
async def reranker() -> dict[str, object]:
    return await reranker_status()


@router.post("/reranker/test")
async def test_reranker() -> dict[str, object]:
    return await test_reranker_connection()


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
