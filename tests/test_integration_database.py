import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from rag_platform.api.schemas import DocumentCreate
from rag_platform.core.auth import Principal
from rag_platform.db.models import Collection, DocumentVersion, Project, Tenant
from rag_platform.services.documents import ingest

TEST_DATABASE_URL = "postgresql+asyncpg://rag:rag-local-password@127.0.0.1:55432/rag"


@pytest.mark.asyncio
async def test_ingestion_is_idempotent_and_advances_current_version() -> None:
    engine = create_async_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    tenant_id = uuid.uuid4()
    project_id = uuid.uuid4()
    external_id = f"integration-{uuid.uuid4()}"
    async with session_factory() as session:
        session.add(Tenant(id=tenant_id, name=f"tenant-{tenant_id}"))
        await session.flush()
        session.add(
            Project(
                id=project_id,
                tenant_id=tenant_id,
                slug=f"project-{project_id}",
                name="Integration project",
            )
        )
        await session.flush()
        session.add(
            Collection(
                tenant_id=tenant_id,
                project_id=project_id,
                name="documents",
            )
        )
        await session.commit()
        principal = Principal(
            tenant_id,
            frozenset({project_id}),
            frozenset({"documents"}),
            frozenset({"documents:write"}),
        )
        first_payload = DocumentCreate(
            project_id=project_id,
            collection="documents",
            external_document_id=external_id,
            content="First stable content",
            version=1,
        )
        missing_payload = first_payload.model_copy(update={"collection": "missing"})
        missing_principal = Principal(
            tenant_id,
            frozenset({project_id}),
            frozenset({"documents", "missing"}),
            frozenset({"documents:write"}),
        )
        with pytest.raises(ValueError, match="collection not found"):
            await ingest(session, missing_principal, missing_payload)
        first = await ingest(session, principal, first_payload)
        duplicate = await ingest(session, principal, first_payload)
        assert duplicate.id == first.id
        count = await session.scalar(
            select(func.count())
            .select_from(DocumentVersion)
            .where(DocumentVersion.external_document_id == external_id)
        )
        assert count == 1

        second = await ingest(
            session,
            principal,
            first_payload.model_copy(update={"content": "Second stable content", "version": 2}),
        )
        await session.refresh(first)
        assert second.is_current is True
        assert first.is_current is False
    await engine.dispose()
