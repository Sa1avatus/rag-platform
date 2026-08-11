import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from rag_platform.api.schemas import DocumentCreate, SearchRequest
from rag_platform.core.auth import Principal
from rag_platform.db.models import Chunk, Collection, Document, DocumentVersion, Project, Tenant
from rag_platform.services.documents import ingest
from rag_platform.services.retrieval import scoped_statement

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


@pytest.mark.asyncio
async def test_retrieval_scope_excludes_historical_document_versions() -> None:
    engine = create_async_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    tenant_id = uuid.uuid4()
    project_id = uuid.uuid4()
    document_id = uuid.uuid4()
    async with session_factory() as session:
        session.add(Tenant(id=tenant_id, name=f"tenant-{tenant_id}"))
        await session.flush()
        session.add(
            Project(
                id=project_id,
                tenant_id=tenant_id,
                slug=f"project-{project_id}",
                name="Version isolation project",
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
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            project_id=project_id,
            collection="documents",
            external_document_id=f"versioned-{uuid.uuid4()}",
            current_version=2,
        )
        session.add(document)
        await session.flush()
        historical = DocumentVersion(
            document_id=document_id,
            tenant_id=tenant_id,
            project_id=project_id,
            collection="documents",
            external_document_id=document.external_document_id,
            document_type="text",
            content="Historical searchable content",
            content_hash="1" * 64,
            version=1,
            is_current=False,
        )
        current = DocumentVersion(
            document_id=document_id,
            tenant_id=tenant_id,
            project_id=project_id,
            collection="documents",
            external_document_id=document.external_document_id,
            document_type="text",
            content="Current searchable content",
            content_hash="2" * 64,
            version=2,
            is_current=True,
        )
        session.add_all([historical, current])
        await session.flush()
        for index, version in enumerate((historical, current)):
            session.add(
                Chunk(
                    document_id=document_id,
                    document_version_id=version.id,
                    tenant_id=tenant_id,
                    project_id=project_id,
                    collection="documents",
                    chunk_index=index,
                    content=version.content,
                    token_count=3,
                    language="en",
                    content_hash=str(index + 3) * 64,
                    embedding_model="BAAI/bge-m3",
                    embedding_dimension=1024,
                )
            )
        await session.commit()

        who = Principal(
            tenant_id,
            frozenset({project_id}),
            frozenset({"documents"}),
            frozenset({"retrieval:search"}),
        )
        rows = (
            await session.execute(
                scoped_statement(
                    who,
                    SearchRequest(
                        project_id=project_id,
                        collections=["documents"],
                        query="searchable content",
                    ),
                )
            )
        ).all()

        assert [row[0].document_version_id for row in rows] == [current.id]
    await engine.dispose()
