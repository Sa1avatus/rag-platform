import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Status(StrEnum):
    received = "received"
    queued = "queued"
    processing = "processing"
    indexed = "indexed"
    partially_indexed = "partially_indexed"
    failed = "failed"
    deleted = "deleted"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True)


class Project(Base, TimestampMixin):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("tenant_id", "slug"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    slug: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class ApiKey(Base, TimestampMixin):
    __tablename__ = "api_keys"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    allowed_project_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)))
    allowed_collections: Mapped[list[str]] = mapped_column(ARRAY(String))
    permissions: Mapped[list[str]] = mapped_column(ARRAY(String))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class Collection(Base, TimestampMixin):
    __tablename__ = "collections"
    __table_args__ = (UniqueConstraint("tenant_id", "project_id", "name"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class RuntimeSetting(Base, TimestampMixin):
    __tablename__ = "runtime_settings"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB)


class Document(Base, TimestampMixin):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "project_id",
            "collection",
            "owner_user_id",
            "external_document_id",
            name="uq_documents_scope_external_id",
        ),
        Index("ix_documents_scope", "tenant_id", "project_id", "collection", "owner_user_id"),
        Index("ix_documents_metadata", "metadata", postgresql_using="gin"),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    owner_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    collection: Mapped[str] = mapped_column(String(100), index=True)
    external_document_id: Mapped[str] = mapped_column(String(300), index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    lock_version: Mapped[int] = mapped_column(Integer, default=1)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentVersion(Base, TimestampMixin):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "project_id", "collection", "external_document_id", "version"
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    owner_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    collection: Mapped[str] = mapped_column(String(100), index=True)
    external_document_id: Mapped[str] = mapped_column(String(300), index=True)
    document_type: Mapped[str] = mapped_column(String(100), index=True)
    title: Mapped[str] = mapped_column(String(500), default="")
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    language: Mapped[str] = mapped_column(String(16), default="und")
    version: Mapped[int] = mapped_column(Integer)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    parser_version: Mapped[str] = mapped_column(
        String(100), default="text-v1", server_default="text-v1"
    )
    chunker_version: Mapped[str] = mapped_column(
        String(100), default="word-window-v1", server_default="word-window-v1"
    )
    embedding_model: Mapped[str] = mapped_column(
        String(300), default="BAAI/bge-m3", server_default="BAAI/bge-m3"
    )
    embedding_revision: Mapped[str] = mapped_column(
        String(200), default="default", server_default="default"
    )
    index_version: Mapped[str] = mapped_column(
        String(100), default="rag-chunks-v1", server_default="rag-chunks-v1"
    )
    status: Mapped[Status] = mapped_column(Enum(Status), default=Status.received)
    error: Mapped[str | None] = mapped_column(Text)
    document: Mapped[Document] = relationship(back_populates="versions")


class DocumentBlob(Base, TimestampMixin):
    __tablename__ = "document_blobs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), unique=True
    )
    object_key: Mapped[str] = mapped_column(String(1000))
    mime_type: Mapped[str] = mapped_column(String(200))
    size_bytes: Mapped[int] = mapped_column(Integer)


class Chunk(Base, TimestampMixin):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_version_id", "content_hash", "chunk_index"),
        Index("ix_chunks_metadata", "metadata", postgresql_using="gin"),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    owner_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    collection: Mapped[str] = mapped_column(String(100), index=True)
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL")
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_type: Mapped[str] = mapped_column(String(40), default="child")
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(16))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    source_type: Mapped[str] = mapped_column(String(100), default="text", server_default="text")
    source_id: Mapped[str] = mapped_column(String(300), default="", server_default="")
    section_title: Mapped[str | None] = mapped_column(String(500))
    start_offset: Mapped[int | None] = mapped_column(Integer)
    end_offset: Mapped[int | None] = mapped_column(Integer)
    chunker_version: Mapped[str] = mapped_column(
        String(100), default="word-window-v1", server_default="word-window-v1"
    )
    index_version: Mapped[str] = mapped_column(
        String(100), default="rag-chunks-v1", server_default="rag-chunks-v1"
    )
    embedding_model: Mapped[str] = mapped_column(String(300))
    embedding_dimension: Mapped[int] = mapped_column(Integer)


class ChunkEmbedding(Base, TimestampMixin):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "model",
            "model_revision",
            name="uq_chunk_embeddings_identity",
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), index=True
    )
    model: Mapped[str] = mapped_column(String(300))
    model_revision: Mapped[str] = mapped_column(String(200), default="default")
    backend: Mapped[str] = mapped_column(
        String(100), default="sentence-transformers", server_default="sentence-transformers"
    )
    normalization: Mapped[str] = mapped_column(String(40), default="l2", server_default="l2")
    embedding_dimension: Mapped[int] = mapped_column(Integer, default=1024, server_default="1024")
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))


class EventRecord(Base, TimestampMixin):
    __abstract__ = True
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    project_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class IndexingJob(EventRecord):
    __tablename__ = "indexing_jobs"


class RetrievalRequest(EventRecord):
    __tablename__ = "retrieval_requests"


class RetrievalCandidate(EventRecord):
    __tablename__ = "retrieval_candidates"


class RetrievalResult(EventRecord):
    __tablename__ = "retrieval_results"


class RetrievalFeedback(EventRecord):
    __tablename__ = "retrieval_feedback"


class EvaluationDataset(EventRecord):
    __tablename__ = "evaluation_datasets"


class EvaluationCase(EventRecord):
    __tablename__ = "evaluation_cases"


class EvaluationRun(EventRecord):
    __tablename__ = "evaluation_runs"


class EvaluationResult(EventRecord):
    __tablename__ = "evaluation_results"


class OutboxEvent(EventRecord):
    __tablename__ = "outbox_events"


class AuditLog(EventRecord):
    __tablename__ = "audit_log"
