import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    project_id: uuid.UUID
    collection: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,99}$")
    external_document_id: str = Field(min_length=1, max_length=300)
    document_type: str = "text"
    title: str = ""
    content: str = Field(min_length=1)
    language: str = "und"
    version: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRead(BaseModel):
    id: uuid.UUID
    external_document_id: str
    version: int
    status: str
    content_hash: str


class DocumentBatchCreate(BaseModel):
    documents: list[DocumentCreate] = Field(min_length=1, max_length=100)


class DocumentUpdate(BaseModel):
    expected_lock_version: int = Field(ge=1)
    content: str = Field(min_length=1)
    title: str | None = None
    document_type: str | None = None
    language: str | None = None
    metadata: dict[str, Any] | None = None


class UploadRead(BaseModel):
    documents: list[DocumentRead]
    source_object_key: str


class SearchRequest(BaseModel):
    project_id: uuid.UUID
    collections: list[str] = Field(min_length=1)
    query: str = Field(min_length=1, max_length=10_000)
    filters: dict[str, Any] = Field(default_factory=dict)
    vector_top_k: int = Field(default=30, ge=1, le=200)
    bm25_top_k: int = Field(default=30, ge=1, le=200)
    fusion_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    use_reranker: bool = True
    include_parent_content: bool = True
    include_trace: bool = False


class SearchResult(BaseModel):
    request_id: uuid.UUID
    results: list[dict[str, Any]]
    trace: dict[str, Any] | None = None


class FeedbackCreate(BaseModel):
    project_id: uuid.UUID
    request_id: uuid.UUID
    chunk_id: uuid.UUID
    relevant: bool
    relevance_grade: int | None = Field(default=None, ge=0, le=3)
    comment: str = Field(default="", max_length=2000)


class ProjectCreate(BaseModel):
    tenant_id: uuid.UUID
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,99}$")
    name: str
    description: str = ""


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    enabled: bool | None = None


class TenantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ApiKeyCreate(BaseModel):
    tenant_id: uuid.UUID
    allowed_project_ids: list[uuid.UUID]
    allowed_collections: list[str]
    permissions: list[str]


class CollectionCreate(BaseModel):
    tenant_id: uuid.UUID
    project_id: uuid.UUID
    name: str
    description: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)


class CollectionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None
    settings: dict[str, Any] | None = None


class RetrievalConfiguration(BaseModel):
    vector_top_k: int = Field(default=30, ge=1, le=200)
    bm25_top_k: int = Field(default=30, ge=1, le=200)
    fusion_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    use_reranker: bool = True
    include_parent_content: bool = True


class ConfigurationComparisonRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    filters: dict[str, Any] = Field(default_factory=dict)
    baseline: RetrievalConfiguration
    candidate: RetrievalConfiguration


class RuntimeSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_vector_top_k: int | None = Field(default=None, ge=1, le=200)
    default_bm25_top_k: int | None = Field(default=None, ge=1, le=200)
    default_fusion_top_k: int | None = Field(default=None, ge=1, le=100)
    default_rerank_top_k: int | None = Field(default=None, ge=1, le=50)
    reranker_enabled: bool | None = None
    query_normalization_enabled: bool | None = None
    query_expansion_enabled: bool | None = None
    parent_content_enabled: bool | None = None
    indexing_concurrency: int | None = Field(default=None, ge=1, le=64)
    embedding_batch_size: int | None = Field(default=None, ge=1, le=256)
    document_max_bytes: int | None = Field(default=None, ge=1_048_576, le=104_857_600)
    trace_retention_days: int | None = Field(default=None, ge=1, le=3650)
    completed_job_retention_days: int | None = Field(default=None, ge=1, le=3650)


class EmbeddingReindexRequest(BaseModel):
    confirm: bool


class ContextResponse(BaseModel):
    request_id: uuid.UUID
    sources: list[dict[str, Any]]
    context: str


class EvaluationCaseInput(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    filters: dict[str, Any] = Field(default_factory=dict)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    relevance_grades: dict[str, int] = Field(default_factory=dict)
    forbidden_results: list[str] = Field(default_factory=list)


class EvaluationDatasetCreate(BaseModel):
    project_id: uuid.UUID
    name: str = Field(min_length=1, max_length=200)
    version: int = Field(default=1, ge=1)
    collections: list[str] = Field(min_length=1)
    cases: list[EvaluationCaseInput] = Field(min_length=1)


class EvaluationRunCreate(BaseModel):
    dataset_id: uuid.UUID
    vector_top_k: int = Field(default=30, ge=1, le=200)
    bm25_top_k: int = Field(default=30, ge=1, le=200)
    fusion_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    use_reranker: bool = True
