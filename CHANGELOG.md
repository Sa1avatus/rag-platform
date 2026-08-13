# Changelog

## 0004 — Owner-scoped multi-user isolation

**Breaking change:** every service API request (`/v1/documents`, `/v1/retrieval`, `/v1/feedback`,
`/v1/evaluations`) now requires the `X-Owner-User-Id` header with a valid UUID. Requests without the
header are rejected with 400. The admin API (`/v1/admin/*`) is unaffected.

### Database

- Added `owner_user_id UUID NOT NULL` to `documents`, `document_versions`, `chunks`.
- Existing rows are backfilled with sentinel `00000000-0000-0000-0000-000000000000`.
- Unique constraint on `documents` updated to
  `(tenant_id, project_id, collection, owner_user_id, external_document_id)`.
- `ix_documents_scope` index updated to include `owner_user_id`.
- Alembic migration: `0004_owner_user_id`.

### Retrieval

- Vector search (pgvector cosine), BM25 search (OpenSearch), hybrid RRF fusion, and reranker input
  are all filtered by `owner_user_id` inside the storage layer — before candidate ranking.
- OpenSearch index mapping includes `owner_user_id` as a `keyword` field.
- Failure to apply the owner filter is fail-closed: retrieval returns an error, never unfiltered data.

### Ingestion

- `Document`, `DocumentVersion`, and `Chunk` records store `owner_user_id` from the request principal.
- `stable_document_id()` includes `owner_user_id` in the UUID5 identity — two users can have the
  same `external_document_id` without collision.
- Background indexing and deletion tasks propagate `owner_user_id` from the version/document record.

### Admin API

- `GET /v1/admin/documents` and `GET /v1/admin/documents/{id}/chunks` accept optional
  `owner_user_id` query parameter. When omitted, admin sees all owners.

### Cache

- Query embedding cache is owner-agnostic (same text → same vector). No changes needed.
- Retrieval result cache does not exist; if added, keys must include `owner_user_id`.

### Migration guide

1. Run `alembic upgrade head` (migration `0004`).
2. Update all service API clients to send `X-Owner-User-Id: ` header.
3. Optionally reindex collections via `POST /v1/admin/collections/{id}/reindex` to populate
   `owner_user_id` in the OpenSearch BM25 index.

### Files changed

```
src/rag_platform/core/auth.py
src/rag_platform/db/models.py
src/rag_platform/services/versioning.py
src/rag_platform/services/documents.py
src/rag_platform/services/retrieval.py
src/rag_platform/services/vector_search.py
src/rag_platform/services/opensearch.py
src/rag_platform/api/routes/documents.py
src/rag_platform/api/routes/admin.py
src/rag_platform/worker/evaluation.py
src/rag_platform/worker/tasks.py
alembic/versions/0004_owner_user_id.py
tests/test_owner_isolation.py
tests/test_admin_routes.py
tests/test_evaluation_feedback_routes.py
tests/test_e2e_pipeline.py
tests/test_versioning.py
tests/test_retrieval.py
tests/test_core.py
README.md
API.md
SECURITY.md
ARCHITECTURE.md
OPERATIONS.md
CHANGELOG.md
AGENTS.md
```
