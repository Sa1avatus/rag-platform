# Changelog

All notable changes to RAG Platform are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The repository had no historical Git tags. The prerelease entries below were reconstructed from
dated `master`-branch milestones so that the initial implementation history remains reviewable.
`0.1.0` is the current package version declared in `pyproject.toml`.

## [Unreleased]

### Fixed

- Fixed `503 vector search is unavailable` errors under concurrent load caused by
  Celery's JSON result backend corrupting large embedding vectors (1024-dim) when
  multiple search and indexing tasks run simultaneously. The worker now stores the
  embedding result in Redis directly and returns only the key reference, bypassing
  the Celery result backend entirely.
- Fixed Celery task routing: `embed_query_task` was matched by the `.*` wildcard
  and routed to the `indexing` queue instead of `search`. Updated the route to
  use the exact task name.
- Added retry logic (3 attempts with backoff) in `embed_query` for transient
  decode errors from the Celery result backend.
- Added API-side heartbeat thread to ensure embedding model readiness is always
  reported even when Celery workers are saturated with indexing tasks.
- Fixed `/models/embeddings` admin endpoint to return flat model profile alongside
  model list for consistent UI consumption.

### Changed

- Added collection creation and masked service-key authorization controls to the administrative
  Collections page.
- Added a tenant- and project-validated admin endpoint for updating an active service key's
  collection allowlist without rotating or exposing the secret.
- Rewrote the project README around the current architecture, owner-isolation contract,
  provisioning order, retrieval modes, degradation behavior, and verification commands.
- Corrected the Compose startup instructions: the one-shot `rag-migrate` service applies Alembic
  migrations before API and worker startup, so a normal deployment does not run a second manual
  migration.
- Replaced the migration-only changelog with a complete release history from the initial bootstrap
  through the current owner-scoped version.

## [0.1.0] - 2026-08-13

### Added

- Added `owner_user_id` to documents, document versions, and chunks through Alembic migration
  `0004_owner_user_id`.
- Added the required `X-Owner-User-Id` service-request header and a UUID owner field on the
  authenticated principal.
- Added owner filters to administrative document and chunk inspection endpoints.
- Added persistent URL parameters for scoped admin views.
- Added regression coverage for cross-owner ingestion, document access, vector retrieval, BM25
  retrieval, hybrid fusion, reranker input, feedback, and evaluation routes.

### Changed

- **Breaking:** every service API request now requires a valid `X-Owner-User-Id` header. Missing or
  malformed values return HTTP 400. Administrative endpoints continue to use the separate admin
  token.
- Included the owner UUID in deterministic document identity and the document uniqueness boundary,
  allowing different users to reuse the same external document ID without collision.
- Added owner terms to OpenSearch documents and filters, and propagated owner scope through
  background indexing and deletion jobs.
- Backfilled pre-existing rows with sentinel owner
  `00000000-0000-0000-0000-000000000000`; existing OpenSearch collections require reindexing to
  populate the new owner field.

### Security

- Enforced owner scope inside PostgreSQL and OpenSearch queries before ranking or metadata filters.
- Ensured the external reranker receives only candidates that already passed tenant, project,
  collection, and owner filters.
- Made missing owner scope fail closed rather than falling back to an unfiltered retrieval path.

## [0.1.0-rc.2] - 2026-08-12

### Added

- Added tenant listing and creation controls to the Projects page.
- Added project IDs to the project table for easier API and retrieval setup.
- Added document chunk counts and expandable metadata/content inspection to the Documents page.

### Changed

- Replaced free-form tenant UUID entry during project creation with a tenant selector.
- Moved the Prometheus multiprocess directory to container-local temporary storage.

### Fixed

- Returned a controlled `404` when project creation references an unknown tenant instead of
  allowing a database failure to surface as a server error.
- Fixed project forms that accessed a stale React `currentTarget` after an asynchronous request.

## [0.1.0-rc.1] - 2026-08-11

### Added

- Added versioned document, parser, chunker, embedding, and index identities through migration
  `0003_document_index_identity`.
- Added deterministic document/version/chunk IDs, chunk provenance, safe reindexing, and current
  version filtering.
- Added explicit `lexical`, `dense`, and RRF-backed `hybrid` retrieval modes with per-stage scores,
  timings, requested/effective mode, and degraded-state traces.
- Added bounded context selection with maximum chunks, estimated-token budget, per-document limits,
  and near-duplicate suppression.
- Added version-aware query-embedding caching in a dedicated Redis namespace.
- Added comparable evaluation metrics before and after reranking, per-metric deltas, aggregate
  uplift, and a generic hard-negative dataset.
- Added end-to-end pipeline coverage for ingestion, deterministic IDs, current-version filtering,
  tenant isolation, chunking, hybrid retrieval, reranking, context selection, and duplicate
  suppression.
- Added Prometheus multiprocess aggregation shared by the API and indexing worker.

### Changed

- Made PostgreSQL indexing authoritative and allowed an OpenSearch failure to finish as
  `completed_degraded`/`partially_indexed` instead of discarding usable dense-search data.
- Made lexical requests fall back to dense retrieval when OpenSearch is unavailable.
- Made reranker failures return the fused candidate order with explicit degradation metadata.

### Fixed

- Accepted reranker responses that provide valid raw logits without requiring a normalized score.
- Preserved normalized ranking behavior when an external reranker supplies either raw or normalized
  score fields.

## [0.1.0-alpha.2] - 2026-08-09

### Added

- Added administrative tenant provisioning, project and collection management, scoped service API
  key creation/revocation, and registered-collection enforcement.
- Added paginated document queries, batch ingestion, optimistic document updates, chunk inspection,
  document reindex, deletion, and collection-wide reindex operations.
- Added indexing job details, retry/cancel operations, filtered bulk retry, reconciliation, and
  dead-letter recovery controls.
- Added retrieval trace list/detail/repeat APIs and configuration comparison for the same scoped
  query.
- Added reranker status/test controls, embedding-worker compatibility checks, guarded embedding
  reindex, system resource metrics, time-series metrics, persistent runtime settings, scoped cache
  clearing, and administrative audit events.
- Added admin UI pages for projects, documents, jobs, retrieval, embeddings, reranker, evaluations,
  feedback, settings, system health, metrics, and audit logs.
- Added admin Playwright coverage, theme support, and a UI error boundary.
- Added Alembic migration `0002_runtime_settings` for the typed runtime-settings catalog.
- Added the one-shot `rag-migrate` Compose service and dependency health checks so schema upgrades
  complete before the API and worker start.

### Changed

- Required ingestion to target a collection already registered for the authenticated tenant and
  project.
- Made destructive admin operations require explicit confirmation and audit successful mutations.
- Restricted cache clearing to the configured RAG namespace instead of flushing unrelated Redis
  data.

### Fixed

- Prevented an authorized key from creating an implicit or cross-project collection during
  ingestion.
- Improved UI failure handling for provisioning and administrative mutations.

## [0.1.0-alpha.1] - 2026-08-08

### Added

- Bootstrapped the FastAPI service, React admin UI, PostgreSQL/pgvector, OpenSearch, Redis/Celery,
  MinIO, Docker Compose, and Alembic runtime.
- Added tenant-scoped, HMAC-hashed service API keys with project, collection, and permission scopes,
  plus a separate administrative token.
- Added Alembic migration `0001_initial` for tenants, projects, collections, API keys, documents,
  versions, chunks, embeddings, jobs, retrieval events, feedback, evaluations, outbox events, and
  audit records.
- Added JSON and multipart document ingestion, safe source extraction, deterministic chunking,
  pgvector embeddings, OpenSearch indexing, outbox dispatch, and idempotent worker tasks.
- Added vector and BM25 candidate retrieval, reciprocal-rank fusion, optional external reranking,
  retrieval traces, context responses, feedback, and evaluation metrics.
- Added worker model/dimension heartbeats and readiness checks without importing the embedding model
  into the API process.
- Added reconciliation, source-safety, fusion, chunking, evaluation, infrastructure, and API
  contract tests with an 85% coverage gate.

### Changed

- Split the API and embedding worker into separate container targets so the API image does not carry
  the sentence-transformers runtime.
- Kept PostgreSQL authoritative and treated OpenSearch as a rebuildable projection from the first
  version.

### Fixed

- Stabilized worker startup, model cache ownership, and container runtime dependencies.
- Fixed embedding model reuse so each worker process loads the configured model once rather than per
  task.
