# RAG Platform

RAG Platform is a standalone, multi-tenant document ingestion and hybrid retrieval service. It
owns source blobs, immutable document versions, parsing, chunking, embeddings, vector and BM25
search, RRF fusion, optional reranking, context selection, retrieval traces, feedback, and
evaluation. Client-specific business logic and answer generation remain outside this repository.

Current development release: **0.1.0** (2026-08-13). The release history was reconstructed from
the `master` branch because the repository did not previously contain Git tags; see
[CHANGELOG.md](CHANGELOG.md).

## Capabilities

- Tenant-, project-, collection-, and owner-scoped document ingestion and retrieval.
- Immutable document versions, optimistic updates, deterministic identifiers, and idempotent
  indexing jobs.
- Safe uploads for PDF, DOCX, text, HTML, and bounded ZIP archives, with original blobs stored in
  MinIO.
- Explicit `lexical`, `dense`, and RRF-backed `hybrid` retrieval modes.
- Bounded context selection with chunk, estimated-token, per-document, and near-duplicate limits.
- Optional external reranking with conservative fallback to fused retrieval results.
- Query-embedding caching in an isolated Redis namespace with model- and revision-aware keys.
- Retrieval traces, relevance feedback, versioned evaluation datasets, and before/after reranking
  quality metrics.
- Administrative UI for tenants, projects, collections, documents, indexing jobs, retrieval,
  models, evaluations, feedback, runtime settings, health, metrics, and audit events.

## Architecture

```mermaid
flowchart LR
    Client["Service client"] --> API["FastAPI / rag-api"]
    Admin["Administrator"] --> Web["React admin UI / Nginx"]
    Web --> API
    API --> PG[("PostgreSQL + pgvector")]
    API --> OS[("OpenSearch")]
    API --> Redis[("Redis")]
    API --> RR["External reranker"]
    Worker["Celery indexing worker"] --> PG
    Worker --> OS
    Worker --> Redis
    Worker --> MinIO[("MinIO source blobs")]
```

PostgreSQL is authoritative. OpenSearch is a rebuildable lexical projection, Redis coordinates
work and caches query embeddings, and MinIO stores original source files. The API process never
loads the embedding model; the worker loads `BAAI/bge-m3` once per process and advertises its model
and vector dimension through a readiness heartbeat.

Ingestion commits the document version and outbox event atomically. The worker then extracts text,
creates deterministic chunks, writes chunks and embeddings to PostgreSQL, and updates OpenSearch.
An OpenSearch failure leaves the version `partially_indexed` but available to dense retrieval.

## Authentication and isolation

Service API calls require both headers:

- `Authorization: Bearer <service-api-key>` identifies the tenant and its allowed projects,
  collections, and permissions.
- `X-Owner-User-Id: <uuid>` identifies the end user who owns the documents involved in the
  request.

The owner UUID is stored on documents, versions, and chunks. Dense search, BM25 search, fusion, and
reranker input are filtered by owner before ranking. A missing or invalid owner header is rejected
with HTTP 400. Two owners may use the same `external_document_id` without sharing data or
deduplication boundaries.

The administrative API uses a separate admin token and is a trusted cross-owner control plane.
Document inspection endpoints accept an optional owner filter; ordinary service endpoints never
bypass owner scope.

## Quick start

Requirements: Docker Desktop with Compose v2. Python 3.12 and Node.js 22 are needed only for direct
development outside the containers.

1. Create local configuration and replace every development secret:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Build and start the stack:

   ```powershell
   docker compose up -d --build
   docker compose ps --all
   ```

   The one-shot `rag-migrate` service runs `alembic upgrade head` before the API and worker start.
   A successful migration appears as `Exited (0)`; a failed migration blocks dependent services.
   Do not run a second manual migration as part of the normal Compose startup.

3. Open:

   - API documentation: `http://127.0.0.1:8100/docs` locally or
     `http://192.168.1.93:8100/docs` from the trusted LAN
   - Admin UI: `http://127.0.0.1:8300` locally or `http://192.168.1.93:8300` from the trusted LAN

   Compose publishes only the authenticated API and admin UI to the LAN. PostgreSQL, OpenSearch,
   Redis, and MinIO remain bound to loopback.

The admin UI keeps its token only in browser session storage. The checked-in `.env.example`
contains the non-secret local example token `local-rag-admin-token`; replace it before any shared or
deployed use.

## First-time provisioning

Use the admin UI or `/v1/admin/*` endpoints to create resources in this order:

1. tenant;
2. project belonging to that tenant;
3. one or more registered collections;
4. a scoped service API key.

An API key alone cannot create an implicit collection. Save a newly created raw service key when it
is shown: later admin responses expose only its prefix and scopes, never the key or its stored hash.

## Ingest a document

```bash
curl -X POST http://127.0.0.1:8100/v1/documents \
  -H 'Authorization: Bearer rag_example_service_key' \
  -H 'X-Owner-User-Id: a1b2c3d4-e5f6-7890-abcd-ef1234567890' \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"4b14572f-c62f-40bd-b6e0-79530f955d73","collection":"manuals","external_document_id":"guide-1","content":"A production operations guide.","version":1}'
```

The project and collection must already exist and be authorized for the supplied service key.

## Retrieval and degradation

- `lexical` uses OpenSearch and does not request a query embedding while OpenSearch is healthy.
- `dense` uses the embedding worker and PostgreSQL/pgvector without calling OpenSearch.
- `hybrid` combines dense and lexical rankings with reciprocal-rank fusion.
- OpenSearch failure degrades `lexical` and `hybrid` requests to dense retrieval.
- Reranker failure returns the fused order and marks the trace degraded.
- Loss of PostgreSQL, Redis readiness, or the compatible embedding-worker heartbeat makes the
  service unready; `/health/live` remains a process-only liveness check.

`/v1/retrieval/context` returns selected source chunks and assembled context, never an LLM answer.

## Development and verification

Backend checks:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pytest
```

Frontend checks from `web/`:

```powershell
pnpm install --frozen-lockfile
pnpm test
pnpm build
pnpm e2e
```

The Playwright suite expects the Compose UI at `http://127.0.0.1:8300`. Integration checks must use
disposable infrastructure and must not reuse deployment credentials or private source documents.

## Documentation

- [API.md](API.md) — service and administrative HTTP contracts.
- [ARCHITECTURE.md](ARCHITECTURE.md) — component ownership and data flow.
- [SECURITY.md](SECURITY.md) — authentication, isolation, uploads, and secret boundaries.
- [OPERATIONS.md](OPERATIONS.md) — migrations, health, recovery, reindexing, and backups.
- [EVALUATION.md](EVALUATION.md) — datasets and retrieval-quality metrics.
- [DEVELOPMENT.md](DEVELOPMENT.md) — local development requirements and checks.
- [CHANGELOG.md](CHANGELOG.md) — complete version history.

## Product boundary

RAG Platform retrieves and assembles evidence. It does not generate answers, embed client-specific
business rules, silently cross tenant or owner scope, or treat OpenSearch as authoritative. Model,
chunking, or index changes require a new immutable revision, evaluation against a pinned dataset,
and an explicit reindex and rollback plan.
