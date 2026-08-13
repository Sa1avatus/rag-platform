# RAG Platform

Standalone multi-tenant document ingestion and hybrid retrieval service. It owns source blobs,
chunking, embeddings, vector/BM25 retrieval, RRF fusion, retrieval traces, feedback and evaluation;
client-specific business logic and answer generation remain outside this repository.

## Quick start

1. Copy `.env.example` to `.env` and replace every development secret.
2. Run `docker compose up -d --build`.
3. Run `docker compose exec rag-api alembic upgrade head`.
4. Open the API at `http://127.0.0.1:8100/docs` and admin UI at
   `http://127.0.0.1:8300`.

The admin UI uses the local admin token configured for the API. For the checked-in development
configuration, use the non-secret example token `local-rag-admin-token`. The UI exposes scoped
project, document, retrieval, evaluation, feedback, model, health, settings, and audit views. It
never connects directly to PostgreSQL, OpenSearch, Redis, or MinIO; nginx proxies `/api/*` to
`rag-api` on the internal Compose network.

Run frontend verification while the Compose stack is healthy:

```powershell
cd web
pnpm test
pnpm build
pnpm e2e
```

The API container never imports the embedding module. The worker loads `BAAI/bge-m3` once per
process. Reranking is an optional external HTTP dependency and is deliberately absent from Compose.

Every service API request requires an `X-Owner-User-Id` header carrying a stable UUID that
identifies the resource owner (typically the end-user identity from the upstream service).
Documents, chunks, embeddings, and retrieval results are scoped to this owner; a request
without the header is rejected, and queries never return data belonging to another owner.
The admin API (behind the admin token) can optionally filter by owner but is not restricted to one.

Search supports explicit `lexical`, `dense`, and RRF-backed `hybrid` modes. Context selection is a
separate bounded stage with chunk, estimated-token, per-document and near-duplicate controls.
Migration `0004` adds the `owner_user_id` column to documents, versions, and chunks, backfills
existing rows with a zero sentinel UUID, and updates the unique constraint and indexes.
Document, version and chunk identities are deterministic for new ingestion, while migration `0003`
adds parser/chunker/embedding/index revisions and chunk provenance needed for safe reindexing.
Version-aware query-embedding caching is fail-open and isolated to a dedicated Redis namespace.
Evaluation reports comparable quality before and after reranking, including aggregate uplift.

## Example

```bash
curl -X POST http://127.0.0.1:8100/v1/documents \
  -H 'Authorization: Bearer rag_ex..._key' \
  -H 'X-Owner-User-Id: a1b2c3d4-e5f6-7890-abcd-ef1234567890' \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"4b14572f-c62f-40bd-b6e0-79530f955d73","collection":"manuals","external_document_id":"guide-1","content":"A production operations guide.","version":1}'
```

See [ARCHITECTURE.md](ARCHITECTURE.md), [API.md](API.md), [SECURITY.md](SECURITY.md),
[OPERATIONS.md](OPERATIONS.md), [EVALUATION.md](EVALUATION.md), and
[DEVELOPMENT.md](DEVELOPMENT.md).
