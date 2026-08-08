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

The API container never imports the embedding module. The worker loads `BAAI/bge-m3` once per
process. Reranking is an optional external HTTP dependency and is deliberately absent from Compose.

## Example

```bash
curl -X POST http://127.0.0.1:8100/v1/documents \
  -H 'Authorization: Bearer rag_example_service_key' -H 'Content-Type: application/json' \
  -d '{"project_id":"4b14572f-c62f-40bd-b6e0-79530f955d73","collection":"manuals","external_document_id":"guide-1","content":"A production operations guide.","version":1}'
```

See [ARCHITECTURE.md](ARCHITECTURE.md), [API.md](API.md), [SECURITY.md](SECURITY.md),
[OPERATIONS.md](OPERATIONS.md), [EVALUATION.md](EVALUATION.md), and
[DEVELOPMENT.md](DEVELOPMENT.md).
