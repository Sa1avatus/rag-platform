# API

The OpenAPI document at `/openapi.json` is canonical. `/v1` is backward-compatible; breaking
changes require `/v2`. Service calls use `Authorization: Bearer ` to authenticate the upstream
service and `X-Owner-User-Id: ` to identify the resource owner. The header is required on every
service API request; the API returns 400 if it is missing or not a valid UUID. Administrative calls
use the separately managed admin token and are not restricted to a single owner. Pagination endpoints
use server-side filters and stable IDs.

Core paths are `/v1/documents`, `/v1/retrieval/search`, `/v1/retrieval/context`, `/v1/feedback`,
`/v1/evaluations/*`, and `/v1/admin/*`. Context returns sources and assembled text, never an LLM
answer.

Retrieval accepts `mode` as `lexical`, `dense`, or `hybrid` (the default). All retrieval modes
apply the `X-Owner-User-Id` filter inside the vector, BM25, and fusion stages; the reranker
receives only owner-scoped candidates. Lexical mode avoids the
embedding worker while OpenSearch is healthy; dense mode does not call OpenSearch; hybrid mode
combines both rankings with RRF. If OpenSearch fails, hybrid and lexical requests degrade to dense
retrieval and expose `requested_mode`, `effective_mode`, stage timings, and the degraded state in
the trace. Results include source provenance, per-stage scores, reranker score when present, and
final rank. `/v1/retrieval/context` additionally enforces `max_context_chunks`,
`max_context_tokens`, and `per_document_limit`, and suppresses near-duplicate chunks.

`POST /v1/documents/upload` accepts multipart fields `project_id`, `collection`,
`external_document_id`, `file`, optional `document_type`, `language`, `version`, and JSON-object
`metadata`. The service validates the complete source before extraction, stores the original in
MinIO, and creates one version per safe ZIP member. The 25 MiB compressed and 100 MiB expanded
limits are enforced server-side.

Administrators can inspect jobs through `GET /v1/admin/indexing/jobs`, retry failed or dead-letter
jobs through `POST /v1/admin/indexing/jobs/{job_id}/retry`, and cancel jobs that are still queued
through `POST /v1/admin/indexing/jobs/{job_id}/cancel`. Running jobs cannot be canceled by this API.
`POST /v1/admin/indexing/jobs/retry-filtered` retries up to 500 failed or dead-letter jobs selected
by status and optional project, skips malformed legacy targets, and commits the batch atomically.
`GET /v1/admin/indexing/jobs/{job_id}` returns one job using the same status and stage fields as the
list response.
Document operators can use `POST /v1/admin/documents/{document_id}/reindex` and
`POST /v1/admin/documents/{document_id}/delete`; both require tenant, project, and collection scope
parameters plus `{"confirm": true}`, enqueue an idempotent worker job, and write an audit event.
Create a tenant with `POST /v1/admin/tenants` before registering projects or service API keys for
that tenant.
Projects and collections support detail reads and partial updates at
`/v1/admin/projects/{project_id}` and `/v1/admin/collections/{collection_id}`. Their tenant and
project ownership fields are immutable through these endpoints.
`POST /v1/admin/collections/{collection_id}/reindex` requeues every current, non-deleted version in
that collection while skipping versions that already have active indexing jobs.
`POST /v1/admin/collections/{collection_id}/compare-configurations` runs baseline and candidate
retrieval settings against the same query and collection scope, returning both traces plus chunk
overlap counts.
Administrative retrieval traces are available through `GET /v1/admin/retrieval/traces` and its
detail endpoint. `POST /v1/admin/retrieval/traces/{request_id}/repeat` replays the stored query,
collections, metadata filters, and retrieval configuration into a new trace.
`GET /v1/admin/api-keys` exposes prefixes and scopes but never raw keys or hashes. Administrators
can idempotently revoke a key with `DELETE /v1/admin/api-keys/{key_id}`.
`GET /v1/admin/reranker/status` reports safe readiness metadata. `POST /v1/admin/reranker/test`
sends one inert test query and reports latency/result count; external failures are returned as
`unavailable` rather than failing the admin API.
`GET /v1/admin/models/embeddings` and `POST /v1/admin/models/embeddings/check` report the worker
heartbeat and verify model name/dimension compatibility without loading the model in the API.
`POST /v1/admin/models/embeddings/reindex` requires `{"confirm": true}`, refuses incompatible model
profiles, and requeues all current, non-deleted versions while preserving active jobs.
`GET /v1/admin/system/resources` reports CPU load, memory, disk, and GPU detection for the `rag-api`
container only. It never executes shell commands or exposes host/Docker control interfaces.
`GET /v1/admin/metrics/timeseries` aggregates allowlisted PostgreSQL event metrics into hourly or
daily count buckets over a maximum 90-day range, with optional project and collection filters.
`GET /v1/admin/settings` returns persistent runtime values with descriptions, defaults, ranges, and
restart/reindex flags. `PATCH /v1/admin/settings` accepts only the documented typed allowlist and
records the mutation in the audit log; it never accepts or returns secrets.
`POST /v1/admin/cache/clear` requires `{"confirm": true}` and deletes only keys below the configured
RAG cache namespace. It does not flush Celery results or unrelated Redis data and records an audit
event.
`GET /v1/admin/audit-log` lists newest administrative mutations and can filter by action, tenant,
or project. Audit payloads contain resource identifiers only; service-key secrets and hashes are
never recorded.
Project reconciliation, collection reindex, and indexing-job retry/cancel operations are also
recorded after they succeed.

Service clients can page through active documents with `GET /v1/documents?project_id=...` and may
restrict the result to one authorized `collection`. All results are scoped to the requesting
`X-Owner-User-Id`; two users with the same API key see disjoint document sets.
`GET /v1/documents/{document_id}/chunks` returns
the current indexed derivatives without exposing records outside the owner scope.

Document ingestion accepts only collections registered for the same tenant and project through the
administrative API. An authorized key alone cannot create an implicit collection.

`POST /v1/documents/batch` accepts `{"documents": [...]}` with 1 to 100 document payloads and
returns accepted versions in input order. The operation is intentionally non-atomic: processing
stops on the first `409`, while documents accepted before that failure remain committed and queued.

`PATCH /v1/documents/{document_id}` creates the next immutable version and requires
`expected_lock_version` plus new `content`. Optional title, type, language, and metadata values
inherit from the current document when omitted. Document list and detail responses expose the
current `lock_version`; stale writers receive `409`.

Evaluation cases accept optional `tags`, `difficulty`, and `category`. Every run pins the retrieval,
embedding, chunker, index, and reranker-contract identity. Results persist metrics before reranking,
after reranking, and their per-metric delta; the run stores the corresponding aggregate uplift.
`examples/evaluation-hard-negatives.json` is a schema-validated generic cross-domain regression
dataset; it demonstrates diagnosable hard negatives without embedding client business rules.
