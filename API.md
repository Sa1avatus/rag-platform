# API

The OpenAPI document at `/openapi.json` is canonical. `/v1` is backward-compatible; breaking
changes require `/v2`. Service calls use `Authorization: Bearer <service-key>`. Administrative calls
use the separately managed admin token. Pagination endpoints use server-side filters and stable IDs.

Core paths are `/v1/documents`, `/v1/retrieval/search`, `/v1/retrieval/context`, `/v1/feedback`,
`/v1/evaluations/*`, and `/v1/admin/*`. Context returns sources and assembled text, never an LLM
answer.

`POST /v1/documents/upload` accepts multipart fields `project_id`, `collection`,
`external_document_id`, `file`, optional `document_type`, `language`, `version`, and JSON-object
`metadata`. The service validates the complete source before extraction, stores the original in
MinIO, and creates one version per safe ZIP member. The 25 MiB compressed and 100 MiB expanded
limits are enforced server-side.

Administrators can inspect jobs through `GET /v1/admin/indexing/jobs`, retry failed or dead-letter
jobs through `POST /v1/admin/indexing/jobs/{job_id}/retry`, and cancel jobs that are still queued
through `POST /v1/admin/indexing/jobs/{job_id}/cancel`. Running jobs cannot be canceled by this API.
`GET /v1/admin/indexing/jobs/{job_id}` returns one job using the same status and stage fields as the
list response.
Create a tenant with `POST /v1/admin/tenants` before registering projects or service API keys for
that tenant.
Projects and collections support detail reads and partial updates at
`/v1/admin/projects/{project_id}` and `/v1/admin/collections/{collection_id}`. Their tenant and
project ownership fields are immutable through these endpoints.

Service clients can page through active documents with `GET /v1/documents?project_id=...` and may
restrict the result to one authorized `collection`. `GET /v1/documents/{document_id}/chunks` returns
the current indexed derivatives without exposing records outside the API key scope.

Document ingestion accepts only collections registered for the same tenant and project through the
administrative API. An authorized key alone cannot create an implicit collection.

`POST /v1/documents/batch` accepts `{"documents": [...]}` with 1 to 100 document payloads and
returns accepted versions in input order. The operation is intentionally non-atomic: processing
stops on the first `409`, while documents accepted before that failure remain committed and queued.

`PATCH /v1/documents/{document_id}` creates the next immutable version and requires
`expected_lock_version` plus new `content`. Optional title, type, language, and metadata values
inherit from the current document when omitted. Document list and detail responses expose the
current `lock_version`; stale writers receive `409`.
