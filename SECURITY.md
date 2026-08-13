# Security

Tenant identity comes only from the API key record. Keys are stored as peppered HMAC-SHA256 hashes,
shown once, redacted from logs, and scoped to projects, collections and permissions. Query filters
cannot override server scope. Rotate local passwords before deployment and supply secrets through a
secret manager. Upload processing must reject executable/mismatched MIME content, archives over
25 MiB, expanded archives over 100 MiB, depth over three, symlinks and path traversal. Private
document content is excluded from normal logs and traces require content permission.

Every service API request carries an `X-Owner-User-Id` header that identifies the resource owner.
The owner is stored on every document, version, and chunk, and is enforced as a hard filter in all
retrieval paths — vector search, BM25, hybrid fusion, and reranker input. The filter is applied
inside the storage layer (SQL WHERE and OpenSearch filter clause) before candidate ranking, not as
a post-retrieval mask. If the header is missing or invalid the request is rejected (fail-closed);
retrieval never returns data belonging to another owner. Two users may ingest documents with the
same `external_document_id` — they are logically separate and share no deduplication boundary.
The admin API can optionally filter by owner but is not restricted to one; it is protected by the
separate admin token.