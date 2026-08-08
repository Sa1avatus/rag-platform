# Security

Tenant identity comes only from the API key record. Keys are stored as peppered HMAC-SHA256 hashes,
shown once, redacted from logs, and scoped to projects, collections and permissions. Query filters
cannot override server scope. Rotate local passwords before deployment and supply secrets through a
secret manager. Upload processing must reject executable/mismatched MIME content, archives over
25 MiB, expanded archives over 100 MiB, depth over three, symlinks and path traversal. Private
document content is excluded from normal logs and traces require content permission.
