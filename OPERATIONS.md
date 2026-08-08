# Operations

Use `make up`, `make migrate`, `make test`, `make lint`, and `make down`. Back up PostgreSQL with
`pg_dump --format=custom` and MinIO with versioned object replication. Restore PostgreSQL and MinIO
before replaying the PostgreSQL-authoritative chunks into a fresh versioned OpenSearch index; switch
the alias only after count and sample-query validation.

To change embeddings, create a new immutable model profile, verify dimension/device, estimate disk,
build parallel embeddings and indexes, run evaluation regression, switch aliases atomically, and
retain the old index for rollback. Never mutate an active collection model in place.

Readiness requires PostgreSQL and worker model readiness; OpenSearch and reranker report degraded
status. Graceful worker shutdown allows 60 seconds. Investigate dead-letter jobs before retrying and
run reconciliation to repair statuses or orphaned chunks.

The indexing worker also runs the local Celery beat scheduler. Every two seconds it claims pending
outbox rows with `FOR UPDATE SKIP LOCKED`, publishes deterministic task IDs, and records publication
in the event payload. Delivery is at least once; indexing remains idempotent. In a horizontally
scaled production deployment, run exactly one beat scheduler separately from worker replicas.

An OpenSearch outage does not fail PostgreSQL indexing. Affected versions remain
`partially_indexed`, indexing jobs finish as `completed_degraded`, and retrieval traces set
`opensearch_degraded=true`. After recovery, replay partially indexed versions or rebuild the
versioned index from PostgreSQL before switching the read alias.

`/health/ready` returns 200 only when PostgreSQL and Redis respond and a compatible embedding worker
has refreshed its readiness heartbeat within 60 seconds. It returns component-safe error class names
without connection strings or credentials. `/health/live` remains process-only liveness.
