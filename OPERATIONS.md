# Operations

## Admin UI

Open `http://127.0.0.1:8300` and authenticate with the configured admin token. The token is retained
only in browser session storage and is cleared by **Log out**. Secrets and infrastructure
credentials are not displayed by the UI.

Filters for audit events, indexing jobs, documents, evaluation runs, and feedback are represented
in the URL where applicable, so a scoped view can be bookmarked. Reindex, retry, cancel, repeat,
and evaluation-run actions require explicit confirmation or are limited to safe backend states.

The reranker is an optional external dependency. An unavailable reranker is shown as degraded and
does not make retrieval unavailable. Use **Test connection** on the Reranker page to validate the
configured backend connection without revealing its URL or credentials.

For a deployment check, verify `docker compose ps`, open the System Health page, then run `pnpm
e2e` from `web/`. The E2E suite expects the UI at `http://127.0.0.1:8300` and uses the public local
development token from `.env.example`; do not reuse that token in a deployed environment.

`docker compose up -d` runs the one-shot `rag-migrate` service before API and worker startup. The
service applies `alembic upgrade head` and must exit successfully; Compose blocks dependent
services if migration fails. `docker compose ps --all` shows the completed migration container as
`Exited (0)`. API, worker, web, PostgreSQL, Redis, OpenSearch, and MinIO expose container
healthchecks.

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
