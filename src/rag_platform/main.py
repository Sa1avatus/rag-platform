from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST

from rag_platform.api.routes import admin, documents, evaluations, feedback, retrieval
from rag_platform.core.observability import RequestContextMiddleware
from rag_platform.services.query_embeddings import QueryEmbeddingUnavailable
from rag_platform.services.readiness import readiness_status

app = FastAPI(title="RAG Platform", version="1.0.0")
app.add_middleware(RequestContextMiddleware)
app.include_router(documents.router)
app.include_router(retrieval.router)
app.include_router(feedback.router)
app.include_router(evaluations.router)
app.include_router(admin.router)


@app.on_event("startup")
async def _cleanup_prometheus_multiprocess() -> None:
    """Remove stale metric files from dead processes at startup."""
    import os

    mp_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not mp_dir or not os.path.isdir(mp_dir):  # noqa: ASYNC240
        return
    from prometheus_client.multiprocess import mark_process_dead

    for name in os.listdir(mp_dir):
        if not name.endswith(".db"):
            continue
        try:
            pid = int(name.split("_", 1)[0])
        except (ValueError, IndexError):
            continue
        try:
            os.kill(pid, 0)
        except OSError:
            mark_process_dead(pid)  # type: ignore[no-untyped-call]


@app.exception_handler(QueryEmbeddingUnavailable)
async def query_embedding_unavailable(
    request: object,
    exc: QueryEmbeddingUnavailable,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "vector search is unavailable"},
    )


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> JSONResponse:
    is_ready, components = await readiness_status()
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "components": components,
        },
    )


@app.get("/metrics")
async def metrics() -> Response:
    import os

    from prometheus_client import CollectorRegistry, generate_latest, multiprocess

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
