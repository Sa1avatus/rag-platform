from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

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
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
