from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rag_platform.api.routes import admin, documents, retrieval
from rag_platform.core.observability import RequestContextMiddleware

app = FastAPI(title="RAG Platform", version="1.0.0")
app.add_middleware(RequestContextMiddleware)
app.include_router(documents.router); app.include_router(retrieval.router); app.include_router(admin.router)


@app.get("/health/live")
async def live() -> dict[str, str]: return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> dict[str, str]: return {"status": "ready"}


@app.get("/metrics")
async def metrics() -> Response: return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
