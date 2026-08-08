import time
import uuid

import structlog
from fastapi import Request
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"])
DURATION = Histogram("rag_http_request_duration_seconds", "HTTP latency", ["method", "route"])


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            correlation_id=request.headers.get("x-correlation-id", request_id),
        )
        response = await call_next(request)
        route = request.url.path
        REQUESTS.labels(request.method, route, response.status_code).inc()
        DURATION.labels(request.method, route).observe(time.perf_counter() - started)
        response.headers["x-request-id"] = request_id
        return response
