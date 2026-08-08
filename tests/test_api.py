import os

os.environ.setdefault("RAG_ADMIN_TOKEN", "local-rag-admin-token")
os.environ.setdefault("RAG_API_KEY_PEPPER", "local-development-pepper")

from fastapi.testclient import TestClient

from rag_platform.main import app


def test_liveness() -> None:
    response = TestClient(app).get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def test_admin_rejects_missing_token() -> None:
    assert TestClient(app).get("/v1/admin/settings").status_code in {401, 422}


def test_metrics_are_prometheus_compatible() -> None:
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert "rag_http_requests_total" in response.text
