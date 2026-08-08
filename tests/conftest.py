import os

os.environ.setdefault(
    "RAG_DATABASE_URL",
    "postgresql+asyncpg://rag:rag-local-password@127.0.0.1:55432/rag",
)
os.environ.setdefault("RAG_REDIS_URL", "redis://127.0.0.1:56379/0")
os.environ.setdefault("RAG_OPENSEARCH_URL", "http://127.0.0.1:59200")
os.environ.setdefault("RAG_MINIO_ENDPOINT", "127.0.0.1:59000")
os.environ.setdefault("RAG_ADMIN_TOKEN", "local-rag-admin-token")
os.environ.setdefault("RAG_API_KEY_PEPPER", "local-development-pepper")
