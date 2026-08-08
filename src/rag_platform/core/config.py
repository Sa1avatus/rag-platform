from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    env: str = "development"
    database_url: str = "postgresql+asyncpg://rag:rag-local-password@localhost:55432/rag"
    redis_url: str = "redis://localhost:56379/0"
    opensearch_url: str = "http://localhost:59200"
    minio_endpoint: str = "localhost:59000"
    minio_access_key: str = "rag-local-access"
    minio_secret_key: str = "change-me"
    minio_bucket: str = "rag-documents"
    admin_token: str = Field(min_length=16)
    api_key_pepper: str = Field(min_length=12)
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    embedding_dimension: int = Field(default=1024, ge=1)
    query_embedding_timeout_seconds: float = 30
    document_max_bytes: int = 25 * 1024 * 1024
    reranker_base_url: str = "http://reranker-service:8200"
    reranker_timeout_seconds: float = 8
    reranker_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
