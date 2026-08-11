from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
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
    embedding_backend: str = "sentence-transformers"
    embedding_revision: str = "default"
    embedding_normalization: str = "l2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    embedding_dimension: int = Field(default=1024, ge=1)
    parser_version: str = "text-v1"
    chunker_version: str = "word-window-v1"
    chunk_strategy: str = "recursive"
    chunk_size_words: int = Field(default=330, ge=16, le=4096)
    chunk_overlap_words: int = Field(default=45, ge=0, le=1024)
    chunk_min_words: int = Field(default=20, ge=1, le=1024)
    index_version: str = "rag-chunks-v1"
    query_embedding_timeout_seconds: float = 30
    query_embedding_cache_enabled: bool = True
    query_embedding_cache_ttl_seconds: int = Field(default=3600, ge=60, le=604800)
    cache_namespace: str = "rag:cache:v1"
    document_max_bytes: int = 25 * 1024 * 1024
    reranker_base_url: str = Field(
        default="http://reranker-service:8200",
        validation_alias=AliasChoices("RERANKER_BASE_URL", "RAG_RERANKER_BASE_URL"),
    )
    reranker_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("RERANKER_API_KEY", "RAG_RERANKER_API_KEY"),
    )
    reranker_timeout_seconds: float = Field(
        default=8,
        validation_alias=AliasChoices(
            "RERANKER_TIMEOUT_SECONDS",
            "RAG_RERANKER_TIMEOUT_SECONDS",
        ),
    )
    reranker_max_retries: int = Field(default=1, ge=0, le=3)
    reranker_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RERANKER_ENABLED", "RAG_RERANKER_ENABLED"),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
