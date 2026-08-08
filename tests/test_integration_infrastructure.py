import json
import uuid

import pytest
from redis.asyncio import Redis

from rag_platform.core.config import get_settings
from rag_platform.services.blobs import client as minio_client
from rag_platform.services.blobs import object_key, put
from rag_platform.services.opensearch import (
    bm25_search,
    delete_document_chunks,
    index_chunks,
)
from rag_platform.services.readiness import MODEL_READY_KEY, readiness_status


@pytest.mark.asyncio
async def test_opensearch_indexes_searches_and_deletes_scoped_chunk() -> None:
    tenant_id = uuid.uuid4()
    project_id = uuid.uuid4()
    document_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    await index_chunks(
        [
            {
                "tenant_id": str(tenant_id),
                "project_id": str(project_id),
                "collection": "integration",
                "document_id": str(document_id),
                "chunk_id": str(chunk_id),
                "content": "Distinctive platypus retrieval phrase",
                "language": "en",
                "metadata": {"approved": True},
            }
        ]
    )
    hits = await bm25_search(
        tenant_id,
        project_id,
        ["integration"],
        "platypus retrieval",
        {"approved": True},
        10,
    )
    assert hits
    assert hits[0][0] == chunk_id
    await delete_document_chunks(tenant_id, project_id, document_id)
    assert not await bm25_search(
        tenant_id,
        project_id,
        ["integration"],
        "platypus retrieval",
        {},
        10,
    )


@pytest.mark.asyncio
async def test_minio_stores_source_object() -> None:
    tenant_id = uuid.uuid4()
    project_id = uuid.uuid4()
    key = object_key(tenant_id, project_id, uuid.uuid4(), "source.txt")
    await put(key, b"integration source", "text/plain")
    stat = minio_client().stat_object(get_settings().minio_bucket, key)
    assert stat.size == len(b"integration source")
    assert stat.content_type == "text/plain"


@pytest.mark.asyncio
async def test_readiness_accepts_compatible_worker_heartbeat() -> None:
    settings = get_settings()
    cache = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await cache.set(
            MODEL_READY_KEY,
            json.dumps(
                {
                    "model": settings.embedding_model,
                    "dimension": settings.embedding_dimension,
                    "device": "cpu",
                }
            ),
            ex=60,
        )
    finally:
        await cache.aclose()
    ready, components = await readiness_status()
    assert ready is True
    assert components["postgresql"]["status"] == "up"
    assert components["redis"]["status"] == "up"
    assert components["embedding_model"]["status"] == "ready"
