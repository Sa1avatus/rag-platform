import json
import uuid
from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import OpenSearchException

from rag_platform.core.config import get_settings

INDEX_NAME = "rag-chunks-v1"


class OpenSearchUnavailable(RuntimeError):
    pass


def client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[get_settings().opensearch_url],
        http_compress=True,
        retry_on_timeout=True,
        max_retries=2,
    )


async def ensure_index(search_client: AsyncOpenSearch) -> None:
    if await search_client.indices.exists(index=INDEX_NAME):
        return
    await search_client.indices.create(
        index=INDEX_NAME,
        body={
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "tenant_id": {"type": "keyword"},
                    "project_id": {"type": "keyword"},
                    "collection": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "language": {"type": "keyword"},
                    "metadata": {"type": "flat_object"},
                },
            },
        },
    )


async def index_chunks(documents: list[dict[str, Any]]) -> None:
    search_client = client()
    try:
        await ensure_index(search_client)
        lines: list[str] = []
        for document in documents:
            lines.append(json.dumps({"index": {"_index": INDEX_NAME, "_id": document["chunk_id"]}}))
            lines.append(json.dumps(document, default=str))
        response = await search_client.bulk(body="\n".join(lines) + "\n", refresh=True)
        if response.get("errors"):
            raise OpenSearchUnavailable("OpenSearch bulk indexing partially failed")
    except (OpenSearchException, OSError, TimeoutError) as exc:
        raise OpenSearchUnavailable("OpenSearch indexing is unavailable") from exc
    finally:
        await search_client.close()


async def bm25_search(
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    collections: list[str],
    query: str,
    filters: dict[str, Any],
    top_k: int,
) -> list[tuple[uuid.UUID, float]]:
    clauses: list[dict[str, Any]] = [
        {"term": {"tenant_id": str(tenant_id)}},
        {"term": {"project_id": str(project_id)}},
        {"terms": {"collection": collections}},
    ]
    clauses.extend({"term": {f"metadata.{key}": value}} for key, value in filters.items())
    search_client = client()
    try:
        response = await search_client.search(
            index=INDEX_NAME,
            body={
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [{"match": {"content": query}}],
                        "filter": clauses,
                    }
                },
            },
        )
        return [
            (uuid.UUID(hit["_id"]), float(hit["_score"] or 0.0)) for hit in response["hits"]["hits"]
        ]
    except (OpenSearchException, OSError, TimeoutError) as exc:
        raise OpenSearchUnavailable("OpenSearch search is unavailable") from exc
    finally:
        await search_client.close()


async def delete_document_chunks(
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
) -> None:
    search_client = client()
    try:
        if not await search_client.indices.exists(index=INDEX_NAME):
            return
        await search_client.delete_by_query(
            index=INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"tenant_id": str(tenant_id)}},
                            {"term": {"project_id": str(project_id)}},
                            {"term": {"document_id": str(document_id)}},
                        ]
                    }
                }
            },
            refresh=True,
            conflicts="proceed",
        )
    except (OpenSearchException, OSError, TimeoutError) as exc:
        raise OpenSearchUnavailable("OpenSearch deletion is unavailable") from exc
    finally:
        await search_client.close()
