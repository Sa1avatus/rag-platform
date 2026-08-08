import asyncio
import io
import uuid
from functools import lru_cache
from pathlib import PurePath

from minio import Minio

from rag_platform.core.config import get_settings


def object_key(
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    upload_id: uuid.UUID,
    filename: str,
) -> str:
    safe_name = PurePath(filename.replace("\\", "/")).name
    return f"{tenant_id}/{project_id}/{upload_id}/{safe_name}"


@lru_cache(maxsize=1)
def client() -> Minio:
    settings = get_settings()
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=False,
    )


async def put(key: str, content: bytes, content_type: str) -> None:
    await asyncio.to_thread(_put_sync, key, content, content_type)


async def remove(key: str) -> None:
    await asyncio.to_thread(client().remove_object, get_settings().minio_bucket, key)


def _put_sync(key: str, content: bytes, content_type: str) -> None:
    storage = client()
    bucket = get_settings().minio_bucket
    if not storage.bucket_exists(bucket):
        storage.make_bucket(bucket)
    storage.put_object(
        bucket,
        key,
        io.BytesIO(content),
        length=len(content),
        content_type=content_type,
    )
