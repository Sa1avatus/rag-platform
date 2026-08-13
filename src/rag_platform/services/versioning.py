import hashlib
import unicodedata
import uuid

DOCUMENT_NAMESPACE = uuid.UUID("1bea7b23-284d-5ef2-96f3-cf6c6e8b9d69")


def normalize_content(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_document_id(
    tenant_id: uuid.UUID,
    project_id: uuid.UUID,
    collection: str,
    owner_user_id: uuid.UUID,
    external_document_id: str,
) -> uuid.UUID:
    identity = f"{tenant_id}:{project_id}:{collection}:{owner_user_id}:{external_document_id}"
    return uuid.uuid5(DOCUMENT_NAMESPACE, identity)


def stable_version_id(document_id: uuid.UUID, version: int, digest: str) -> uuid.UUID:
    return uuid.uuid5(document_id, f"{version}:{digest}")


def stable_chunk_id(
    version_id: uuid.UUID,
    chunker_version: str,
    chunk_index: int,
    digest: str,
) -> uuid.UUID:
    return uuid.uuid5(version_id, f"{chunker_version}:{chunk_index}:{digest}")
