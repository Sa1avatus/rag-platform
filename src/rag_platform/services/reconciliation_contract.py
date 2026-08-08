import uuid
from typing import Any

ACTIVE_JOB_STATUSES = frozenset({"queued", "running"})


def active_targets(
    payloads: list[dict[str, Any]],
) -> tuple[set[uuid.UUID], set[uuid.UUID]]:
    version_ids: set[uuid.UUID] = set()
    document_ids: set[uuid.UUID] = set()
    for payload in payloads:
        if payload.get("status") not in ACTIVE_JOB_STATUSES:
            continue
        _add_uuid(version_ids, payload.get("version_id"))
        _add_uuid(document_ids, payload.get("document_id"))
    return version_ids, document_ids


def _add_uuid(target: set[uuid.UUID], value: object) -> None:
    if not isinstance(value, str):
        return
    try:
        target.add(uuid.UUID(value))
    except ValueError:
        return
