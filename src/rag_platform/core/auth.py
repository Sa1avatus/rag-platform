import hashlib
import hmac
import uuid
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.core.config import get_settings
from rag_platform.db.models import ApiKey
from rag_platform.db.session import get_session


@dataclass(frozen=True)
class Principal:
    tenant_id: uuid.UUID
    project_ids: frozenset[uuid.UUID]
    collections: frozenset[str]
    permissions: frozenset[str]

    def authorize(self, project_id: uuid.UUID, collections: list[str], permission: str) -> None:
        if project_id not in self.project_ids or not set(collections) <= self.collections or permission not in self.permissions:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="scope denied")


def hash_key(value: str) -> str:
    pepper = get_settings().api_key_pepper.encode()
    return hmac.new(pepper, value.encode(), hashlib.sha256).hexdigest()


async def principal(
    authorization: str = Header(), session: AsyncSession = Depends(get_session)
) -> Principal:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="bearer token required")
    row = await session.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(authorization[7:]), ApiKey.revoked.is_(False)))
    if row is None:
        raise HTTPException(status_code=401, detail="invalid API key")
    return Principal(row.tenant_id, frozenset(row.allowed_project_ids), frozenset(row.allowed_collections), frozenset(row.permissions))


async def admin(authorization: str = Header()) -> None:
    supplied = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(supplied, get_settings().admin_token):
        raise HTTPException(status_code=401, detail="invalid admin token")
