"""Initial schema.

Revision ID: 0001
"""
from alembic import op

from rag_platform.db.models import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_hnsw ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)")


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
