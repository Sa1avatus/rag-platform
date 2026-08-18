"""Switch embedding backend default from sentence-transformers to onnxruntime.

Revision ID: 0005
"""

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "document_versions",
        "backend",
        server_default="onnxruntime",
        existing_type=sa.String(100),
    )


def downgrade() -> None:
    op.alter_column(
        "document_versions",
        "backend",
        server_default="sentence-transformers",
        existing_type=sa.String(100),
    )
