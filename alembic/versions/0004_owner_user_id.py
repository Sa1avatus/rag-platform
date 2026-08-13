"""Add owner_user_id for multi-user resource isolation.

Revision ID: 0004
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

SENTINEL_OWNER = "00000000-0000-0000-0000-000000000000"


def upgrade() -> None:
    # Step 1: Add owner_user_id as nullable to all three tables
    op.add_column(
        "documents",
        sa.Column("owner_user_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "document_versions",
        sa.Column("owner_user_id", UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "chunks",
        sa.Column("owner_user_id", UUID(as_uuid=True), nullable=True),
    )

    # Step 2: Backfill existing rows with sentinel UUID
    op.execute(
        f"UPDATE documents SET owner_user_id = '{SENTINEL_OWNER}' WHERE owner_user_id IS NULL"
    )
    op.execute(
        f"UPDATE document_versions SET owner_user_id = '{SENTINEL_OWNER}' WHERE owner_user_id IS NULL"
    )
    op.execute(
        f"UPDATE chunks SET owner_user_id = '{SENTINEL_OWNER}' WHERE owner_user_id IS NULL"
    )

    # Step 3: Alter columns to NOT NULL
    op.alter_column("documents", "owner_user_id", nullable=False)
    op.alter_column("document_versions", "owner_user_id", nullable=False)
    op.alter_column("chunks", "owner_user_id", nullable=False)

    # Step 4: Add indexes
    op.create_index("ix_documents_owner_user_id", "documents", ["owner_user_id"])
    op.create_index(
        "ix_document_versions_owner_user_id", "document_versions", ["owner_user_id"]
    )
    op.create_index("ix_chunks_owner_user_id", "chunks", ["owner_user_id"])

    # Step 5: Drop old unique constraint and create new one including owner_user_id
    op.drop_constraint(
        "uq_documents_scope_external_id", "documents", type_="unique"
    )
    op.create_unique_constraint(
        "uq_documents_scope_external_id",
        "documents",
        ["tenant_id", "project_id", "collection", "owner_user_id", "external_document_id"],
    )

    # Step 6: Update the ix_documents_scope index to include owner_user_id
    op.drop_index("ix_documents_scope", table_name="documents")
    op.create_index(
        "ix_documents_scope",
        "documents",
        ["tenant_id", "project_id", "collection", "owner_user_id"],
    )


def downgrade() -> None:
    # Reverse the index changes
    op.drop_index("ix_documents_scope", table_name="documents")
    op.create_index(
        "ix_documents_scope",
        "documents",
        ["tenant_id", "project_id", "collection"],
    )

    # Restore old unique constraint
    op.drop_constraint(
        "uq_documents_scope_external_id", "documents", type_="unique"
    )
    op.create_unique_constraint(
        "uq_documents_scope_external_id",
        "documents",
        ["tenant_id", "project_id", "collection", "external_document_id"],
    )

    # Drop indexes
    op.drop_index("ix_chunks_owner_user_id", table_name="chunks")
    op.drop_index("ix_document_versions_owner_user_id", table_name="document_versions")
    op.drop_index("ix_documents_owner_user_id", table_name="documents")

    # Drop columns
    op.drop_column("chunks", "owner_user_id")
    op.drop_column("document_versions", "owner_user_id")
    op.drop_column("documents", "owner_user_id")
