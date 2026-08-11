"""Add document, chunk, embedding and index identity metadata.

Revision ID: 0003
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_documents_scope_external_id",
        "documents",
        ["tenant_id", "project_id", "collection", "external_document_id"],
    )
    op.add_column(
        "document_versions",
        sa.Column(
            "parser_version", sa.String(length=100), server_default="text-v1", nullable=False
        ),
    )
    op.add_column(
        "document_versions",
        sa.Column(
            "chunker_version",
            sa.String(length=100),
            server_default="word-window-v1",
            nullable=False,
        ),
    )
    op.add_column(
        "document_versions",
        sa.Column(
            "embedding_model",
            sa.String(length=300),
            server_default="BAAI/bge-m3",
            nullable=False,
        ),
    )
    op.add_column(
        "document_versions",
        sa.Column(
            "embedding_revision",
            sa.String(length=200),
            server_default="default",
            nullable=False,
        ),
    )
    op.add_column(
        "document_versions",
        sa.Column(
            "index_version",
            sa.String(length=100),
            server_default="rag-chunks-v1",
            nullable=False,
        ),
    )

    op.add_column(
        "chunks",
        sa.Column("source_type", sa.String(length=100), server_default="text", nullable=False),
    )
    op.add_column(
        "chunks",
        sa.Column("source_id", sa.String(length=300), server_default="", nullable=False),
    )
    op.add_column("chunks", sa.Column("section_title", sa.String(length=500), nullable=True))
    op.add_column("chunks", sa.Column("start_offset", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("end_offset", sa.Integer(), nullable=True))
    op.add_column(
        "chunks",
        sa.Column(
            "chunker_version",
            sa.String(length=100),
            server_default="word-window-v1",
            nullable=False,
        ),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "index_version",
            sa.String(length=100),
            server_default="rag-chunks-v1",
            nullable=False,
        ),
    )
    op.execute(
        "UPDATE chunks AS c SET source_type = dv.document_type, "
        "source_id = dv.external_document_id, chunker_version = dv.chunker_version, "
        "index_version = dv.index_version FROM document_versions AS dv "
        "WHERE dv.id = c.document_version_id"
    )

    op.add_column(
        "chunk_embeddings",
        sa.Column(
            "backend",
            sa.String(length=100),
            server_default="sentence-transformers",
            nullable=False,
        ),
    )
    op.add_column(
        "chunk_embeddings",
        sa.Column("normalization", sa.String(length=40), server_default="l2", nullable=False),
    )
    op.add_column(
        "chunk_embeddings",
        sa.Column("embedding_dimension", sa.Integer(), server_default="1024", nullable=False),
    )
    op.create_unique_constraint(
        "uq_chunk_embeddings_identity",
        "chunk_embeddings",
        ["chunk_id", "model", "model_revision"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_chunk_embeddings_identity", "chunk_embeddings", type_="unique")
    op.drop_column("chunk_embeddings", "embedding_dimension")
    op.drop_column("chunk_embeddings", "normalization")
    op.drop_column("chunk_embeddings", "backend")
    op.drop_column("chunks", "index_version")
    op.drop_column("chunks", "chunker_version")
    op.drop_column("chunks", "end_offset")
    op.drop_column("chunks", "start_offset")
    op.drop_column("chunks", "section_title")
    op.drop_column("chunks", "source_id")
    op.drop_column("chunks", "source_type")
    op.drop_column("document_versions", "index_version")
    op.drop_column("document_versions", "embedding_revision")
    op.drop_column("document_versions", "embedding_model")
    op.drop_column("document_versions", "chunker_version")
    op.drop_column("document_versions", "parser_version")
    op.drop_constraint("uq_documents_scope_external_id", "documents", type_="unique")
