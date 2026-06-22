"""add_unique_constraint_on_data_process_file_id

Revision ID: 6a7b8c9d0e1f
Revises: f4a8c2e91b37
Create Date: 2026-06-22 12:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "6a7b8c9d0e1f"
down_revision = "f4a8c2e91b37"
branch_labels = None
depends_on = None


def upgrade():
    # Deduplicate before adding constraint: keep the most recently updated
    # row per file_id (NULLS LAST so rows with NULL updated_on are deprioritised).
    op.execute(
        """
        DELETE FROM data_process
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                    ROW_NUMBER() OVER (
                        PARTITION BY file_id
                        ORDER BY updated_on DESC NULLS LAST, created_on DESC NULLS LAST, id DESC
                    ) AS rn
                FROM data_process
            ) sub
            WHERE rn > 1
        )
        """
    )

    # The model previously used index=True on file_id, which created a
    # non-unique index.  The unique constraint below creates its own
    # btree index, so drop the redundant old one.
    op.drop_index("ix_data_process_file_id", table_name="data_process")

    op.create_unique_constraint("uq_data_process_file_id", "data_process", ["file_id"])


def downgrade():
    op.drop_constraint("uq_data_process_file_id", "data_process", type_="unique")
    op.create_index("ix_data_process_file_id", "data_process", ["file_id"], unique=False)
