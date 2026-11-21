"""add cascade delete to data_process foreign key

Adds ON DELETE CASCADE to data_process.file_id FK constraint so that
deleting a dataset file automatically deletes its associated target/data_process record.

Revision ID: j8k9l0m1n2o3
Revises: i7j8k9l0m1n2
Create Date: 2026-11-21 12:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "j8k9l0m1n2o3"
down_revision = "i7j8k9l0m1n2"
branch_labels = None
depends_on = None


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def upgrade():
    """Add ON DELETE CASCADE to data_process.file_id FK constraint."""
    if _is_sqlite():
        return

    # Drop existing FK constraint (no cascade)
    op.drop_constraint("data_process_file_id_fkey", "data_process", type_="foreignkey")

    # Re-create with ON DELETE CASCADE
    op.create_foreign_key(
        "data_process_file_id_fkey",
        "data_process",
        "data_file",
        ["file_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade():
    """Remove ON DELETE CASCADE from data_process.file_id FK constraint."""
    if _is_sqlite():
        return

    op.drop_constraint("data_process_file_id_fkey", "data_process", type_="foreignkey")
    op.create_foreign_key(
        "data_process_file_id_fkey",
        "data_process",
        "data_file",
        ["file_id"],
        ["id"],
    )
