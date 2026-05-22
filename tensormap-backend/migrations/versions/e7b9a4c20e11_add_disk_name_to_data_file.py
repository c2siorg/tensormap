"""add disk_name to data_file

Revision ID: e7b9a4c20e11
Revises: d4b7f1a03e82
Create Date: 2026-05-22 12:00:00.000000

The upload service stores files under a unique on-disk name (e.g.
``iris_sample_abc12345.csv``) to prevent collisions when two users upload
the same filename. The DataFile model gained a ``disk_name`` column to
record that name, but the corresponding schema migration was never
committed. This migration adds the column, backfills existing rows by
falling back to ``file_name.file_type`` (the pre-collision-fix naming
convention, which still matches the seeded ``iris_sample.csv``), and
then enforces NOT NULL.
"""

import sqlalchemy as sa
from alembic import op

revision = "e7b9a4c20e11"
down_revision = "d4b7f1a03e82"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("data_file", sa.Column("disk_name", sa.String(length=150), nullable=True))
    op.execute(
        """
        UPDATE data_file
        SET disk_name = file_name || '.' || file_type
        WHERE disk_name IS NULL
        """
    )
    op.alter_column("data_file", "disk_name", nullable=False)


def downgrade() -> None:
    op.drop_column("data_file", "disk_name")
