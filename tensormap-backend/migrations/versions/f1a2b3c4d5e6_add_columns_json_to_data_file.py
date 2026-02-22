"""add columns json and row_count to data_file

Revision ID: f1a2b3c4d5e6
Revises: f4a8c2e91b37
Create Date: 2026-02-22 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f1a2b3c4d5e6"
down_revision = "f4a8c2e91b37"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("data_file", sa.Column("columns", sa.JSON(), nullable=True))
    op.add_column("data_file", sa.Column("row_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("data_file", "row_count")
    op.drop_column("data_file", "columns")
