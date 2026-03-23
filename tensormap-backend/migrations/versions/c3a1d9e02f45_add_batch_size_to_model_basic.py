"""add_batch_size_to_model_basic

Revision ID: c3a1d9e02f45
Revises: f1a2b3c4d5e6
Create Date: 2026-02-27 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c3a1d9e02f45"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "model_basic",
        sa.Column("batch_size", sa.Integer(), nullable=True),
    )


def downgrade():
    op.drop_column("model_basic", "batch_size")
