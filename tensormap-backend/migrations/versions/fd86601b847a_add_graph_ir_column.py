"""add_graph_ir_column

Revision ID: fd86601b847a
Revises: e7b9a4c20e11
Create Date: 2026-06-17 10:48:15.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "fd86601b847a"
down_revision = "e7b9a4c20e11"
branch_labels = None
depends_on = None


def upgrade():
    """Add graph_ir JSON column to model_basic for IRGraph storage.

    This migration adds a nullable JSON column that will eventually replace
    the model_configs key-value table. During the dual-write period, both
    storage methods coexist.
    """
    op.add_column(
        "model_basic",
        sa.Column("graph_ir", sa.JSON(), nullable=True),
    )


def downgrade():
    """Remove graph_ir column from model_basic."""
    op.drop_column("model_basic", "graph_ir")
