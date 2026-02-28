"""add_graph_json_to_model_basic

Revision ID: d4b7f1a03e82
Revises: f1a2b3c4d5e6
Create Date: 2026-02-27 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d4b7f1a03e82"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "model_basic",
        sa.Column("graph_json", sa.JSON(), nullable=True),
    )


def downgrade():
    op.drop_column("model_basic", "graph_json")
