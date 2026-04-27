"""merge model_basic migration heads

Revision ID: e8c4d7f91a2b
Revises: c3a1d9e02f45, d4b7f1a03e82
Create Date: 2026-03-13 00:00:00.000000

"""

# revision identifiers, used by Alembic.
revision = "e8c4d7f91a2b"
down_revision = ("c3a1d9e02f45", "d4b7f1a03e82")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Merge the batch_size and graph_json schema branches."""


def downgrade() -> None:
    """Unmerge the batch_size and graph_json schema branches."""
