"""merge divergent heads

Unifies the two migration heads that diverged after the seed migration:
  - fd86601b847a (add graph_ir column)
  - 6a7b8c9d0e1f (add unique constraint on data_process.file_id)

Without this merge, ``alembic upgrade head`` is ambiguous (multiple heads).
This migration has no schema changes of its own.

Revision ID: a1f2b3c4d5e6
Revises: fd86601b847a, 6a7b8c9d0e1f
Create Date: 2026-06-27 00:00:00.000000

"""

# revision identifiers, used by Alembic.
revision = "a1f2b3c4d5e6"
down_revision = ("fd86601b847a", "6a7b8c9d0e1f")
branch_labels = None
depends_on = None


def upgrade():
    """No-op: merge node only."""
    pass


def downgrade():
    """No-op: merge node only."""
    pass
