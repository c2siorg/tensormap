"""add cascade delete to data_process fk

Revision ID: l0m1n2o3p4q5
Revises: k9l0m1n2o3p4
Create Date: 2026-04-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "l0m1n2o3p4q5"
down_revision: str | Sequence[str] | None = "k9l0m1n2o3p4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FALLBACK_FK_NAME = "data_process_file_id_fkey"


def _resolve_file_id_fk_name() -> str:
    """Inspect the existing data_process.file_id foreign key constraint name."""
    inspector = sa.inspect(op.get_bind())
    for fk in inspector.get_foreign_keys("data_process"):
        if fk.get("referred_table") == "data_file" and fk.get("constrained_columns") == ["file_id"] and fk.get("name"):
            return str(fk["name"])
    return _FALLBACK_FK_NAME


def upgrade() -> None:
    """Add ON DELETE CASCADE to data_process.file_id foreign key."""
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return

    fk_name = _resolve_file_id_fk_name()
    op.drop_constraint(fk_name, "data_process", type_="foreignkey")
    op.create_foreign_key(
        _FALLBACK_FK_NAME,
        "data_process",
        "data_file",
        ["file_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Remove ON DELETE CASCADE from data_process.file_id foreign key."""
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return

    fk_name = _resolve_file_id_fk_name()
    op.drop_constraint(fk_name, "data_process", type_="foreignkey")
    op.create_foreign_key(
        _FALLBACK_FK_NAME,
        "data_process",
        "data_file",
        ["file_id"],
        ["id"],
    )
