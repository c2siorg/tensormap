"""make_model_name_unique_per_project

Revision ID: e1d7ac9f5b21
Revises: c3a1d9e02f45, d4b7f1a03e82
Create Date: 2026-03-14 11:05:00.000000
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "e1d7ac9f5b21"
down_revision = ("c3a1d9e02f45", "d4b7f1a03e82")
branch_labels = None
depends_on = None


def upgrade():
    # Old schema had a global UNIQUE(model_name). Drop it so names can repeat
    # across different projects.
    op.execute("ALTER TABLE model_basic DROP CONSTRAINT IF EXISTS model_basic_model_name_key")
    op.execute("ALTER TABLE model_basic DROP CONSTRAINT IF EXISTS uq_model_basic_model_name")

    # Keep names unique per project.
    op.create_index(
        "uq_model_basic_project_id_model_name",
        "model_basic",
        ["project_id", "model_name"],
        unique=True,
    )

    # For legacy/global models (project_id IS NULL), keep model_name unique.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_model_basic_null_project_model_name "
        "ON model_basic (model_name) WHERE project_id IS NULL"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS uq_model_basic_null_project_model_name")
    op.drop_index("uq_model_basic_project_id_model_name", table_name="model_basic")
    op.create_unique_constraint("uq_model_basic_model_name", "model_basic", ["model_name"])
