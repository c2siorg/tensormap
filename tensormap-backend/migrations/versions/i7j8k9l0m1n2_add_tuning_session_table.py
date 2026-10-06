"""add tuning_session table and FK on training_job

Creates the tuning_session table for hyperparameter tuning (Week 11) and
adds a foreign-key constraint on the existing training_job.tuning_session_id
column pointing to tuning_session.id.

Revision ID: i7j8k9l0m1n2
Revises: h6i7j8k9l0m1
Create Date: 2026-08-09 22:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "i7j8k9l0m1n2"
down_revision = "h6i7j8k9l0m1"
branch_labels = None
depends_on = None


def upgrade():
    """Create tuning_session table and add FK on training_job.tuning_session_id."""
    op.create_table(
        "tuning_session",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column(
            "strategy",
            sa.Enum("grid", "random", native_enum=False, length=20),
            nullable=False,
        ),
        sa.Column("search_space", sa.JSON(), nullable=False),
        sa.Column("max_trials", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("metric", sa.String(length=50), nullable=False, server_default="val_accuracy"),
        sa.Column("direction", sa.String(length=20), nullable=False, server_default="maximize"),
        sa.Column("early_stop_threshold", sa.Float(), nullable=True),
        sa.Column("best_job_id", sa.String(length=36), nullable=True),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "completed", "failed", "cancelled", native_enum=False, length=20),
            nullable=False,
        ),
        sa.Column("total_trials", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_trials", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("early_stopped", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["model_basic.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["best_job_id"], ["training_job.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tuning_session_model_id", "tuning_session", ["model_id"])
    op.create_index("ix_tuning_session_status", "tuning_session", ["status"])

    # Add FK constraint on the existing training_job.tuning_session_id column.
    # batch_alter_table so SQLite, which cannot ALTER a constraint, rebuilds the table instead.
    with op.batch_alter_table("training_job") as batch_op:
        batch_op.create_foreign_key(
            "fk_training_job_tuning_session_id",
            "tuning_session",
            ["tuning_session_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade():
    """Drop FK on training_job.tuning_session_id and tuning_session table."""
    with op.batch_alter_table("training_job") as batch_op:
        batch_op.drop_constraint("fk_training_job_tuning_session_id", type_="foreignkey")
    op.drop_index("ix_tuning_session_status", table_name="tuning_session")
    op.drop_index("ix_tuning_session_model_id", table_name="tuning_session")
    op.drop_table("tuning_session")
