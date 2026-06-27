"""add training_job and training_metric tables

Creates the persistence layer for training jobs (Week 4):
  - training_job: one row per training run, tracking lifecycle + hyperparams
  - training_metric: long-form per-epoch metrics, indexed for chart queries

Revision ID: b2c3d4e5f6a7
Revises: a1f2b3c4d5e6
Create Date: 2026-06-27 00:05:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PgUUID

# revision identifiers, used by Alembic.
revision = "b2c3d4e5f6a7"
down_revision = "a1f2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade():
    """Create training_job and training_metric tables."""
    op.create_table(
        "training_job",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("project_id", PgUUID(as_uuid=True), nullable=True),
        # native_enum=False -> portable VARCHAR + CHECK (no Postgres ENUM type).
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "running",
                "completed",
                "failed",
                "cancelled",
                native_enum=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("hyperparams", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.String(length=2000), nullable=True),
        sa.Column("analysis_cache", sa.JSON(), nullable=True),
        sa.Column("last_export_download_at", sa.DateTime(), nullable=True),
        sa.Column("tuning_session_id", sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["model_basic.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["project.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_job_model_id", "training_job", ["model_id"])
    op.create_index("ix_training_job_project_id", "training_job", ["project_id"])
    op.create_index("ix_training_job_status", "training_job", ["status"])

    op.create_table(
        "training_metric",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("metric_name", sa.String(length=50), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["training_job.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_metric_job_id", "training_metric", ["job_id"])
    op.create_index("ix_metric_job_epoch", "training_metric", ["job_id", "epoch"])


def downgrade():
    """Drop training_metric and training_job tables."""
    op.drop_index("ix_metric_job_epoch", table_name="training_metric")
    op.drop_index("ix_training_metric_job_id", table_name="training_metric")
    op.drop_table("training_metric")
    op.drop_index("ix_training_job_status", table_name="training_job")
    op.drop_index("ix_training_job_project_id", table_name="training_job")
    op.drop_index("ix_training_job_model_id", table_name="training_job")
    op.drop_table("training_job")
