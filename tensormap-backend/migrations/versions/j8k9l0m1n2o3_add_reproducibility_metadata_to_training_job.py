"""add reproducibility_metadata to training_job

Adds a nullable JSON column to store reproducibility metadata for training jobs.
This metadata includes dataset fingerprint, training config, randomness state,
and runtime versions captured before training starts.

Revision ID: j8k9l0m1n2o3
Revises: i7j8k9l0m1n2
Create Date: 2026-11-07 23:45:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "j8k9l0m1n2o3"
down_revision = "i7j8k9l0m1n2"
branch_labels = None
depends_on = None


def upgrade():
    """Add reproducibility_metadata JSON column to training_job table."""
    op.add_column(
        "training_job",
        sa.Column("reproducibility_metadata", sa.JSON(), nullable=True),
    )


def downgrade():
    """Remove reproducibility_metadata column from training_job table."""
    op.drop_column("training_job", "reproducibility_metadata")
