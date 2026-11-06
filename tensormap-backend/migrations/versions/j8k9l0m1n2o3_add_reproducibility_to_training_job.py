"""add reproducibility to training_job

Adds reproducibility column to training_job table to store training run
reproducibility snapshot metadata.

Revision ID: j8k9l0m1n2o3
Revises: i7j8k9l0m1n2
Create Date: 2026-11-06 04:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "j8k9l0m1n2o3"
down_revision = "i7j8k9l0m1n2"
branch_labels = None
depends_on = None


def upgrade():
    """Add reproducibility column to training_job table."""
    op.add_column("training_job", sa.Column("reproducibility", sa.JSON(), nullable=True))


def downgrade():
    """Remove reproducibility column from training_job table."""
    op.drop_column("training_job", "reproducibility")
