"""add class_weight_mode to model_basic

Adds class_weight_mode column to model_basic table to support balanced
cost-sensitive class weighting during model training and code generation.

Revision ID: l0m1n2o3p4q5
Revises: j8k9l0m1n2o3
Create Date: 2026-11-24 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "l0m1n2o3p4q5"
down_revision = "j8k9l0m1n2o3"
branch_labels = None
depends_on = None


def upgrade():
    """Add class_weight_mode column to model_basic table."""
    op.add_column("model_basic", sa.Column("class_weight_mode", sa.String(length=50), nullable=True))


def downgrade():
    """Remove class_weight_mode column from model_basic table."""
    op.drop_column("model_basic", "class_weight_mode")
