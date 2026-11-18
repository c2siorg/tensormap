"""add class_weight_mode to model_basic

Revision ID: j8k9l0m1n2o3
Revises: i7j8k9l0m1n2
Create Date: 2026-11-05 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "j8k9l0m1n2o3"
down_revision = "i7j8k9l0m1n2"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("model_basic") as batch_op:
        batch_op.add_column(sa.Column("class_weight_mode", sa.String(length=50), nullable=True))


def downgrade():
    with op.batch_alter_table("model_basic") as batch_op:
        batch_op.drop_column("class_weight_mode")
