"""add_model_evaluation_table

Revision ID: 6cbbfc386658
Revises: d4b7f1a03e82
Create Date: 2026-04-25 13:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "6cbbfc386658"
down_revision = "d4b7f1a03e82"
branch_labels = None
depends_on = None


def upgrade():
    # Create model_evaluation table (GAP-3)
    op.create_table(
        "model_evaluation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("training_run_id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("test_loss", sa.Float(), nullable=True),
        sa.Column("test_metric", sa.Float(), nullable=True),
        sa.Column("per_class_metrics", sa.JSON(), nullable=True),
        sa.Column("confusion_matrix", sa.JSON(), nullable=True),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("roc_curve_data", sa.JSON(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("r_squared", sa.Float(), nullable=True),
        sa.Column("created_on", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["model_basic.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["training_run_id"], ["model_training_run.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_evaluation_model_id", "model_evaluation", ["model_id"], unique=False)
    op.create_index("ix_model_evaluation_training_run_id", "model_evaluation", ["training_run_id"], unique=False)


def downgrade():
    op.drop_index("ix_model_evaluation_training_run_id", table_name="model_evaluation")
    op.drop_index("ix_model_evaluation_model_id", table_name="model_evaluation")
    op.drop_table("model_evaluation")
