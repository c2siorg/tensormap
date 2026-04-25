"""add_data_quality_metrics_to_data_file

Revision ID: 6f6188c43604
Revises: d4b7f1a03e82
Create Date: 2026-04-25 12:44:04.496932

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "6f6188c43604"
down_revision = "d4b7f1a03e82"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("data_file", sa.Column("file_size_mb", sa.Float(), nullable=True))
    op.add_column("data_file", sa.Column("upload_duration_seconds", sa.Float(), nullable=True))
    op.add_column("data_file", sa.Column("data_quality_score", sa.Integer(), nullable=True))
    op.add_column("data_file", sa.Column("has_missing_values", sa.Boolean(), nullable=False, server_default="0"))
    op.add_column("data_file", sa.Column("missing_value_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("data_file", sa.Column("missing_columns", sa.JSON(), nullable=True))
    op.add_column("data_file", sa.Column("column_nulls", sa.JSON(), nullable=True))
    op.add_column("data_file", sa.Column("column_dtypes", sa.JSON(), nullable=True))
    op.add_column("data_file", sa.Column("numeric_columns", sa.JSON(), nullable=True))
    op.add_column("data_file", sa.Column("categorical_columns", sa.JSON(), nullable=True))
    op.add_column("data_file", sa.Column("has_duplicates", sa.Boolean(), nullable=False, server_default="0"))
    op.add_column("data_file", sa.Column("duplicate_row_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("data_file", sa.Column("class_distribution", sa.JSON(), nullable=True))
    op.add_column("data_file", sa.Column("is_imbalanced", sa.Boolean(), nullable=False, server_default="0"))
    op.add_column("data_file", sa.Column("imbalance_ratio", sa.Float(), nullable=True))
    op.add_column(
        "data_file",
        sa.Column(
            "validation_status",
            sqlmodel.sql.sqltypes.AutoString(length=30),
            nullable=False,
            server_default="pending",
        ),
    )
    op.add_column("data_file", sa.Column("validation_messages", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("data_file", "validation_messages")
    op.drop_column("data_file", "validation_status")
    op.drop_column("data_file", "imbalance_ratio")
    op.drop_column("data_file", "is_imbalanced")
    op.drop_column("data_file", "class_distribution")
    op.drop_column("data_file", "duplicate_row_count")
    op.drop_column("data_file", "has_duplicates")
    op.drop_column("data_file", "categorical_columns")
    op.drop_column("data_file", "numeric_columns")
    op.drop_column("data_file", "column_dtypes")
    op.drop_column("data_file", "column_nulls")
    op.drop_column("data_file", "missing_columns")
    op.drop_column("data_file", "missing_value_count")
    op.drop_column("data_file", "has_missing_values")
    op.drop_column("data_file", "data_quality_score")
    op.drop_column("data_file", "upload_duration_seconds")
    op.drop_column("data_file", "file_size_mb")
