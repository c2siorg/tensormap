import uuid as uuid_pkg
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, CheckConstraint, Column, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.ml import ModelBasic
    from app.models.project import Project


class DataFile(SQLModel, table=True):
    """Uploaded dataset file (CSV or ZIP image archive)."""

    __tablename__ = "data_file"

    id: uuid_pkg.UUID = Field(sa_column=Column(PgUUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4))
    file_name: str = Field(max_length=100, nullable=False)
    file_type: str = Field(max_length=10, nullable=False)
    disk_name: str = Field(max_length=150, nullable=False)
    columns: list[str] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    row_count: int | None = Field(default=None, nullable=True)
    project_id: uuid_pkg.UUID | None = Field(
        sa_column=Column(PgUUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), index=True, nullable=True)
    )
    # GAP-2: file size and upload duration
    file_size_mb: float | None = Field(default=None, nullable=True)
    upload_duration_seconds: float | None = Field(default=None, nullable=True)

    # GAP-2: quality score and missing value info
    data_quality_score: int | None = Field(default=None, nullable=True)
    has_missing_values: bool = Field(default=False, nullable=False)
    missing_value_count: int = Field(default=0, nullable=False)
    missing_columns: list | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    column_nulls: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    # GAP-2: column type info
    column_dtypes: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    numeric_columns: list | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    categorical_columns: list | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    # GAP-2: duplicate info
    has_duplicates: bool = Field(default=False, nullable=False)
    duplicate_row_count: int = Field(default=0, nullable=False)

    # GAP-2: class distribution (for classification targets)
    class_distribution: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    is_imbalanced: bool = Field(default=False, nullable=False)
    imbalance_ratio: float | None = Field(default=None, nullable=True)

    # GAP-2: validation status
    validation_status: str = Field(default="pending", max_length=30, nullable=False)
    validation_messages: list | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )

    project: Optional["Project"] = Relationship(back_populates="files")
    model_basic: Optional["ModelBasic"] = Relationship(
        back_populates="file",
        sa_relationship_kwargs={"uselist": False, "cascade": "all,delete"},
    )
    target: Optional["DataProcess"] = Relationship(
        back_populates="file",
        sa_relationship_kwargs={"uselist": False, "cascade": "all,delete"},
    )


class DataProcess(SQLModel, table=True):
    """Target field assignment linking a DataFile to its prediction column."""

    __tablename__ = "data_process"

    id: int | None = Field(default=None, primary_key=True)
    target: str = Field(max_length=50, nullable=False)
    file_id: uuid_pkg.UUID = Field(
        sa_column=Column(PgUUID(as_uuid=True), ForeignKey("data_file.id"), index=True, nullable=False)
    )
    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )

    file: DataFile | None = Relationship(back_populates="target")


class ImageProperties(SQLModel, table=True):
    """Image dataset preprocessing parameters (size, batch, color/label mode)."""

    __tablename__ = "image_properties"

    id: uuid_pkg.UUID = Field(sa_column=Column(PgUUID(as_uuid=True), ForeignKey("data_file.id"), primary_key=True))
    image_size: int = Field(nullable=False)
    batch_size: int = Field(nullable=False)
    color_mode: str = Field(max_length=10, nullable=False)
    label_mode: str = Field(max_length=15, nullable=False)

    __table_args__ = (
        CheckConstraint("color_mode IN ('grayscale', 'rgb', 'rgba')", name="color_mode_check"),
        CheckConstraint("label_mode IN ('int', 'categorical', 'binary')", name="label_mode_check"),
    )
