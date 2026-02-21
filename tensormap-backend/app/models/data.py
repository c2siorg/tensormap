import uuid as uuid_pkg
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.ml import ModelBasic
    from app.models.project import Project


class DataFile(SQLModel, table=True):
    __tablename__ = "data_file"

    id: uuid_pkg.UUID = Field(sa_column=Column(PgUUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4))
    file_name: str = Field(max_length=100, nullable=False)
    file_type: str = Field(max_length=10, nullable=False)
    project_id: uuid_pkg.UUID | None = Field(
        sa_column=Column(PgUUID(as_uuid=True), ForeignKey("project.id"), index=True, nullable=True)
    )
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
