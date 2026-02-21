import uuid as uuid_pkg
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Column, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.project import Project


class ModelBasic(SQLModel, table=True):
    __tablename__ = "model_basic"

    id: int | None = Field(default=None, primary_key=True)
    model_name: str = Field(max_length=50, nullable=False, unique=True)
    file_id: uuid_pkg.UUID = Field(
        sa_column=Column(PgUUID(as_uuid=True), ForeignKey("data_file.id"), index=True, nullable=False)
    )
    project_id: uuid_pkg.UUID | None = Field(
        sa_column=Column(PgUUID(as_uuid=True), ForeignKey("project.id"), index=True, nullable=True)
    )
    model_type: int = Field(nullable=False)
    target_field: str | None = Field(default=None, max_length=50)
    training_split: float = Field(nullable=False)
    optimizer: str = Field(max_length=50, nullable=False)
    metric: str = Field(max_length=50, nullable=False)
    epochs: int = Field(nullable=False)
    loss: str = Field(max_length=50, nullable=False)
    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )

    project: Optional["Project"] = Relationship(back_populates="models")
    file: Optional["DataFile"] = Relationship(back_populates="model_basic")
    configs: list["ModelConfigs"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={"cascade": "all,delete"},
    )
    results: list["ModelResults"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={"cascade": "all,delete"},
    )


class ModelConfigs(SQLModel, table=True):
    __tablename__ = "model_configs"

    id: int | None = Field(default=None, primary_key=True)
    parameter: str = Field(max_length=50, nullable=False)
    value: str = Field(max_length=50, nullable=False)
    model_id: int = Field(foreign_key="model_basic.id", index=True)
    created_on: datetime | None = Field(default=None, sa_column=Column(DateTime, server_default=func.now()))
    updated_on: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now(), onupdate=func.now())
    )

    model: ModelBasic | None = Relationship(back_populates="configs")


class ModelResults(SQLModel, table=True):
    __tablename__ = "model_results"

    id: int | None = Field(default=None, primary_key=True)
    model_id: int = Field(foreign_key="model_basic.id", index=True)
    epoch: int = Field(nullable=False)
    iteration: int = Field(nullable=False)
    metric: str = Field(max_length=50, nullable=False)
    value: float = Field(nullable=False)

    model: ModelBasic | None = Relationship(back_populates="results")


# Resolve forward references
from app.models.data import DataFile  # noqa: E402

ModelBasic.model_rebuild()
