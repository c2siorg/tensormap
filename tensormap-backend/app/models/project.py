import uuid as uuid_pkg
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models.data import DataFile
    from app.models.ml import ModelBasic


class Project(SQLModel, table=True):
    __tablename__ = "project"

    id: uuid_pkg.UUID = Field(sa_column=Column(PgUUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4))
    name: str = Field(max_length=100, index=True)
    description: str | None = Field(default=None, max_length=500)
    created_on: datetime = Field(default_factory=datetime.utcnow)
    updated_on: datetime = Field(default_factory=datetime.utcnow)

    files: list["DataFile"] = Relationship(
        back_populates="project",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    models: list["ModelBasic"] = Relationship(
        back_populates="project",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
