"""Request schemas for project CRUD endpoints."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _strip_str(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


class ProjectCreateRequest(BaseModel):
    """Request body for creating a new project."""

    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> Any:
        return _strip_str(value)


class ProjectUpdateRequest(BaseModel):
    """Request body for partially updating a project.

    Omitted fields are left unchanged. ``name`` may be omitted but not set to null.
    """

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("name cannot be null")
        return _strip_str(value)
