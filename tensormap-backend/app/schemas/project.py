"""Request schemas for project CRUD endpoints."""

from pydantic import BaseModel, Field


class ProjectCreateRequest(BaseModel):
    """Request body for creating a new project."""

    name: str = Field(max_length=100)
    description: str | None = Field(default=None, max_length=500)


class ProjectUpdateRequest(BaseModel):
    """Request body for partially updating a project."""

    name: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=500)
