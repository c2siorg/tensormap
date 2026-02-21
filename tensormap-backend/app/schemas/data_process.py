"""Request schemas for data processing endpoints."""

import uuid as uuid_pkg

from pydantic import BaseModel


class TargetAddRequest(BaseModel):
    """Request body for setting a dataset's target field."""

    target: str
    file_id: uuid_pkg.UUID


class TransformationItem(BaseModel):
    """A single column transformation (e.g., one-hot encoding, drop)."""

    transformation: str
    feature: str


class PreprocessRequest(BaseModel):
    """Request body for applying preprocessing transformations to a dataset."""

    transformations: list[TransformationItem]
