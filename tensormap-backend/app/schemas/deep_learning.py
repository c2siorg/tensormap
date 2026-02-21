import uuid as uuid_pkg
from typing import Any

from pydantic import BaseModel


class ModelValidateRequest(BaseModel):
    model: dict[str, Any]
    code: dict[str, Any]
    project_id: uuid_pkg.UUID | None = None


class ModelNameRequest(BaseModel):
    model_name: str
