"""Request/response schemas for the training-job API."""

import uuid as uuid_pkg
from datetime import datetime

from pydantic import BaseModel, Field


class TrainingStartRequest(BaseModel):
    """Request body for starting a training run.

    Only ``model_name`` is required; hyperparameters default to whatever was
    saved on the model via the training-config endpoint, so existing callers
    keep working. Any field provided here overrides the stored value for this
    run (recorded in the job's ``hyperparams``).
    """

    model_name: str = Field(min_length=1)
    project_id: uuid_pkg.UUID | None = None
    optimizer: str | None = None
    lr: float | None = Field(default=None, gt=0)
    epochs: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)


class TrainingJobResponse(BaseModel):
    """Full detail for a single training job."""

    job_id: str
    model_id: int
    status: str
    hyperparams: dict | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    latest_metrics: dict | None = None


class TrainingJobSummary(BaseModel):
    """Lightweight job descriptor for list views."""

    job_id: str
    model_id: int
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
