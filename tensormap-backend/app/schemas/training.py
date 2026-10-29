"""Request/response schemas for the training-job API."""

import uuid as uuid_pkg
from datetime import datetime

from pydantic import BaseModel, Field

from app.shared.enums import Optimizer


class TrainingStartRequest(BaseModel):
    """Request body for starting a training run.

    Only ``model_name`` is required; hyperparameters default to whatever was
    saved on the model via the training-config endpoint, so existing callers
    keep working. Any field provided here overrides the stored value for this
    run (recorded in the job's ``hyperparams``).
    """

    model_name: str = Field(min_length=1)
    project_id: uuid_pkg.UUID | None = None
    optimizer: Optimizer | None = None
    lr: float | None = Field(default=None, gt=0)
    epochs: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    early_stopping: bool | None = False
    early_stopping_patience: int | None = Field(default=5, ge=1)
    early_stopping_min_delta: float | None = Field(default=0.001, ge=0.0)
    early_stopping_restore_best_weights: bool | None = True
    reduce_lr_on_plateau: bool | None = False
    reduce_lr_patience: int | None = Field(default=3, ge=1)
    reduce_lr_factor: float | None = Field(default=0.2, gt=0.0, lt=1.0)
    reduce_lr_min_lr: float | None = Field(default=1e-6, ge=0.0)


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
