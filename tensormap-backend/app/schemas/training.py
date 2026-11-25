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
    reduce_lr_on_plateau: bool | None = None
    reduce_lr_patience: int | None = Field(default=None, ge=1)
    reduce_lr_factor: float | None = Field(default=None, gt=0.0, lt=1.0)
    reduce_lr_min_lr: float | None = Field(default=None, ge=0.0)
    reduce_lr_monitor: str | None = None


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
    reproducibility: dict | None = None


class JobReproducibilityResponse(BaseModel):
    """Reproducibility metadata for a single training job."""

    job_id: str
    has_snapshot: bool
    reproducibility: dict | None = None


class TrainingJobSummary(BaseModel):
    """Lightweight job descriptor for list views."""

    job_id: str
    model_id: int
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None


class LRFinderRequest(BaseModel):
    """Request parameters for the automated learning rate range test profiler."""

    model_name: str = Field(min_length=1)
    start_lr: float = Field(default=1e-6, gt=0.0)
    end_lr: float = Field(default=1.0, gt=0.0)
    num_batches: int = Field(default=50, ge=10, le=500)
    batch_size: int | None = Field(default=None, ge=1)


class LRFinderPoint(BaseModel):
    learning_rate: float
    loss: float


class LRFinderData(BaseModel):
    history: list[LRFinderPoint]
    suggested_lr: float | None = None
    min_loss: float | None = None
    diverged: bool = False
    num_batches_completed: int = 0


class LRFinderResponse(BaseModel):
    success: bool
    message: str
    data: LRFinderData
