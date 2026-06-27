"""Training-job API: start, inspect, list, and cancel persistent training runs.

Supersedes the old stateless ``POST /model/run``. A run now returns immediately
(202) with a job id; progress and outcome are persisted and streamed via
Socket.IO rooms.

Note: TensorMap has no auth/user model, so jobs are scoped to a project rather
than an owner. ``get_job_or_404`` therefore returns 404 (not 403) for unknown
jobs; there is no per-user authorization to enforce.
"""

import asyncio

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response
from sqlmodel import Session, select

from app.database import get_db
from app.exceptions import AppException
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.schemas.training import TrainingStartRequest
from app.services.training_service import (
    check_concurrency_limit,
    create_training_job,
    get_job_metrics_grouped,
    get_latest_metrics,
    launch_training_job,
    update_job_status,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/model", tags=["training"])


def get_job_or_404(job_id: str, db: Session = Depends(get_db)) -> TrainingJob:
    """Fetch a training job or raise 404. Injected into job-scoped endpoints."""
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise AppException(404, "Training job not found")
    return job


def _envelope(message: str, data) -> dict:
    """Standard success envelope used across the API."""
    return {"success": True, "message": message, "data": data}


@router.post("/run", status_code=202)
async def start_training(request: TrainingStartRequest, db: Session = Depends(get_db)) -> JSONResponse:
    """Create a training job (PENDING), fire it in the background, return 202.

    Hyperparameters default to the model's saved training config; any provided
    in the request override them for this run.
    """
    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == request.model_name)).first()
    if model is None or (request.project_id is not None and model.project_id != request.project_id):
        raise AppException(404, "Model not found")
    if model.file_id is None or model.epochs is None:
        raise AppException(
            400,
            "Training configuration not set. Please configure training parameters first.",
        )

    # Reject early if we're already at the concurrency limit (raises 429).
    check_concurrency_limit(db)

    hyperparams = {
        "optimizer": request.optimizer or model.optimizer,
        "lr": request.lr,
        "epochs": request.epochs or model.epochs,
        "batch_size": request.batch_size or model.batch_size,
    }
    job = create_training_job(
        model_id=model.id,
        project_id=model.project_id,
        hyperparams=hyperparams,
        session=db,
    )

    logger.info("Training job %s created for model '%s'", job.id, request.model_name)
    loop = asyncio.get_running_loop()
    launch_training_job(request.model_name, job.id, loop)

    return JSONResponse(
        status_code=202,
        content=_envelope("Training job accepted", {"job_id": job.id, "status": job.status.value}),
    )


@router.get("/training-jobs")
def list_training_jobs(
    model_name: str = Query(...),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """List all jobs for a model, most recently started first."""
    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if model is None:
        raise AppException(404, "Model not found")

    jobs = db.exec(
        select(TrainingJob).where(TrainingJob.model_id == model.id).order_by(TrainingJob.started_at.desc())
    ).all()
    data = [
        {
            "job_id": j.id,
            "model_id": j.model_id,
            "status": j.status.value,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        }
        for j in jobs
    ]
    return JSONResponse(status_code=200, content=_envelope("Training jobs retrieved", data))


@router.get("/training-job/{job_id}")
def get_training_job(job: TrainingJob = Depends(get_job_or_404), db: Session = Depends(get_db)) -> JSONResponse:
    """Return a job's status, hyperparameters, and latest metrics."""
    data = {
        "job_id": job.id,
        "model_id": job.model_id,
        "status": job.status.value,
        "hyperparams": job.hyperparams,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_message": job.error_message,
        "latest_metrics": get_latest_metrics(job.id, db),
    }
    return JSONResponse(status_code=200, content=_envelope("Training job retrieved", data))


@router.get("/training-job/{job_id}/metrics")
def get_job_metrics(job: TrainingJob = Depends(get_job_or_404), db: Session = Depends(get_db)) -> JSONResponse:
    """Return the full per-epoch metric history for charts."""
    metrics = get_job_metrics_grouped(job.id, db)
    return JSONResponse(status_code=200, content=_envelope("Metrics retrieved", metrics))


@router.delete("/training-job/{job_id}", status_code=204)
def cancel_training_job(job: TrainingJob = Depends(get_job_or_404), db: Session = Depends(get_db)) -> Response:
    """Request cancellation of a job.

    Only sets status=CANCELLED; the running training loop stops itself at the
    next epoch boundary via ``CancellationCheckCallback``. Cancelling an
    already-finished job is a no-op (still 204).
    """
    if job.status in (TrainingStatus.PENDING, TrainingStatus.RUNNING):
        update_job_status(job.id, TrainingStatus.CANCELLED, db)
        logger.info("Cancellation requested for job %s", job.id)
    return Response(status_code=204)
