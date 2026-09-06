"""Tuning-session API: start, inspect, cancel, and apply-best hyperparameter tuning.

A tuning session creates multiple training-job trials with different
hyperparameter combinations.  Progress is streamed via Socket.IO to a
``tuning:{session_id}`` room.
"""

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from sqlmodel import Session, select

from app.database import get_db
from app.exceptions import AppException
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.tuning_session import TuningSession, TuningStatus, TuningStrategy
from app.schemas.tuning import TuningStartRequest
from app.services.tuning_service import launch_tuning_session, tuning_service
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/model", tags=["tuning"])


def _envelope(message: str, data) -> dict:
    """Standard success envelope used across the API."""
    return {"success": True, "message": message, "data": data}


@router.post("/tuning/{model_name}", status_code=202)
async def start_tuning(
    model_name: str,
    body: TuningStartRequest,
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Create a tuning session and fire the background tuning loop.

    Returns 202 with the session ID, estimated duration, and trial count.
    """
    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if model is None:
        raise AppException(404, "Model not found")
    if model.file_id is None or model.epochs is None:
        raise AppException(
            400,
            "Training configuration not set. Please configure training parameters first.",
        )

    # Validate grid search size upfront.
    strategy = TuningStrategy(body.strategy)
    if strategy == TuningStrategy.GRID:
        combos = tuning_service.generate_grid_combinations(body.search_space)
        n_trials = len(combos)
    else:
        n_trials = body.max_trials

    # Validate the whole search space so malformed/misconfigured inputs return
    # a clear 400 instead of silently failing or leaving the session stuck.
    tuning_service.validate_search_space(body.search_space, strategy)

    # Create the tuning session.
    session_obj = TuningSession(
        model_id=model.id,
        strategy=strategy,
        search_space=body.search_space,
        max_trials=body.max_trials,
        metric=body.metric,
        direction=body.direction,
        early_stop_threshold=body.early_stop_threshold,
        total_trials=n_trials,
    )
    db.add(session_obj)
    db.commit()
    db.refresh(session_obj)

    # Estimate duration.
    estimated_seconds = tuning_service.estimate_session_duration(model.id, n_trials)

    logger.info(
        "Tuning session %s created for model '%s' (%s, %d trials)",
        session_obj.id,
        model_name,
        strategy.value,
        n_trials,
    )

    # Launch background tuning loop.
    loop = asyncio.get_running_loop()
    launch_tuning_session(session_obj.id, model_name, loop)

    return JSONResponse(
        status_code=202,
        content=_envelope(
            "Tuning session accepted",
            {
                "tuning_id": session_obj.id,
                "status": session_obj.status.value,
                "estimated_seconds": estimated_seconds,
                "n_trials": n_trials,
            },
        ),
    )


@router.get("/tuning/{tuning_id}")
def get_tuning_session(tuning_id: str, db: Session = Depends(get_db)) -> JSONResponse:
    """Return session status, all trial results, and best params."""
    ts = db.get(TuningSession, tuning_id)
    if ts is None:
        raise AppException(404, "Tuning session not found")

    # Gather trials (training_jobs linked to this session).
    trials = db.exec(
        select(TrainingJob).where(TrainingJob.tuning_session_id == tuning_id).order_by(TrainingJob.started_at)
    ).all()

    trial_data = []
    for job in trials:
        # Get the target metric for this job.
        metric_val = None
        if job.status.value == "completed":
            from app.models.training_metric import TrainingMetric

            row = db.exec(
                select(TrainingMetric)
                .where(TrainingMetric.job_id == job.id, TrainingMetric.metric_name == ts.metric)
                .order_by(TrainingMetric.epoch.desc())
                .limit(1)
            ).first()
            if row:
                metric_val = row.metric_value

        trial_data.append(
            {
                "job_id": job.id,
                "status": job.status.value,
                "hyperparams": job.hyperparams,
                "metric_value": metric_val,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }
        )

    # Get best job hyperparams.
    best_hyperparams = None
    if ts.best_job_id:
        best_job = db.get(TrainingJob, ts.best_job_id)
        if best_job:
            best_hyperparams = best_job.hyperparams

    data = {
        "tuning_id": ts.id,
        "model_id": ts.model_id,
        "strategy": ts.strategy.value,
        "search_space": ts.search_space,
        "max_trials": ts.max_trials,
        "metric": ts.metric,
        "direction": ts.direction,
        "early_stop_threshold": ts.early_stop_threshold,
        "best_job_id": ts.best_job_id,
        "best_hyperparams": best_hyperparams,
        "status": ts.status.value,
        "total_trials": ts.total_trials,
        "completed_trials": ts.completed_trials,
        "early_stopped": ts.early_stopped,
        "created_at": ts.created_at.isoformat() if ts.created_at else None,
        "completed_at": ts.completed_at.isoformat() if ts.completed_at else None,
        "trials": trial_data,
    }
    return JSONResponse(status_code=200, content=_envelope("Tuning session retrieved", data))


@router.delete("/tuning/{tuning_id}", status_code=204)
def cancel_tuning(tuning_id: str, db: Session = Depends(get_db)) -> Response:
    """Cancel a tuning session.

    Sets the session status to CANCELLED.  The tuning loop checks this at each
    trial boundary and stops.  Running trials are also cancelled via the
    existing training cancellation mechanism.
    """
    ts = db.get(TuningSession, tuning_id)
    if ts is None:
        raise AppException(404, "Tuning session not found")

    if ts.status in (TuningStatus.PENDING, TuningStatus.RUNNING):
        ts.status = TuningStatus.CANCELLED
        db.add(ts)
        db.commit()
        logger.info("Cancellation requested for tuning session %s", tuning_id)

        # Also cancel any currently running trial jobs.
        running_jobs = db.exec(
            select(TrainingJob).where(
                TrainingJob.tuning_session_id == tuning_id,
                TrainingJob.status.in_((TrainingStatus.PENDING, TrainingStatus.RUNNING)),
            )
        ).all()
        for job in running_jobs:
            job.status = TrainingStatus.CANCELLED
            db.add(job)
        if running_jobs:
            db.commit()

    return Response(status_code=204)


@router.post("/tuning/{tuning_id}/apply-best", status_code=200)
def apply_best_params(tuning_id: str, db: Session = Depends(get_db)) -> JSONResponse:
    """Copy the best trial's hyperparams to the model's training config.

    Uses the tuning service's ``apply_best_params`` which operates with its own
    session for atomicity.
    """
    ts = db.get(TuningSession, tuning_id)
    if ts is None:
        raise AppException(404, "Tuning session not found")
    if ts.status != TuningStatus.COMPLETED:
        raise AppException(400, "Tuning session is not completed yet")

    applied = tuning_service.apply_best_params(tuning_id)
    return JSONResponse(
        status_code=200,
        content=_envelope("Best hyperparams applied to model", {"applied_hyperparams": applied}),
    )
