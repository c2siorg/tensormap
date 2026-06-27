"""Business logic for persistent training jobs.

Replaces the old stateless fire-and-block training flow. A job is created
(PENDING) by the run endpoint, executed in a background thread (RUNNING), and
ends COMPLETED / FAILED / CANCELLED. Metrics are persisted per epoch by
``MetricsCallback`` and read back here for the API and Socket.IO catch-up.
"""

import asyncio
import os
import uuid as uuid_pkg
from datetime import UTC, datetime

from sqlmodel import Session, select

from app.database import engine
from app.exceptions import AppException
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

# Maximum number of jobs allowed to run concurrently. Counted across PENDING and
# RUNNING so a burst of quick requests can't all slip past before any flips to
# RUNNING (the spec's RUNNING-only check has that race).
MAX_CONCURRENT_TRAINING_JOBS = int(os.getenv("MAX_CONCURRENT_TRAINING_JOBS", "3"))

# Strong references to in-flight background tasks. Without this, asyncio may
# garbage-collect a fire-and-forget task mid-run (see asyncio.create_task docs).
_background_tasks: set[asyncio.Task] = set()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def make_session() -> Session:
    """Create a new DB session bound to the app engine.

    Used as a session *factory* by background threads and Keras callbacks, which
    run outside the request lifecycle and must not share the request's session.
    """
    return Session(engine)


def create_training_job(
    model_id: int,
    project_id: uuid_pkg.UUID | None,
    hyperparams: dict | None,
    session: Session,
) -> TrainingJob:
    """Persist a new job in the PENDING state and return it."""
    job = TrainingJob(
        model_id=model_id,
        project_id=project_id,
        hyperparams=hyperparams,
        status=TrainingStatus.PENDING,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def update_job_status(
    job_id: str,
    status: TrainingStatus,
    session: Session,
    error_message: str | None = None,
) -> None:
    """Transition a job to ``status``, stamping timestamps and error as needed."""
    job = session.get(TrainingJob, job_id)
    if job is None:
        logger.warning("update_job_status: job %s not found", job_id)
        return
    job.status = status
    if status == TrainingStatus.RUNNING and job.started_at is None:
        job.started_at = _utcnow()
    if status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
        job.completed_at = _utcnow()
    if error_message is not None:
        job.error_message = error_message[:2000]
    session.add(job)
    session.commit()


def get_job_metrics_grouped(job_id: str, session: Session) -> list[dict]:
    """Return the full metric history grouped into one dict per epoch.

    Shape: ``[{"epoch": 1, "loss": .., "accuracy": .., "val_loss": ..,
    "val_accuracy": ..}, ...]`` ordered by epoch ascending. Missing metrics for
    an epoch are omitted from that epoch's dict.
    """
    rows = session.exec(
        select(TrainingMetric).where(TrainingMetric.job_id == job_id).order_by(TrainingMetric.epoch)
    ).all()
    by_epoch: dict[int, dict] = {}
    for row in rows:
        by_epoch.setdefault(row.epoch, {"epoch": row.epoch})[row.metric_name] = row.metric_value
    return [by_epoch[e] for e in sorted(by_epoch)]


def get_latest_metrics(job_id: str, session: Session) -> dict:
    """Return the most recent epoch's metrics, or ``{}`` if none recorded yet."""
    grouped = get_job_metrics_grouped(job_id, session)
    return grouped[-1] if grouped else {}


def check_concurrency_limit(session: Session) -> None:
    """Raise HTTP 429 if the active-job limit is already reached."""
    active = session.exec(
        select(TrainingJob).where(TrainingJob.status.in_((TrainingStatus.PENDING, TrainingStatus.RUNNING)))
    ).all()
    if len(active) >= MAX_CONCURRENT_TRAINING_JOBS:
        raise AppException(
            429,
            f"Max concurrent training jobs reached ({MAX_CONCURRENT_TRAINING_JOBS}). Try again shortly.",
        )


def orphan_recovery(session: Session | None = None) -> int:
    """Mark jobs left PENDING/RUNNING by a crash as FAILED.

    Called on startup: any job still 'active' in the DB cannot actually be
    running (the process that owned it is gone), so it is failed with a clear
    message. Returns the number of jobs recovered.
    """
    own_session = session is None
    session = session or make_session()
    try:
        orphans = session.exec(
            select(TrainingJob).where(TrainingJob.status.in_((TrainingStatus.PENDING, TrainingStatus.RUNNING)))
        ).all()
        for job in orphans:
            job.status = TrainingStatus.FAILED
            job.error_message = "Recovered orphaned job after restart"
            job.completed_at = _utcnow()
            session.add(job)
        if orphans:
            session.commit()
            logger.info("Orphan recovery: marked %d stale job(s) as failed", len(orphans))
        return len(orphans)
    finally:
        if own_session:
            session.close()


def _train_in_thread(model_name: str, job_id: str, loop: asyncio.AbstractEventLoop) -> None:
    """Run a training job in a worker thread with its own DB session.

    Owns the full lifecycle for the background run: status flips happen inside
    ``model_run`` via the callbacks; here we only guarantee the job is marked
    FAILED if training raises before/around the callbacks.
    """
    # Imported lazily to avoid importing TensorFlow at app startup.
    from app.services.model_run import model_run

    with make_session() as session:
        try:
            model_run(model_name, session, loop=loop, job_id=job_id)
        except Exception as e:  # noqa: BLE001 - last-resort guard for the background thread
            logger.exception("Background training failed for job %s: %s", job_id, e)
            try:
                update_job_status(job_id, TrainingStatus.FAILED, session, error_message=str(e))
            except Exception:  # noqa: BLE001
                logger.exception("Failed to mark job %s as failed", job_id)


def launch_training_job(model_name: str, job_id: str, loop: asyncio.AbstractEventLoop) -> None:
    """Schedule a training job to run in the background and return immediately.

    Uses ``asyncio.to_thread`` (training is blocking/CPU-bound) wrapped in a
    tracked task so the event loop keeps a strong reference until it finishes.
    """
    task = asyncio.create_task(asyncio.to_thread(_train_in_thread, model_name, job_id, loop))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
