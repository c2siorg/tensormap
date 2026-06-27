"""Keras callback that persists per-epoch metrics and streams them to a job room.

Replaces the old ``CustomProgressBar`` text emissions: instead of broadcasting
formatted strings to every client, it writes structured metrics to the
``training_metric`` table and emits them to the Socket.IO room for this job only.
"""

import asyncio
from datetime import UTC, datetime

import tensorflow as tf

from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric
from app.shared.constants import SOCKETIO_DL_NAMESPACE, SOCKETIO_LISTENER
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def schedule_room_emit(sio_instance, loop, job_id: str, data: dict) -> None:
    """Thread-safely emit ``data`` to a job's Socket.IO room from a worker thread.

    Shared by the per-epoch metrics emit and the terminal status emit (and the
    failure path in ``model_run``). A dropped event must never crash training.
    """
    if loop is None or not loop.is_running():
        logger.warning("No running event loop for room emit (job %s)", job_id)
        return
    future = asyncio.run_coroutine_threadsafe(
        sio_instance.emit(SOCKETIO_LISTENER, data, room=job_id, namespace=SOCKETIO_DL_NAMESPACE),
        loop,
    )
    try:
        future.result(timeout=5)
    except Exception:  # noqa: BLE001 - a dropped progress event must not kill training
        logger.warning("Failed to emit to room for job %s", job_id)


class MetricsCallback(tf.keras.callbacks.Callback):
    """Persist structured metrics to the DB per epoch and emit them to a room.

    Args:
        job_id: The training job these metrics belong to.
        session_factory: Zero-arg callable returning a fresh DB ``Session``
            (callbacks run in a worker thread, so they own their sessions).
        sio_instance: The Socket.IO server used to emit progress.
        loop: The main event loop, for thread-safe coroutine scheduling.
    """

    def __init__(self, job_id, session_factory, sio_instance, loop) -> None:
        super().__init__()
        self.job_id = job_id
        self.session_factory = session_factory
        self.sio = sio_instance
        self.loop = loop
        self.start_time = None

    def on_train_begin(self, logs: dict = None) -> None:
        """Mark the job RUNNING and record the start time."""
        self.start_time = _utcnow()
        with self.session_factory() as session:
            job = session.get(TrainingJob, self.job_id)
            if job is not None:
                job.status = TrainingStatus.RUNNING
                job.started_at = self.start_time
                session.add(job)
                session.commit()

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """Persist this epoch's metrics and emit them to the job's room."""
        logs = logs or {}
        payload = {
            "epoch": epoch + 1,
            "loss": _as_float(logs.get("loss")),
            "accuracy": _as_float(logs.get("accuracy", logs.get("acc"))),
            "val_loss": _as_float(logs.get("val_loss")) if "val_loss" in logs else None,
            "val_accuracy": (
                _as_float(logs.get("val_accuracy", logs.get("val_acc")))
                if ("val_accuracy" in logs or "val_acc" in logs)
                else None
            ),
        }

        # 1. Persist each present metric as its own row.
        with self.session_factory() as session:
            for name, value in payload.items():
                if name != "epoch" and value is not None:
                    session.add(
                        TrainingMetric(
                            job_id=self.job_id,
                            epoch=epoch + 1,
                            metric_name=name,
                            metric_value=value,
                        )
                    )
            session.commit()

        # 2. Emit to this job's room only (no global broadcast).
        self._emit({"type": "metrics", **payload})

    def on_train_end(self, logs: dict = None) -> None:
        """Mark the job COMPLETED — unless it was cancelled/failed meanwhile.

        A cancelled job sets ``model.stop_training`` which ends ``fit`` cleanly
        and still fires ``on_train_end``; guarding on the current status keeps us
        from overwriting CANCELLED/FAILED with COMPLETED. Emits a terminal status
        event so subscribers know the run is over.
        """
        final_status = None
        with self.session_factory() as session:
            job = session.get(TrainingJob, self.job_id)
            if job is not None:
                if job.status == TrainingStatus.RUNNING:
                    job.status = TrainingStatus.COMPLETED
                    job.completed_at = _utcnow()
                    session.add(job)
                    session.commit()
                final_status = job.status.value
        if final_status is not None:
            self._emit({"type": "status", "status": final_status})

    def _emit(self, data: dict) -> None:
        """Schedule a thread-safe emit to the job room on the main loop."""
        schedule_room_emit(self.sio, self.loop, self.job_id, data)


def _as_float(value) -> float | None:
    """Coerce a Keras log value to float, or None if it is missing."""
    if value is None:
        return None
    return float(value)
