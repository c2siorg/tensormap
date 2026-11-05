"""Keras callback that captures optimal model weights and restores them at train end.

Tracks monitored training/validation metrics, checkpoints optimal weights in memory,
and restores the best weights to the model before serialization and holdout evaluation,
preventing overfitted final-epoch weights from corrupting exported models.
"""

import math
from collections.abc import Callable

import numpy as np
import tensorflow as tf
from sqlmodel import Session

from app.callbacks.metrics_callback import schedule_room_emit, training_event
from app.models.training_job import TrainingJob, TrainingStatus
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class BestModelCheckpointCallback(tf.keras.callbacks.Callback):
    """Monitors metrics, captures best weights, and restores them at training end.

    Args:
        job_id: Training job identifier.
        session_factory: Zero-arg callable returning a database Session.
        monitor: Metric name to monitor (default "val_loss", falls back to "loss" if not present).
        mode: Direction of improvement: "min", "max", or "auto".
        min_delta: Minimum magnitude change to qualify as an improvement.
        sio: Socket.IO server instance for event emission.
        loop: Asyncio event loop for scheduling emissions.
    """

    def __init__(
        self,
        job_id: str,
        session_factory: Callable[[], Session],
        monitor: str = "val_loss",
        mode: str = "auto",
        min_delta: float = 1e-4,
        sio=None,
        loop=None,
    ) -> None:
        super().__init__()
        self.job_id = job_id
        self.session_factory = session_factory
        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.sio = sio
        self.loop = loop

        if mode == "auto":
            if any(term in monitor.lower() for term in ("loss", "mae", "mse", "error")):
                self.mode = "min"
            else:
                self.mode = "max"
        else:
            self.mode = mode.lower()

        if self.mode == "min":
            self.best_value = float("inf")
            self._is_better = lambda current, best: current < (best - self.min_delta)
        else:
            self.best_value = float("-inf")
            self._is_better = lambda current, best: current > (best + self.min_delta)

        self.best_epoch: int = 1
        self.best_weights: list[np.ndarray] | None = None
        self.restored: bool = False

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """Inspect epoch logs, snapshot best weights on improvement, and emit checkpoint event."""
        logs = logs or {}

        # Resolve target metric with fallback
        target_metric = self.monitor
        if target_metric not in logs:
            if target_metric == "val_loss" and "loss" in logs:
                target_metric = "loss"
            elif target_metric == "val_accuracy" and ("accuracy" in logs or "acc" in logs):
                target_metric = "accuracy" if "accuracy" in logs else "acc"
            else:
                # Find any available loss/accuracy metric in logs
                for candidate in ("loss", "accuracy", "val_loss", "val_accuracy"):
                    if candidate in logs:
                        target_metric = candidate
                        break

        raw_val = logs.get(target_metric)
        if raw_val is None:
            return

        try:
            current_value = float(raw_val)
        except (ValueError, TypeError):
            return

        if math.isnan(current_value) or math.isinf(current_value):
            return

        if self._is_better(current_value, self.best_value) or self.best_weights is None:
            self.best_value = current_value
            self.best_epoch = epoch + 1
            if self.model is not None:
                self.best_weights = self.model.get_weights()

            logger.info(
                "Job %s achieved new optimal checkpoint at epoch %d (%s=%.4f)",
                self.job_id,
                self.best_epoch,
                target_metric,
                self.best_value,
            )

            if self.sio and self.loop:
                schedule_room_emit(
                    self.sio,
                    self.loop,
                    self.job_id,
                    training_event(
                        self.job_id,
                        "checkpoint",
                        epoch=self.best_epoch,
                        metric=target_metric,
                        value=self.best_value,
                    ),
                )

    def on_train_end(self, logs: dict = None) -> None:
        """Restore optimal weights to the model before serialization or holdout evaluation."""
        if self.best_weights is None or self.model is None:
            return

        # Check job status: only restore if job has not failed or been cancelled
        with self.session_factory() as session:
            job = session.get(TrainingJob, self.job_id)
            if job is not None and job.status in (TrainingStatus.CANCELLED, TrainingStatus.FAILED):
                logger.info("Job %s was cancelled or failed; skipping best weight restoration", self.job_id)
                return

        # Restore optimal weights in-memory
        try:
            self.model.set_weights(self.best_weights)
            self.restored = True
            logger.info(
                "Restored best weights for job %s from epoch %d (%s=%.4f)",
                self.job_id,
                self.best_epoch,
                self.monitor,
                self.best_value,
            )
        except Exception as e:
            logger.error("Failed to restore best weights for job %s: %s", self.job_id, e)
            return

        # Persist checkpoint metadata to TrainingJob.analysis_cache
        try:
            with self.session_factory() as session:
                job = session.get(TrainingJob, self.job_id)
                if job is not None:
                    cache = dict(job.analysis_cache or {})
                    cache["best_checkpoint"] = {
                        "epoch": self.best_epoch,
                        "metric": self.monitor,
                        "value": float(self.best_value),
                        "restored": True,
                    }
                    job.analysis_cache = cache
                    session.add(job)
                    session.commit()
        except Exception as e:
            logger.warning("Failed to persist checkpoint metadata for job %s: %s", self.job_id, e)
