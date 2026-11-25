"""Adaptive learning rate schedulers and callbacks (Issue #558).

Provides dynamic learning rate scheduling via ReduceLROnPlateau,
tracking learning rate transitions in real-time, emitting Socket.IO events,
and logging learning rate into epoch logs for persistent metric tracking.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import numpy as np
import tensorflow as tf

from app.callbacks.metrics_callback import schedule_room_emit

logger = logging.getLogger(__name__)


def _extract_lr(optimizer: Any) -> float | None:
    """Safely extract float learning rate value from a Keras optimizer."""
    if optimizer is None:
        return None
    try:
        lr_attr = getattr(optimizer, "learning_rate", getattr(optimizer, "lr", None))
        if lr_attr is None:
            return None
        if hasattr(lr_attr, "numpy"):
            return float(lr_attr.numpy())
        if isinstance(lr_attr, tf.keras.optimizers.schedules.LearningRateSchedule) or callable(lr_attr):
            step = getattr(optimizer, "iterations", 0)
            res = lr_attr(step)
            if hasattr(res, "numpy"):
                return float(res.numpy())
            return float(res)
        val = np.array(lr_attr)
        return float(val)
    except Exception:  # noqa: BLE001
        return None


class ReduceLROnPlateauCallback(tf.keras.callbacks.ReduceLROnPlateau):
    """Dynamic learning rate reduction callback with real-time Socket.IO notifications.

    Subclasses Keras ReduceLROnPlateau to:
    1. Maintain current learning rate in epoch metrics logs so MetricsCallback persists it.
    2. Detect learning rate reduction events and emit structured Socket.IO events to the job room.
    3. Safely handle models without validation datasets by falling back to training loss.
    """

    def __init__(
        self,
        job_id: str | None = None,
        sio_instance: Any = None,
        loop: Any = None,
        monitor: str = "val_loss",
        factor: float = 0.2,
        patience: int = 5,
        min_lr: float = 0.0,
        min_delta: float = 1e-4,
        mode: str = "auto",
        cooldown: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            min_delta=min_delta,
            mode=mode,
            cooldown=cooldown,
            **kwargs,
        )
        self.job_id = job_id
        self.sio = sio_instance
        self.loop = loop
        self._last_lr: float | None = None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        super().on_train_begin(logs)
        with contextlib.suppress(Exception):
            if hasattr(self.model, "optimizer"):
                self._last_lr = _extract_lr(self.model.optimizer)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        old_lr = self._last_lr

        super().on_epoch_end(epoch, logs)

        current_lr = None
        if hasattr(self.model, "optimizer"):
            current_lr = _extract_lr(self.model.optimizer)

        if current_lr is not None:
            self._last_lr = current_lr
            logs["lr"] = current_lr

            # Emit room event if learning rate was decayed on plateau
            if old_lr is not None and current_lr < old_lr and self.job_id and self.sio and self.loop:
                schedule_room_emit(
                    self.sio,
                    self.loop,
                    self.job_id,
                    {
                        "job_id": self.job_id,
                        "type": "learning_rate_reduced",
                        "epoch": epoch + 1,
                        "old_lr": old_lr,
                        "new_lr": current_lr,
                    },
                )


def create_lr_scheduler_callback(
    job_id: str | None = None,
    hyperparams: dict[str, Any] | None = None,
    has_validation: bool = True,
    sio_instance: Any = None,
    loop: Any = None,
) -> ReduceLROnPlateauCallback | None:
    """Create a ReduceLROnPlateau callback configured from hyperparameters.

    If has_validation is False and monitor defaults to or is configured with a
    'val_' prefix (e.g. 'val_loss'), automatically adjusts monitor to 'loss'
    to prevent callbacks from silently stalling on datasets without validation data.
    """
    hp = hyperparams or {}
    enabled = hp.get("reduce_lr_on_plateau") or hp.get("enable_lr_scheduler")
    if not enabled:
        return None

    monitor = hp.get("reduce_lr_monitor") or ("val_loss" if has_validation else "loss")
    if not has_validation and str(monitor).startswith("val_"):
        monitor = monitor.replace("val_", "", 1)
        logger.info("Validation data absent; adjusted LR scheduler monitor to '%s'", monitor)

    raw_factor = hp.get("reduce_lr_factor")
    factor = float(raw_factor) if raw_factor is not None else 0.2
    raw_patience = hp.get("reduce_lr_patience")
    patience = int(raw_patience) if raw_patience is not None else 5
    raw_min_lr = hp.get("reduce_lr_min_lr")
    min_lr = float(raw_min_lr) if raw_min_lr is not None else 1e-6
    raw_min_delta = hp.get("reduce_lr_min_delta")
    min_delta = float(raw_min_delta) if raw_min_delta is not None else 1e-4
    mode = str(hp.get("reduce_lr_mode", "auto"))

    from app.callbacks.early_stopping import ReduceLROnPlateauNotificationCallback

    return ReduceLROnPlateauNotificationCallback(
        job_id=job_id,
        sio_instance=sio_instance,
        loop=loop,
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        min_delta=min_delta,
        mode=mode,
    )
