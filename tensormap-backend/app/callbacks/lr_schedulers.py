"""Adaptive learning rate schedulers and callbacks (Issue #558).

Provides dynamic learning rate scheduling via ReduceLROnPlateau, exponential decay,
cosine annealing, and step decay, tracking learning rate transitions in real-time,
emitting Socket.IO events, and logging learning rate into epoch logs for persistent metric tracking.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

import numpy as np
import tensorflow as tf

from app.callbacks.metrics_callback import schedule_room_emit

logger = logging.getLogger(__name__)

SUPPORTED_SCHEDULERS: dict[str, str] = {
    "reduce_on_plateau": "reduce_on_plateau",
    "plateau": "reduce_on_plateau",
    "exponential": "exponential",
    "exponential_decay": "exponential",
    "cosine": "cosine",
    "cosine_annealing": "cosine",
    "step": "step",
    "step_decay": "step",
}


def _extract_lr(optimizer: Any) -> float | None:
    """Safely extract float learning rate value from a Keras optimizer."""
    if optimizer is None:
        return None
    try:
        lr_attr = getattr(optimizer, "learning_rate", getattr(optimizer, "lr", None))
        if lr_attr is None:
            return None
        if hasattr(lr_attr, "numpy") and callable(lr_attr.numpy):
            return float(lr_attr.numpy())
        if hasattr(lr_attr, "value"):
            val = lr_attr.value() if callable(lr_attr.value) else lr_attr.value
            return float(val.numpy() if hasattr(val, "numpy") else val)
        if callable(lr_attr):
            iterations = getattr(optimizer, "iterations", 0)
            iter_val = iterations.numpy() if hasattr(iterations, "numpy") else float(iterations)
            computed = lr_attr(iter_val)
            return float(computed.numpy()) if hasattr(computed, "numpy") else float(computed)
        val = np.array(lr_attr)
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def get_current_learning_rate(model: tf.keras.Model | None) -> float | None:
    """Safely extracts current scalar learning rate across Keras 2 and Keras 3 optimizers.

    Returns None when the model or optimizer is not configured or on extraction failure.
    """
    if model is None:
        return None
    try:
        opt = getattr(model, "optimizer", None)
        return _extract_lr(opt)
    except Exception as e:
        logger.debug("Failed to extract current learning rate from optimizer: %s", e)
        return None


class MetricLearningRateLogger(tf.keras.callbacks.Callback):
    """Logs learning rate changes on epoch end and appends current learning rate to epoch logs."""

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            logs = {}
        curr_lr = get_current_learning_rate(self.model)
        if curr_lr is not None:
            logs["lr"] = curr_lr


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


def get_supported_lr_schedulers() -> list[str]:
    """Returns a list of canonical supported learning rate scheduler names."""
    return ["reduce_on_plateau", "exponential", "cosine", "step"]


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

    factor = float(hp.get("reduce_lr_factor", 0.2))
    patience = int(hp.get("reduce_lr_patience", 5))
    min_lr = float(hp.get("reduce_lr_min_lr", 1e-6))
    min_delta = float(hp.get("reduce_lr_min_delta", 1e-4))
    mode = str(hp.get("reduce_lr_mode", "auto"))

    return ReduceLROnPlateauCallback(
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


def build_lr_scheduler_callback(
    scheduler_type: str | None,
    config: dict[str, Any] | None = None,
    initial_lr: float = 0.001,
    job_id: str | None = None,
    sio_instance: Any = None,
    loop: Any = None,
) -> list[tf.keras.callbacks.Callback]:
    """Factory creating configured dynamic learning rate callback and metric logger.

    Supported schedulers:
    - 'reduce_on_plateau' (alias 'plateau'): ReduceLROnPlateau monitoring val_loss or loss.
    - 'exponential' (alias 'exponential_decay'): Exponential learning rate decay.
    - 'cosine' (alias 'cosine_annealing'): Cosine decay learning rate schedule.
    - 'step' (alias 'step_decay'): Step decay learning rate schedule.
    """
    if not scheduler_type:
        return [MetricLearningRateLogger()]

    stype_raw = str(scheduler_type).lower().strip()
    if stype_raw not in SUPPORTED_SCHEDULERS:
        supported = get_supported_lr_schedulers()
        raise ValueError(f"Unsupported learning rate scheduler '{scheduler_type}'. Supported schedulers: {supported}")

    stype = SUPPORTED_SCHEDULERS[stype_raw]
    cfg = config or {}
    callbacks: list[tf.keras.callbacks.Callback] = []

    if stype == "reduce_on_plateau":
        has_val = bool(cfg.get("has_validation", True))
        monitor = str(cfg.get("monitor", "val_loss" if has_val else "loss"))
        if not has_val and monitor == "val_loss":
            monitor = "loss"
            logger.info("No validation split detected; dynamically switching ReduceLROnPlateau monitor to 'loss'.")

        try:
            factor = float(cfg.get("factor", 0.5))
            if not (0.0 < factor < 1.0):
                raise ValueError("factor must be between 0 and 1")
            patience = int(cfg.get("patience", 3))
            if patience < 1:
                raise ValueError("patience must be at least 1")
            min_lr = float(cfg.get("min_lr", 1e-6))
            if min_lr < 0:
                raise ValueError("min_lr cannot be negative")
            min_delta = float(cfg.get("min_delta", 1e-4))
            if min_delta < 0:
                raise ValueError("min_delta cannot be negative")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid ReduceLROnPlateau configuration: {err}") from err

        mode = str(cfg.get("mode", "auto"))

        cb = ReduceLROnPlateauCallback(
            job_id=job_id,
            sio_instance=sio_instance,
            loop=loop,
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            min_delta=min_delta,
            mode=mode,
            verbose=0,
        )
        callbacks.append(cb)
        logger.info("Configured ReduceLROnPlateau: monitor=%s factor=%.2f patience=%d", monitor, factor, patience)

    elif stype == "exponential":
        try:
            decay_rate = float(cfg.get("decay_rate", 0.95))
            if not (0.0 < decay_rate < 1.0):
                raise ValueError("decay_rate must be between 0 and 1")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid exponential scheduler configuration: {err}") from err

        def _exp_sched(epoch: int, lr: float) -> float:
            return float(initial_lr * math.pow(decay_rate, epoch))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_exp_sched, verbose=0))
        callbacks.append(MetricLearningRateLogger())
        logger.info("Configured Exponential decay scheduler: rate=%.2f", decay_rate)

    elif stype == "cosine":
        try:
            total_epochs = int(cfg.get("total_epochs", 20))
            if total_epochs < 1:
                raise ValueError("total_epochs must be at least 1")
            min_lr = float(cfg.get("min_lr", 1e-6))
            if min_lr < 0:
                raise ValueError("min_lr cannot be negative")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid cosine scheduler configuration: {err}") from err

        def _cosine_sched(epoch: int, lr: float) -> float:
            fraction = min(epoch / total_epochs, 1.0)
            return float(min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * fraction)))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_cosine_sched, verbose=0))
        callbacks.append(MetricLearningRateLogger())
        logger.info("Configured Cosine annealing scheduler: total_epochs=%d", total_epochs)

    elif stype == "step":
        try:
            step_size = int(cfg.get("step_size", 5))
            if step_size < 1:
                raise ValueError("step_size must be at least 1")
            gamma = float(cfg.get("gamma", 0.5))
            if not (0.0 < gamma < 1.0):
                raise ValueError("gamma must be between 0 and 1")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid step scheduler configuration: {err}") from err

        def _step_sched(epoch: int, lr: float) -> float:
            return float(initial_lr * math.pow(gamma, math.floor(epoch / step_size)))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_step_sched, verbose=0))
        callbacks.append(MetricLearningRateLogger())
        logger.info("Configured Step decay scheduler: step_size=%d gamma=%.2f", step_size, gamma)

    return callbacks
