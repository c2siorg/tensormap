"""Dynamic learning rate scheduler callbacks and optimizer inspection utilities for Keras model training."""

import math
from typing import Any

import tensorflow as tf

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def get_current_learning_rate(model: tf.keras.Model) -> float:
    """Safely extracts current scalar learning rate across Keras 2 and Keras 3 optimizers."""
    try:
        opt = getattr(model, "optimizer", None)
        if opt is None:
            return 0.0

        lr_attr = getattr(opt, "learning_rate", None)
        if lr_attr is None:
            lr_attr = getattr(opt, "lr", None)
        if lr_attr is None:
            return 0.0

        # Handle Keras Variable, TF tensor, callable schedule, or scalar float
        if hasattr(lr_attr, "numpy") and callable(lr_attr.numpy):
            return float(lr_attr.numpy())
        if hasattr(lr_attr, "value"):
            val = lr_attr.value() if callable(lr_attr.value) else lr_attr.value
            return float(val.numpy() if hasattr(val, "numpy") else val)
        if callable(lr_attr):
            iterations = getattr(opt, "iterations", 0)
            iter_val = iterations.numpy() if hasattr(iterations, "numpy") else float(iterations)
            computed = lr_attr(iter_val)
            return float(computed.numpy()) if hasattr(computed, "numpy") else float(computed)
        return float(lr_attr)
    except Exception as e:
        logger.debug("Failed to extract current learning rate from optimizer: %s", e)
        return 0.0


class MetricLearningRateLogger(tf.keras.callbacks.Callback):
    """Logs learning rate changes on epoch end and appends current learning rate to epoch logs."""

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            logs = {}
        curr_lr = get_current_learning_rate(self.model)
        logs["learning_rate"] = curr_lr
        logs["lr"] = curr_lr


def build_lr_scheduler_callback(
    scheduler_type: str | None,
    config: dict[str, Any] | None = None,
    initial_lr: float = 0.001,
) -> list[tf.keras.callbacks.Callback]:
    """Factory creating configured dynamic learning rate callback and metric logger.

    Supported schedulers:
    - 'reduce_on_plateau': ReduceLROnPlateau monitoring val_loss or loss.
    - 'exponential': Exponential learning rate decay.
    - 'cosine': Cosine decay learning rate schedule.
    """
    if not scheduler_type:
        return [MetricLearningRateLogger()]

    callbacks: list[tf.keras.callbacks.Callback] = [MetricLearningRateLogger()]
    cfg = config or {}
    stype = str(scheduler_type).lower().strip()

    if stype in ("reduce_on_plateau", "plateau"):
        has_val = bool(cfg.get("has_validation", True))
        monitor = str(cfg.get("monitor", "val_loss" if has_val else "loss"))
        if not has_val and monitor == "val_loss":
            monitor = "loss"
            logger.info("No validation split detected; dynamically switching ReduceLROnPlateau monitor to 'loss'.")
        factor = float(cfg.get("factor", 0.5))
        patience = int(cfg.get("patience", 3))
        min_lr = float(cfg.get("min_lr", 1e-6))
        min_delta = float(cfg.get("min_delta", 1e-4))
        mode = str(cfg.get("mode", "auto"))

        cb = tf.keras.callbacks.ReduceLROnPlateau(
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

    elif stype in ("exponential", "exponential_decay"):
        decay_rate = float(cfg.get("decay_rate", 0.95))

        def _exp_sched(epoch: int, lr: float) -> float:
            return float(initial_lr * math.pow(decay_rate, epoch))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_exp_sched, verbose=0))
        logger.info("Configured Exponential decay scheduler: rate=%.2f", decay_rate)

    elif stype in ("cosine", "cosine_annealing"):
        total_epochs = max(1, int(cfg.get("total_epochs", 20)))
        min_lr = float(cfg.get("min_lr", 1e-6))

        def _cosine_sched(epoch: int, lr: float) -> float:
            fraction = min(epoch / total_epochs, 1.0)
            return float(min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * fraction)))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_cosine_sched, verbose=0))
        logger.info("Configured Cosine annealing scheduler: total_epochs=%d", total_epochs)

    elif stype in ("step", "step_decay"):
        step_size = max(1, int(cfg.get("step_size", 5)))
        gamma = float(cfg.get("gamma", 0.5))

        def _step_sched(epoch: int, lr: float) -> float:
            return float(initial_lr * math.pow(gamma, math.floor(epoch / step_size)))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(_step_sched, verbose=0))
        logger.info("Configured Step decay scheduler: step_size=%d gamma=%.2f", step_size, gamma)

    return callbacks


def get_supported_lr_schedulers() -> list[str]:
    """Returns a list of supported learning rate scheduler names."""
    return ["reduce_on_plateau", "exponential", "cosine", "step"]
