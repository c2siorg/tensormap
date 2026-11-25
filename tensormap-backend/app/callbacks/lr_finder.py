"""Learning Rate Finder callback based on Leslie Smith's LR range test."""

import contextlib
import math
from typing import Any

import numpy as np
import tensorflow as tf

from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _extract_lr(optimizer: Any) -> float:
    """Extract numeric learning rate from Keras 2 / Keras 3 optimizer."""
    lr = getattr(optimizer, "learning_rate", getattr(optimizer, "lr", None))
    if hasattr(lr, "numpy"):
        return float(lr.numpy())
    if hasattr(lr, "value"):
        val = lr.value()
        if hasattr(val, "numpy"):
            return float(val.numpy())
    return float(lr)


def _set_lr(optimizer: Any, value: float) -> None:
    """Set numeric learning rate on Keras 2 / Keras 3 optimizer."""
    lr = getattr(optimizer, "learning_rate", getattr(optimizer, "lr", None))
    if hasattr(lr, "assign"):
        lr.assign(value)
    elif hasattr(optimizer, "learning_rate"):
        try:
            optimizer.learning_rate = value
        except Exception:
            with contextlib.suppress(Exception):
                tf.keras.backend.set_value(optimizer.learning_rate, value)


class LRFinderCallback(tf.keras.callbacks.Callback):
    """Exponentially sweeps learning rate across batches and records smoothed loss.

    Restores original weights and learning rate when completed.
    """

    def __init__(
        self,
        start_lr: float = 1e-6,
        end_lr: float = 1.0,
        num_batches: int = 50,
        beta: float = 0.98,
        stop_factor: float = 4.0,
    ) -> None:
        super().__init__()
        self.start_lr = float(start_lr)
        self.end_lr = float(end_lr)
        self.num_batches = max(10, int(num_batches))
        self.beta = float(beta)
        self.stop_factor = float(stop_factor)

        self.lr_multiplier = (self.end_lr / self.start_lr) ** (1.0 / self.num_batches)

        self.history: list[dict[str, float]] = []
        self.best_loss = float("inf")
        self.avg_loss = 0.0
        self.batch_count = 0
        self.diverged = False
        self._initial_weights: list[np.ndarray] | None = None
        self._initial_lr: float | None = None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.history = []
        self.best_loss = float("inf")
        self.avg_loss = 0.0
        self.batch_count = 0
        self.diverged = False

        self._initial_weights = [w.copy() for w in self.model.get_weights()]

        optimizer = self.model.optimizer
        if hasattr(optimizer, "learning_rate") or hasattr(optimizer, "lr"):
            self._initial_lr = _extract_lr(optimizer)
            _set_lr(optimizer, self.start_lr)

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        loss = logs.get("loss")
        if loss is None or math.isnan(loss):
            self.model.stop_training = True
            self.diverged = True
            return

        loss = float(loss)
        self.batch_count += 1

        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smooth_loss = self.avg_loss / (1 - self.beta**self.batch_count)

        optimizer = self.model.optimizer
        current_lr = _extract_lr(optimizer)

        if self.batch_count > 1 and smooth_loss > self.stop_factor * self.best_loss:
            self.model.stop_training = True
            self.diverged = True
            logger.info("LR finder stopping early at batch %d: loss diverged", self.batch_count)
            return

        if smooth_loss < self.best_loss:
            self.best_loss = smooth_loss

        self.history.append({"learning_rate": current_lr, "loss": smooth_loss})

        if self.batch_count >= self.num_batches:
            self.model.stop_training = True
            return

        next_lr = current_lr * self.lr_multiplier
        _set_lr(optimizer, next_lr)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self._initial_weights is not None:
            self.model.set_weights(self._initial_weights)
        if self._initial_lr is not None and (
            hasattr(self.model.optimizer, "learning_rate") or hasattr(self.model.optimizer, "lr")
        ):
            _set_lr(self.model.optimizer, self._initial_lr)

    def get_results(self) -> dict[str, Any]:
        """Compute optimal suggested learning rate and summary metrics."""
        if not self.history:
            return {
                "history": [],
                "suggested_lr": None,
                "min_loss": None,
                "diverged": self.diverged,
                "num_batches_completed": 0,
            }

        lrs = [pt["learning_rate"] for pt in self.history]
        losses = [pt["loss"] for pt in self.history]
        min_loss = float(min(losses))

        suggested_lr: float | None = None
        if len(self.history) >= 4:
            log_lrs = np.log10(lrs)
            gradients = np.gradient(losses, log_lrs)
            min_loss_idx = int(np.argmin(losses))
            valid_slice = slice(0, max(min_loss_idx, 1))
            steepest_idx = int(np.argmin(gradients[valid_slice]))
            suggested_lr = float(lrs[steepest_idx])
        elif len(self.history) > 0:
            suggested_lr = float(lrs[0])

        return {
            "history": self.history,
            "suggested_lr": suggested_lr,
            "min_loss": min_loss,
            "diverged": self.diverged,
            "num_batches_completed": self.batch_count,
        }
