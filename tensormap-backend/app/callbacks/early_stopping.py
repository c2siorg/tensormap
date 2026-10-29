"""Keras callbacks for adaptive training: early stopping and learning rate reduction.

Provides subclassed Keras callbacks that emit structured events to the job's
Socket.IO room when early stopping triggers or when the learning rate is reduced.
"""

import tensorflow as tf

from app.callbacks.metrics_callback import schedule_room_emit
from app.shared.logging_config import get_logger
from app.shared.training_events import training_event

logger = get_logger(__name__)


class EarlyStoppingNotificationCallback(tf.keras.callbacks.EarlyStopping):
    """EarlyStopping callback that emits an informative event to the job room when triggered.

    Args:
        job_id: The training job being monitored.
        sio: The Socket.IO server instance.
        loop: The main event loop for thread-safe event scheduling.
        *args, **kwargs: Passed directly to ``tf.keras.callbacks.EarlyStopping``.
    """

    def __init__(self, job_id: str, sio, loop, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.job_id = job_id
        self.sio = sio
        self.loop = loop

    def on_train_end(self, logs: dict | None = None) -> None:
        super().on_train_end(logs)
        if self.stopped_epoch > 0:
            best_epoch = max(1, self.stopped_epoch + 1 - self.patience)
            logger.info(
                "Early stopping triggered at epoch %d for job %s (best epoch: %d, monitor: %s)",
                self.stopped_epoch + 1,
                self.job_id,
                best_epoch,
                self.monitor,
            )
            if self.sio and self.loop:
                schedule_room_emit(
                    self.sio,
                    self.loop,
                    self.job_id,
                    training_event(
                        self.job_id,
                        "early_stopping",
                        stopped_epoch=self.stopped_epoch + 1,
                        best_epoch=best_epoch,
                        monitor=self.monitor,
                    ),
                )


class ReduceLROnPlateauNotificationCallback(tf.keras.callbacks.ReduceLROnPlateau):
    """ReduceLROnPlateau callback that emits an event to the job room when the learning rate decreases.

    Args:
        job_id: The training job being monitored.
        sio: The Socket.IO server instance.
        loop: The main event loop for thread-safe event scheduling.
        *args, **kwargs: Passed directly to ``tf.keras.callbacks.ReduceLROnPlateau``.
    """

    def __init__(self, job_id: str, sio, loop, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.job_id = job_id
        self.sio = sio
        self.loop = loop

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        old_lr = self._get_lr()
        super().on_epoch_end(epoch, logs)
        new_lr = self._get_lr()

        if old_lr is not None and new_lr is not None and new_lr < old_lr:
            logger.info(
                "Learning rate reduced from %e to %e at epoch %d for job %s",
                old_lr,
                new_lr,
                epoch + 1,
                self.job_id,
            )
            if self.sio and self.loop:
                schedule_room_emit(
                    self.sio,
                    self.loop,
                    self.job_id,
                    training_event(
                        self.job_id,
                        "learning_rate_reduced",
                        epoch=epoch + 1,
                        old_lr=float(old_lr),
                        new_lr=float(new_lr),
                    ),
                )

    def _get_lr(self) -> float | None:
        """Extract the current learning rate from the model optimizer."""
        if self.model is None or not hasattr(self.model, "optimizer"):
            return None
        opt = self.model.optimizer
        if hasattr(opt, "learning_rate"):
            lr_attr = opt.learning_rate
            if hasattr(lr_attr, "numpy"):
                return float(lr_attr.numpy())
            try:
                return float(lr_attr)
            except (TypeError, ValueError):
                return None
        return None
