"""Keras callback that cooperatively cancels training when requested.

The DELETE endpoint only sets the job's status to CANCELLED in the DB; this
callback is what actually stops the Keras loop. It checks the status at each
epoch boundary and sets ``model.stop_training`` so ``fit`` exits cleanly.
"""

import tensorflow as tf

from app.models.training_job import TrainingJob, TrainingStatus
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class CancellationCheckCallback(tf.keras.callbacks.Callback):
    """Stop training when the job's DB status becomes CANCELLED.

    Args:
        job_id: The training job to watch.
        session_factory: Zero-arg callable returning a fresh DB ``Session``.
        check_interval_batches: Frequency (in batches) to poll the DB during an epoch.
    """

    def __init__(self, job_id, session_factory, check_interval_batches: int = 5) -> None:
        super().__init__()
        self.job_id = job_id
        self.session_factory = session_factory
        self.check_interval_batches = max(1, check_interval_batches)
        # Side-effect flag so callers and evaluation phases can confirm a cancellation fired.
        self.cancelled = False
        self._batch_counter = 0

    def check_cancelled(self) -> bool:
        """Poll the DB for CANCELLED status, updating self.cancelled and stop_training."""
        if self.cancelled:
            if self.model is not None:
                self.model.stop_training = True
            return True

        with self.session_factory() as session:
            job = session.get(TrainingJob, self.job_id)
            if job is not None and job.status == TrainingStatus.CANCELLED:
                self.cancelled = True
                if self.model is not None:
                    self.model.stop_training = True
                return True
        return False

    def on_train_batch_begin(self, batch: int, logs: dict = None) -> None:
        """Check for cancellation during epoch execution every check_interval_batches."""
        self._batch_counter += 1
        if self._batch_counter % self.check_interval_batches == 0 and self.check_cancelled():
            logger.info(
                "Cancellation detected for job %s during batch %d (counter=%d)",
                self.job_id,
                batch,
                self._batch_counter,
            )

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        """Set ``model.stop_training`` if the job has been cancelled."""
        if self.check_cancelled():
            logger.info("Cancellation detected for job %s at epoch %d", self.job_id, epoch)

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """Ensure stop_training is preserved at the end of each epoch if cancelled."""
        if self.cancelled and self.model is not None:
            self.model.stop_training = True

    def on_test_begin(self, logs: dict = None) -> None:
        """Halt immediately if evaluation is triggered on a cancelled model."""
        if self.check_cancelled():
            logger.info("Cancellation detected for job %s during test evaluation", self.job_id)
            if self.model is not None:
                self.model.stop_training = True
