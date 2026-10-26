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
    """

    def __init__(self, job_id, session_factory) -> None:
        super().__init__()
        self.job_id = job_id
        self.session_factory = session_factory
        # Side-effect flag so tests (and callers) can confirm a cancellation fired.
        self.cancelled = False

    def _check_cancellation(self) -> bool:
        """Check if job status is CANCELLED and signal model.stop_training."""
        if self.cancelled:
            return True
        with self.session_factory() as session:
            job = session.get(TrainingJob, self.job_id)
            if job is not None and job.status == TrainingStatus.CANCELLED:
                logger.info("Cancellation detected for job %s", self.job_id)
                self.cancelled = True
                if hasattr(self, "model") and self.model is not None:
                    self.model.stop_training = True
                return True
        return False

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        """Check cancellation at the start of each epoch."""
        self._check_cancellation()

    def on_train_batch_begin(self, batch: int, logs: dict = None) -> None:
        """Check cancellation at batch boundaries for prompt termination."""
        self._check_cancellation()
