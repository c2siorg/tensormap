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

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        """Set ``model.stop_training`` if the job has been cancelled."""
        with self.session_factory() as session:
            job = session.get(TrainingJob, self.job_id)
            if job is not None and job.status == TrainingStatus.CANCELLED:
                logger.info("Cancellation detected for job %s at epoch %d", self.job_id, epoch)
                self.cancelled = True
                self.model.stop_training = True
