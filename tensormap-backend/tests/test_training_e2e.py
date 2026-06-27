"""End-to-end training tests with a real (tiny) Keras model.

Marked ``slow`` so the default test run can skip them with ``-m "not slow"``.
These wire the real ``MetricsCallback``/``CancellationCheckCallback`` into an
actual ``model.fit`` to prove the full lifecycle: real epochs -> persisted
metrics -> status transitions, and a real cooperative cancellation.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from app.callbacks.cancellation_callback import CancellationCheckCallback
from app.callbacks.metrics_callback import MetricsCallback
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.training_service import get_job_metrics_grouped, update_job_status

pytestmark = pytest.mark.slow


def _tiny_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((2,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _binary_dataset(n: int = 100):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 2)).astype("float32")
    y = (x[:, 0] + x[:, 1] > 0).astype("float32")
    return x, y


def _seed_job(factory, status=TrainingStatus.PENDING) -> str:
    with factory() as session:
        job = TrainingJob(model_id=1, status=status)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job.id


def test_e2e_training_completes_and_persists_metrics(training_session_factory):
    job_id = _seed_job(training_session_factory)
    callback = MetricsCallback(job_id, training_session_factory, MagicMock(), loop=None)

    x, y = _binary_dataset()
    _tiny_model().fit(x, y, epochs=5, batch_size=16, verbose=0, callbacks=[callback])

    with training_session_factory() as session:
        job = session.get(TrainingJob, job_id)
        assert job.status == TrainingStatus.COMPLETED
        assert job.started_at is not None
        grouped = get_job_metrics_grouped(job_id, session)

    assert len(grouped) == 5
    assert "loss" in grouped[0]
    assert "accuracy" in grouped[0]


def test_e2e_cancellation_stops_training(training_session_factory):
    job_id = _seed_job(training_session_factory, status=TrainingStatus.RUNNING)
    metrics_cb = MetricsCallback(job_id, training_session_factory, MagicMock(), loop=None)
    cancel_cb = CancellationCheckCallback(job_id, training_session_factory)

    factory = training_session_factory

    class CancelAfterFirstEpoch(tf.keras.callbacks.Callback):
        """Simulate a user pressing Stop after the first epoch completes."""

        def on_epoch_end(self, epoch, logs=None):
            if epoch == 0:
                with factory() as session:
                    update_job_status(job_id, TrainingStatus.CANCELLED, session)

    x, y = _binary_dataset()
    model = _tiny_model()
    model.fit(x, y, epochs=10, batch_size=16, verbose=0, callbacks=[metrics_cb, cancel_cb, CancelAfterFirstEpoch()])

    assert cancel_cb.cancelled is True
    assert model.stop_training is True

    with training_session_factory() as session:
        job = session.get(TrainingJob, job_id)
        assert job.status == TrainingStatus.CANCELLED
        grouped = get_job_metrics_grouped(job_id, session)
    # Training stopped early rather than running all 10 epochs.
    assert len(grouped) < 10
