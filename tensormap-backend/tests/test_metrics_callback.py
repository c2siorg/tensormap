"""Tests for MetricsCallback, CancellationCheckCallback, and the failure path.

These exercise the callbacks directly (no real Keras run) against an isolated
in-memory DB, which is reliable and fast. A genuine end-to-end Keras run lives
in test_training_e2e.py behind the ``slow`` marker.
"""

from unittest.mock import MagicMock, patch

from sqlmodel import select

from app.callbacks.cancellation_callback import CancellationCheckCallback
from app.callbacks.metrics_callback import MetricsCallback
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric
from app.services.training_service import _train_in_thread


def _seed_job(factory, status=TrainingStatus.PENDING) -> str:
    with factory() as session:
        job = TrainingJob(model_id=1, status=status)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job.id


def test_metrics_persist_after_training(training_session_factory):
    job_id = _seed_job(training_session_factory)
    cb = MetricsCallback(job_id, training_session_factory, MagicMock(), loop=None)

    cb.on_train_begin()
    cb.on_epoch_end(0, {"loss": 0.5, "accuracy": 0.8, "val_loss": 0.6, "val_accuracy": 0.7})
    cb.on_epoch_end(1, {"loss": 0.3, "accuracy": 0.9, "val_loss": 0.4, "val_accuracy": 0.85})
    cb.on_train_end()

    with training_session_factory() as session:
        rows = session.exec(select(TrainingMetric).where(TrainingMetric.job_id == job_id)).all()
        # 4 metrics x 2 epochs.
        assert len(rows) == 8
        names = {r.metric_name for r in rows}
        assert names == {"loss", "accuracy", "val_loss", "val_accuracy"}


def test_metrics_omit_missing_validation_metrics(training_session_factory):
    job_id = _seed_job(training_session_factory)
    cb = MetricsCallback(job_id, training_session_factory, MagicMock(), loop=None)
    # No val_* metrics in logs -> only loss + accuracy persisted.
    cb.on_epoch_end(0, {"loss": 0.5, "accuracy": 0.8})

    with training_session_factory() as session:
        rows = session.exec(select(TrainingMetric).where(TrainingMetric.job_id == job_id)).all()
        assert {r.metric_name for r in rows} == {"loss", "accuracy"}


def test_job_status_running_during_training(training_session_factory):
    job_id = _seed_job(training_session_factory)
    cb = MetricsCallback(job_id, training_session_factory, MagicMock(), loop=None)
    cb.on_train_begin()

    with training_session_factory() as session:
        job = session.get(TrainingJob, job_id)
        assert job.status == TrainingStatus.RUNNING
        assert job.started_at is not None


def test_job_status_completed_after_training(training_session_factory):
    job_id = _seed_job(training_session_factory)
    cb = MetricsCallback(job_id, training_session_factory, MagicMock(), loop=None)
    cb.on_train_begin()
    cb.on_train_end()

    with training_session_factory() as session:
        job = session.get(TrainingJob, job_id)
        assert job.status == TrainingStatus.COMPLETED
        assert job.completed_at is not None


def test_on_train_end_does_not_override_cancelled(training_session_factory):
    job_id = _seed_job(training_session_factory, status=TrainingStatus.CANCELLED)
    cb = MetricsCallback(job_id, training_session_factory, MagicMock(), loop=None)
    cb.on_train_end()

    with training_session_factory() as session:
        job = session.get(TrainingJob, job_id)
        # A cancelled job must not be flipped to completed.
        assert job.status == TrainingStatus.CANCELLED


def test_metrics_emitted_to_job_room(training_session_factory):
    job_id = _seed_job(training_session_factory)
    sio = MagicMock()
    loop = MagicMock()
    loop.is_running.return_value = True
    cb = MetricsCallback(job_id, training_session_factory, sio, loop=loop)

    with patch("app.callbacks.metrics_callback.asyncio.run_coroutine_threadsafe") as rcts:
        cb.on_epoch_end(0, {"loss": 0.5, "accuracy": 0.8})

    # The emit is scheduled on the loop; assert it targets this job's room only.
    assert sio.emit.called
    _, kwargs = sio.emit.call_args
    assert kwargs["room"] == job_id
    assert kwargs["namespace"] == "/dl-result"
    assert rcts.called


def test_cancellation_stops_training(training_session_factory):
    job_id = _seed_job(training_session_factory, status=TrainingStatus.CANCELLED)
    cb = CancellationCheckCallback(job_id, training_session_factory)
    cb.set_model(MagicMock())

    cb.on_epoch_begin(0)

    assert cb.model.stop_training is True
    assert cb.cancelled is True


def test_cancellation_no_stop_when_running(training_session_factory):
    job_id = _seed_job(training_session_factory, status=TrainingStatus.RUNNING)
    cb = CancellationCheckCallback(job_id, training_session_factory)
    model = MagicMock()
    model.stop_training = False
    cb.set_model(model)

    cb.on_epoch_begin(0)

    assert cb.model.stop_training is False
    assert cb.cancelled is False


def test_job_stores_error_on_exception(training_session_factory):
    job_id = _seed_job(training_session_factory)

    with patch("app.services.model_run.model_run", side_effect=RuntimeError("boom")):
        _train_in_thread("m1", job_id, loop=None)

    with training_session_factory() as session:
        job = session.get(TrainingJob, job_id)
        assert job.status == TrainingStatus.FAILED
        assert "boom" in job.error_message
