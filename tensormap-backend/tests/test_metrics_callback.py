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
from app.services.training_service import _train_in_thread, get_job_metrics_grouped


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


def test_cancelled_training_emits_cancelled_terminal_envelope(training_session_factory, monkeypatch, tmp_path):
    from app.services import model_export

    monkeypatch.setattr(model_export, "EXPORTS_BASE", tmp_path)
    job_id = _seed_job(training_session_factory, status=TrainingStatus.CANCELLED)
    sio = MagicMock()
    loop = MagicMock()
    loop.is_running.return_value = True
    cb = MetricsCallback(job_id, training_session_factory, sio, loop=loop)
    cb.set_model(MagicMock())

    with patch("app.callbacks.metrics_callback.asyncio.run_coroutine_threadsafe"):
        cb.on_train_end()

    event = sio.emit.call_args.args[1]
    assert event["type"] == "status"
    assert event["status"] == "cancelled"
    assert event["job_id"] == job_id
    assert event["version"] == 1


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
    args, kwargs = sio.emit.call_args

    assert kwargs["room"] == job_id
    assert kwargs["namespace"] == "/dl-result"

    payload = args[1]
    assert payload["version"] == 1
    assert payload["type"] == "metrics"
    assert payload["job_id"] == job_id
    assert payload["phase"] == "train"
    assert "timestamp" in payload
    assert payload["batch"] == 0
    assert payload["steps"] == 0

    assert rcts.called


def test_validation_metrics_share_epoch_event_and_preserve_training_counters(training_session_factory):
    job_id = _seed_job(training_session_factory)
    sio = MagicMock()
    loop = MagicMock()
    loop.is_running.return_value = True
    cb = MetricsCallback(job_id, training_session_factory, sio, loop=loop)

    with patch("app.callbacks.metrics_callback.asyncio.run_coroutine_threadsafe"):
        cb.on_epoch_begin(0)
        cb.on_train_batch_end(0)
        cb.on_train_batch_end(1)
        cb.on_test_batch_end(0)
        cb.on_test_batch_end(1)
        cb.on_epoch_end(0, {"loss": 0.25, "accuracy": 0.9, "val_loss": 0.3, "val_accuracy": 0.85})

    assert sio.emit.call_count == 1
    payload = sio.emit.call_args.args[1]
    assert payload["phase"] == "train"
    assert payload["loss"] == 0.25
    assert payload["accuracy"] == 0.9
    assert payload["val_loss"] == 0.3
    assert payload["val_accuracy"] == 0.85
    assert payload["batch"] == 2
    assert payload["steps"] == 2
    assert payload["validation_batch"] == 2
    assert payload["validation_steps"] == 2

    with training_session_factory() as session:
        persisted = get_job_metrics_grouped(job_id, session)[0]
    assert {key: payload[key] for key in persisted} == persisted


def test_batch_counters_reset_each_epoch_and_validation_does_not_change_training(training_session_factory):
    sio = MagicMock()
    loop = MagicMock()
    loop.is_running.return_value = True
    cb = MetricsCallback(_seed_job(training_session_factory), training_session_factory, sio, loop=loop)
    cb.on_epoch_begin(0)
    cb.on_train_batch_end(0)
    cb.on_train_batch_end(1)
    cb.on_test_batch_end(0)
    cb.on_test_batch_end(1)
    cb.on_epoch_begin(1)
    assert (cb.current_batch, cb.current_steps) == (0, 0)
    assert (cb.validation_batch, cb.validation_steps) == (0, 0)
    cb.on_train_batch_end(0)
    cb.on_test_batch_end(0)
    with patch("app.callbacks.metrics_callback.asyncio.run_coroutine_threadsafe"):
        cb.on_epoch_end(1, {"loss": 0.2})
    event = sio.emit.call_args.args[1]
    assert event["epoch"] == 2
    assert event["batch"] == event["steps"] == 1
    assert event["validation_batch"] == event["validation_steps"] == 1


def test_terminal_status_event_has_structured_success_envelope(training_session_factory, monkeypatch, tmp_path):
    from app.services import model_export

    monkeypatch.setattr(model_export, "EXPORTS_BASE", tmp_path)
    job_id = _seed_job(training_session_factory)
    sio = MagicMock()
    loop = MagicMock()
    loop.is_running.return_value = True
    cb = MetricsCallback(job_id, training_session_factory, sio, loop=loop)
    cb.set_model(MagicMock())

    with patch("app.callbacks.metrics_callback.asyncio.run_coroutine_threadsafe"):
        cb.on_train_begin()
        cb.on_train_end()

    event = sio.emit.call_args.args[1]
    assert event["type"] == "status"
    assert event["version"] == 1
    assert event["job_id"] == job_id
    assert event["status"] == "completed"
    assert event["timestamp"]
    assert event["phase"] is None
    assert event["epoch"] is None
    assert event["batch"] == 0
    assert event["steps"] == 0
    assert sio.emit.call_args.kwargs["room"] == job_id


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


def test_cancellation_skips_model_artifact_save(training_session_factory, tmp_path):
    """When a job is cancelled, model.save() must NOT be called in on_train_end."""

    job_id = _seed_job(training_session_factory, status=TrainingStatus.CANCELLED)
    sio = MagicMock()
    cb = MetricsCallback(job_id, training_session_factory, sio, None)
    mock_model = MagicMock()
    cb.set_model(mock_model)

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        cb.on_train_end()

    mock_model.save.assert_not_called()
    assert not (tmp_path / "exports" / job_id / "model.keras").exists()


def test_cancellation_stops_at_batch_boundary(training_session_factory):
    """CancellationCheckCallback detects cancellation on batch boundaries."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.CANCELLED)
    cb = CancellationCheckCallback(job_id, training_session_factory)
    model = MagicMock()
    model.stop_training = False
    cb.set_model(model)

    cb.on_train_batch_begin(0)

    assert cb.model.stop_training is True
    assert cb.cancelled is True
