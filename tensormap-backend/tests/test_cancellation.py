"""Unit and integration tests for training job cancellation lifecycle,

eval skipping, and artifact protection (Fixes Issue #469).
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.callbacks.cancellation_callback import CancellationCheckCallback
from app.callbacks.metrics_callback import MetricsCallback
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.model_run import _should_skip_evaluation


def _seed_job(session_factory, status=TrainingStatus.RUNNING, hyperparams=None):
    with session_factory() as session:
        job = TrainingJob(
            id=str(uuid4()),
            model_id=1,
            status=status,
            hyperparams=hyperparams or {"epochs": 5, "batch_size": 32},
            started_at=datetime.now(UTC),
        )
        session.add(job)
        session.commit()
        return job.id


# ---------------------------------------------------------------------------
# CancellationCheckCallback unit tests
# ---------------------------------------------------------------------------


def test_cancellation_detected_during_batch_execution(training_session_factory):
    """Callback detects cancellation mid-epoch on batch boundaries."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.RUNNING)
    cb = CancellationCheckCallback(job_id, training_session_factory, check_interval_batches=2)
    model = MagicMock()
    model.stop_training = False
    cb.set_model(model)

    # Batch 1: Not a check interval, shouldn't trigger
    cb.on_train_batch_begin(0)
    assert cb.cancelled is False
    assert model.stop_training is False

    # Flip status in DB to CANCELLED
    with training_session_factory() as session:
        job = session.get(TrainingJob, job_id)
        job.status = TrainingStatus.CANCELLED
        session.add(job)
        session.commit()

    # Batch 2: Hits check_interval (counter=2), detects cancellation
    cb.on_train_batch_begin(1)
    assert cb.cancelled is True
    assert model.stop_training is True


def test_cancellation_halt_on_epoch_begin(training_session_factory):
    """Callback halts training at the start of an epoch."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.CANCELLED)
    cb = CancellationCheckCallback(job_id, training_session_factory)
    model = MagicMock()
    model.stop_training = False
    cb.set_model(model)

    cb.on_epoch_begin(0)
    assert cb.cancelled is True
    assert model.stop_training is True


def test_cancellation_halt_on_test_begin(training_session_factory):
    """Callback halts evaluation immediately if cancellation flag is active."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.CANCELLED)
    cb = CancellationCheckCallback(job_id, training_session_factory)
    model = MagicMock()
    model.stop_training = False
    cb.set_model(model)

    cb.on_test_begin()
    assert cb.cancelled is True
    assert model.stop_training is True


def test_cancellation_no_action_for_running_job(training_session_factory):
    """Callback leaves model untouched when job remains RUNNING."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.RUNNING)
    cb = CancellationCheckCallback(job_id, training_session_factory, check_interval_batches=2)
    model = MagicMock()
    model.stop_training = False
    cb.set_model(model)

    cb.on_train_batch_begin(0)
    cb.on_train_batch_begin(1)
    cb.on_epoch_begin(1)
    cb.on_epoch_end(1)

    assert cb.cancelled is False
    assert model.stop_training is False


# ---------------------------------------------------------------------------
# Model evaluation skipping logic
# ---------------------------------------------------------------------------


def test_should_skip_evaluation_when_cancelled_flag_set(training_session_factory):
    """_should_skip_evaluation returns True if cancellation callback was cancelled."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.RUNNING)
    cb = CancellationCheckCallback(job_id, training_session_factory)
    cb.cancelled = True
    model = MagicMock()
    model.stop_training = False

    assert _should_skip_evaluation([cb], model) is True


def test_should_skip_evaluation_when_model_stop_training_set():
    """_should_skip_evaluation returns True if model.stop_training is True."""
    model = MagicMock()
    model.stop_training = True

    assert _should_skip_evaluation([], model) is True


def test_should_not_skip_evaluation_for_normal_run(training_session_factory):
    """_should_skip_evaluation returns False when no cancellation occurred."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.RUNNING)
    cb = CancellationCheckCallback(job_id, training_session_factory)
    model = MagicMock()
    model.stop_training = False

    assert _should_skip_evaluation([cb], model) is False


# ---------------------------------------------------------------------------
# MetricsCallback artifact protection & status emit tests
# ---------------------------------------------------------------------------


def test_cancelled_job_does_not_save_model_artifact(training_session_factory, tmp_path):
    """on_train_end skips model.save() and cleans up partial files when job was CANCELLED."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.CANCELLED)

    sio = MagicMock()
    cb = MetricsCallback(job_id, training_session_factory, sio, loop=None)
    model = MagicMock()
    cb.set_model(model)

    # Pre-create a partial model.keras in export_dir to test cleanup
    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True, exist_ok=True)
    partial_file = export_dir / "model.keras"
    partial_file.write_text("corrupted_data")

    with (
        patch("app.callbacks.metrics_callback.schedule_room_emit") as mock_emit,
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
    ):
        cb.on_train_end()

    # Verify model.save was NOT called
    model.save.assert_not_called()

    # Verify partial corrupted model file was cleaned up
    assert not partial_file.exists()

    # Verify terminal status event emitted with "cancelled"
    mock_emit.assert_called_once()
    event = mock_emit.call_args[0][3]
    assert event["type"] == "status"
    assert event["status"] == "cancelled"


def test_failed_job_does_not_save_model_artifact(training_session_factory, tmp_path):
    """on_train_end skips model.save() when job has FAILED."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.FAILED)

    sio = MagicMock()
    cb = MetricsCallback(job_id, training_session_factory, sio, loop=None)
    model = MagicMock()
    cb.set_model(model)

    with (
        patch("app.callbacks.metrics_callback.schedule_room_emit") as mock_emit,
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
    ):
        cb.on_train_end()

    model.save.assert_not_called()
    event = mock_emit.call_args[0][3]
    assert event["type"] == "status"
    assert event["status"] == "failed"


def test_successful_job_saves_model_artifact(training_session_factory, tmp_path):
    """on_train_end saves model.keras and marks COMPLETED for normal RUNNING jobs."""
    job_id = _seed_job(training_session_factory, status=TrainingStatus.RUNNING)

    sio = MagicMock()
    cb = MetricsCallback(job_id, training_session_factory, sio, loop=None)
    model = MagicMock()
    cb.set_model(model)

    with (
        patch("app.callbacks.metrics_callback.schedule_room_emit") as mock_emit,
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
    ):
        cb.on_train_end()

    # Model save was called
    model.save.assert_called_once()

    # Job status updated to COMPLETED in DB
    with training_session_factory() as session:
        job = session.get(TrainingJob, job_id)
        assert job.status == TrainingStatus.COMPLETED
        assert job.completed_at is not None

    # Status event emitted with "completed"
    mock_emit.assert_called_once()
    event = mock_emit.call_args[0][3]
    assert event["type"] == "status"
    assert event["status"] == "completed"
