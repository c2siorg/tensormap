"""Unit and integration tests for BestModelCheckpointCallback and optimal weight restoration."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
import tensorflow as tf
from sqlmodel import Session

from app.callbacks.checkpoint_callback import BestModelCheckpointCallback
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.model_run import _build_training_callbacks


def _create_simple_model():
    """Create a minimal 1-layer Keras model for testing."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(2, use_bias=False),
        ]
    )
    model.compile(optimizer="sgd", loss="mse")
    return model


@pytest.fixture
def test_job(db_session: Session) -> str:
    """Create a running training job in the test database."""
    model = ModelBasic(
        model_name=f"test_model_{uuid4().hex[:8]}",
        model_type=1,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.RUNNING,
    )
    db_session.add(job)
    db_session.commit()
    return job_id


# -----------------------------------------------------------------------------
# Checkpoint Callback Initialization & Mode Resolution Tests
# -----------------------------------------------------------------------------


def test_checkpoint_mode_auto_deduction(db_session: Session):
    """Auto mode correctly identifies 'min' for loss and 'max' for accuracy."""
    cb_loss = BestModelCheckpointCallback("job-1", lambda: db_session, monitor="val_loss", mode="auto")
    assert cb_loss.mode == "min"
    assert cb_loss.best_value == float("inf")

    cb_acc = BestModelCheckpointCallback("job-2", lambda: db_session, monitor="val_accuracy", mode="auto")
    assert cb_acc.mode == "max"
    assert cb_acc.best_value == float("-inf")

    cb_mae = BestModelCheckpointCallback("job-3", lambda: db_session, monitor="mae", mode="auto")
    assert cb_mae.mode == "min"

    cb_explicit = BestModelCheckpointCallback("job-4", lambda: db_session, monitor="val_loss", mode="max")
    assert cb_explicit.mode == "max"


# -----------------------------------------------------------------------------
# Checkpoint Weight Tracking & Restoration Tests
# -----------------------------------------------------------------------------


def test_checkpoint_tracks_best_epoch_and_restores_optimal_weights(test_job: str, db_session: Session):
    """BestModelCheckpointCallback snapshots optimal epoch weights and restores them on train end."""
    model = _create_simple_model()

    cb = BestModelCheckpointCallback(
        job_id=test_job,
        session_factory=lambda: db_session,
        monitor="val_loss",
    )
    cb.set_model(model)

    # Epoch 1: val_loss 0.8
    w1 = [np.ones((4, 2), dtype=np.float32) * 1.0]
    model.set_weights(w1)
    cb.on_epoch_end(0, logs={"loss": 0.9, "val_loss": 0.8})
    assert cb.best_epoch == 1
    assert abs(cb.best_value - 0.8) < 1e-5
    np.testing.assert_allclose(cb.best_weights[0], w1[0])

    # Epoch 2: val_loss 0.35 (Improvement!)
    w2 = [np.ones((4, 2), dtype=np.float32) * 2.0]
    model.set_weights(w2)
    cb.on_epoch_end(1, logs={"loss": 0.5, "val_loss": 0.35})
    assert cb.best_epoch == 2
    assert abs(cb.best_value - 0.35) < 1e-5
    np.testing.assert_allclose(cb.best_weights[0], w2[0])

    # Epoch 3: val_loss 0.70 (Overfitting — degraded performance)
    w3 = [np.ones((4, 2), dtype=np.float32) * 3.0]
    model.set_weights(w3)
    cb.on_epoch_end(2, logs={"loss": 0.2, "val_loss": 0.70})
    # Best remains Epoch 2
    assert cb.best_epoch == 2
    assert abs(cb.best_value - 0.35) < 1e-5
    np.testing.assert_allclose(cb.best_weights[0], w2[0])

    # At the end of epoch 3, the model currently holds degraded weights w3
    np.testing.assert_allclose(model.get_weights()[0], w3[0])

    # On train end, callback restores optimal weights w2 from Epoch 2
    cb.on_train_end()
    assert cb.restored is True
    np.testing.assert_allclose(model.get_weights()[0], w2[0])

    # Verify metadata in TrainingJob.analysis_cache
    job = db_session.get(TrainingJob, test_job)
    assert job.analysis_cache is not None
    checkpoint_meta = job.analysis_cache.get("best_checkpoint")
    assert checkpoint_meta["epoch"] == 2
    assert abs(checkpoint_meta["value"] - 0.35) < 1e-5
    assert checkpoint_meta["restored"] is True


def test_checkpoint_fallback_to_training_loss(test_job: str, db_session: Session):
    """When val_loss is not available, callback cleanly falls back to monitoring training loss."""
    model = _create_simple_model()

    cb = BestModelCheckpointCallback(
        job_id=test_job,
        session_factory=lambda: db_session,
        monitor="val_loss",
    )
    cb.set_model(model)

    # Logs only contain training loss (no validation split)
    w1 = [np.ones((4, 2), dtype=np.float32) * 1.5]
    model.set_weights(w1)
    cb.on_epoch_end(0, logs={"loss": 0.65})
    assert cb.best_epoch == 1
    assert abs(cb.best_value - 0.65) < 1e-5

    w2 = [np.ones((4, 2), dtype=np.float32) * 2.5]
    model.set_weights(w2)
    cb.on_epoch_end(1, logs={"loss": 0.40})
    assert cb.best_epoch == 2
    assert abs(cb.best_value - 0.40) < 1e-5


def test_checkpoint_skips_restoration_on_cancelled_job(test_job: str, db_session: Session):
    """When a job is cancelled, optimal weight restoration is safely skipped."""
    model = _create_simple_model()

    cb = BestModelCheckpointCallback(
        job_id=test_job,
        session_factory=lambda: db_session,
        monitor="val_loss",
    )
    cb.set_model(model)

    w1 = [np.ones((4, 2), dtype=np.float32) * 1.0]
    model.set_weights(w1)
    cb.on_epoch_end(0, logs={"val_loss": 0.5})

    w2 = [np.ones((4, 2), dtype=np.float32) * 2.0]
    model.set_weights(w2)

    # Mark job as CANCELLED
    job = db_session.get(TrainingJob, test_job)
    job.status = TrainingStatus.CANCELLED
    db_session.add(job)
    db_session.commit()

    cb.on_train_end()
    assert cb.restored is False
    # Model retains w2 because restoration was bypassed
    np.testing.assert_allclose(model.get_weights()[0], w2[0])


def test_checkpoint_handles_nan_and_inf_gracefully(test_job: str, db_session: Session):
    """NaN and Inf metric values do not overwrite best checkpoint or raise exceptions."""
    model = _create_simple_model()

    cb = BestModelCheckpointCallback(
        job_id=test_job,
        session_factory=lambda: db_session,
        monitor="val_loss",
    )
    cb.set_model(model)

    w1 = [np.ones((4, 2), dtype=np.float32) * 1.0]
    model.set_weights(w1)
    cb.on_epoch_end(0, logs={"val_loss": 0.45})
    assert cb.best_epoch == 1

    # NaN epoch
    cb.on_epoch_end(1, logs={"val_loss": float("nan")})
    assert cb.best_epoch == 1
    assert abs(cb.best_value - 0.45) < 1e-5

    # Inf epoch
    cb.on_epoch_end(2, logs={"val_loss": float("inf")})
    assert cb.best_epoch == 1
    assert abs(cb.best_value - 0.45) < 1e-5


def test_checkpoint_emits_socketio_event(test_job: str, db_session: Session):
    """On improvement, checkpoint event is dispatched via Socket.IO."""
    model = _create_simple_model()
    mock_sio = MagicMock()
    mock_loop = MagicMock()
    mock_loop.is_running.return_value = True

    cb = BestModelCheckpointCallback(
        job_id=test_job,
        session_factory=lambda: db_session,
        monitor="val_loss",
        sio=mock_sio,
        loop=mock_loop,
    )
    cb.set_model(model)

    with patch("app.callbacks.metrics_callback.asyncio.run_coroutine_threadsafe") as mock_rcts:
        cb.on_epoch_end(0, logs={"val_loss": 0.22})
        assert cb.best_epoch == 1
        assert mock_rcts.called
        args, kwargs = mock_sio.emit.call_args
        assert args[0] == "result :::"
        payload = args[1]
        assert payload["type"] == "checkpoint"
        assert payload["job_id"] == test_job
        assert payload["epoch"] == 1
        assert payload["metric"] == "val_loss"
        assert abs(payload["value"] - 0.22) < 1e-5


def test_build_training_callbacks_includes_checkpoint_callback(test_job: str):
    """_build_training_callbacks includes BestModelCheckpointCallback as first callback."""
    callbacks = _build_training_callbacks(test_job)
    assert len(callbacks) == 3
    assert isinstance(callbacks[0], BestModelCheckpointCallback)
    assert callbacks[0].job_id == test_job
