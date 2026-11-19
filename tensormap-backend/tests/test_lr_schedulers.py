"""Unit and integration tests for dynamic learning rate schedulers (Issue #558)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf
from pydantic import ValidationError
from sqlmodel import Session

from app.callbacks.lr_schedulers import (
    ReduceLROnPlateauCallback,
    _extract_lr,
    create_lr_scheduler_callback,
)
from app.callbacks.metrics_callback import MetricsCallback
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob
from app.models.training_metric import TrainingMetric
from app.schemas.training import TrainingStartRequest
from app.services.model_run import _build_training_callbacks, _training_loop_ctx


def test_extract_lr_none_optimizer():
    """Extracting lr from None safely returns None."""
    assert _extract_lr(None) is None


def test_extract_lr_from_keras_optimizer():
    """Extracting lr from standard Keras optimizer returns float."""
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    lr = _extract_lr(opt)
    assert lr is not None
    assert np.isclose(lr, 0.005, atol=1e-6)


def test_create_lr_scheduler_disabled_by_default():
    """When reduce_lr_on_plateau is not specified, factory returns None."""
    assert create_lr_scheduler_callback(job_id="job-1", hyperparams=None) is None
    assert create_lr_scheduler_callback(job_id="job-1", hyperparams={}) is None
    assert create_lr_scheduler_callback(job_id="job-1", hyperparams={"reduce_lr_on_plateau": False}) is None


def test_create_lr_scheduler_configured_parameters():
    """Factory sets custom factor, patience, min_lr, mode, and monitor."""
    cb = create_lr_scheduler_callback(
        job_id="job-custom",
        hyperparams={
            "reduce_lr_on_plateau": True,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 3,
            "reduce_lr_min_lr": 1e-5,
            "reduce_lr_min_delta": 1e-3,
            "reduce_lr_monitor": "val_loss",
            "reduce_lr_mode": "min",
        },
        has_validation=True,
    )
    assert cb is not None
    assert isinstance(cb, ReduceLROnPlateauCallback)
    assert cb.factor == 0.5
    assert cb.patience == 3
    assert cb.min_lr == 1e-5
    assert cb.min_delta == 1e-3
    assert cb.monitor == "val_loss"
    assert cb.mode == "min"
    assert cb.job_id == "job-custom"


def test_create_lr_scheduler_fallback_when_no_validation_data():
    """When validation data is absent, monitor adjusts from 'val_loss' to 'loss'."""
    cb = create_lr_scheduler_callback(
        job_id="job-fallback",
        hyperparams={"reduce_lr_on_plateau": True, "reduce_lr_monitor": "val_loss"},
        has_validation=False,
    )
    assert cb is not None
    assert cb.monitor == "loss"


def test_callback_records_lr_in_logs_on_epoch_end():
    """ReduceLROnPlateauCallback injects current learning rate into epoch logs."""
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss="mse")

    cb = ReduceLROnPlateauCallback(job_id="job-1", monitor="loss", factor=0.5, patience=2)
    cb.set_model(model)
    cb.on_train_begin()

    logs = {"loss": 1.0}
    cb.on_epoch_end(epoch=0, logs=logs)

    assert "lr" in logs
    assert np.isclose(logs["lr"], 0.01, atol=1e-6)


def test_callback_emits_event_on_reduction(monkeypatch):
    """When learning rate decreases, a Socket.IO event is scheduled."""
    emit_mock = MagicMock()
    monkeypatch.setattr("app.callbacks.lr_schedulers.schedule_room_emit", emit_mock)

    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="mse")

    sio_mock = MagicMock()
    loop_mock = MagicMock()
    cb = ReduceLROnPlateauCallback(
        job_id="job-lr-event",
        sio_instance=sio_mock,
        loop=loop_mock,
        monitor="loss",
        factor=0.5,
        patience=1,
    )
    cb.set_model(model)
    cb.on_train_begin()

    # Simulate learning rate reduction
    opt.learning_rate.assign(0.005)
    logs = {"loss": 1.0}
    cb.on_epoch_end(epoch=1, logs=logs)

    emit_mock.assert_called_once()
    args, _ = emit_mock.call_args
    assert args[0] is sio_mock
    assert args[1] is loop_mock
    assert args[2] == "job-lr-event"
    assert args[3]["type"] == "learning_rate_reduced"
    assert args[3]["job_id"] == "job-lr-event"
    assert args[3]["epoch"] == 2
    assert np.isclose(args[3]["old_lr"], 0.01, atol=1e-6)
    assert np.isclose(args[3]["new_lr"], 0.005, atol=1e-6)


def test_metrics_callback_persists_lr_metric(db_session: Session):
    """MetricsCallback persists lr to training_metric table and emits it."""
    model = ModelBasic(
        model_name="lr_metrics_model",
        model_type=1,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job = TrainingJob(
        id="job-metrics-lr",
        model_id=model.id,
        status="RUNNING",
    )
    db_session.add(job)
    db_session.commit()

    emit_mock = MagicMock()
    cb = MetricsCallback(
        job_id="job-metrics-lr",
        session_factory=lambda: db_session,
        sio_instance=MagicMock(),
        loop=MagicMock(),
    )
    cb._emit = emit_mock

    # Mock model with optimizer learning rate
    model_mock = MagicMock()
    opt_mock = MagicMock()
    opt_mock.learning_rate.numpy.return_value = 0.002
    model_mock.optimizer = opt_mock
    cb.set_model(model_mock)

    cb.on_epoch_end(epoch=0, logs={"loss": 0.5, "accuracy": 0.8})

    # Verify persisted TrainingMetric records
    from sqlmodel import select

    metrics = db_session.exec(select(TrainingMetric).where(TrainingMetric.job_id == "job-metrics-lr")).all()
    metric_map = {m.metric_name: m.metric_value for m in metrics}
    assert "loss" in metric_map
    assert "lr" in metric_map
    assert np.isclose(metric_map["lr"], 0.002, atol=1e-6)

    # Verify Socket.IO emit contained lr
    emit_mock.assert_called_once()
    emitted_event = emit_mock.call_args[0][0]
    assert np.isclose(emitted_event["lr"], 0.002, atol=1e-6)


def test_build_training_callbacks_attaches_scheduler():
    """_build_training_callbacks includes ReduceLROnPlateauCallback when enabled."""
    token = _training_loop_ctx.set(MagicMock())
    try:
        # When disabled
        callbacks_default = _build_training_callbacks(job_id="job-1", hyperparams={})
        assert len(callbacks_default) == 2  # MetricsCallback and CancellationCheckCallback

        # When enabled
        callbacks_with_lr = _build_training_callbacks(
            job_id="job-1",
            hyperparams={"reduce_lr_on_plateau": True, "reduce_lr_factor": 0.1},
            has_validation=True,
        )
        assert len(callbacks_with_lr) == 3
        lr_cbs = [cb for cb in callbacks_with_lr if isinstance(cb, ReduceLROnPlateauCallback)]
        assert len(lr_cbs) == 1
        assert lr_cbs[0].factor == 0.1
    finally:
        _training_loop_ctx.reset(token)


def test_training_start_request_schema_validation():
    """TrainingStartRequest validates learning rate reduction options."""
    req = TrainingStartRequest(
        model_name="my_model",
        reduce_lr_on_plateau=True,
        reduce_lr_patience=4,
        reduce_lr_factor=0.3,
        reduce_lr_min_lr=1e-5,
    )
    assert req.reduce_lr_on_plateau is True
    assert req.reduce_lr_patience == 4
    assert req.reduce_lr_factor == 0.3
    assert req.reduce_lr_min_lr == 1e-5

    # Invalid factor >= 1.0 or <= 0.0
    with pytest.raises(ValidationError):
        TrainingStartRequest(model_name="my_model", reduce_lr_factor=1.5)

    with pytest.raises(ValidationError):
        TrainingStartRequest(model_name="my_model", reduce_lr_patience=0)
