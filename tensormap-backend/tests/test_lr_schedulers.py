"""Unit and integration tests for dynamic learning rate schedulers (Issue #558)."""

import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf
from pydantic import ValidationError
from sqlmodel import Session

from app.callbacks.lr_schedulers import (
    MetricLearningRateLogger,
    ReduceLROnPlateauCallback,
    _extract_lr,
    build_lr_scheduler_callback,
    create_lr_scheduler_callback,
    get_current_learning_rate,
    get_supported_lr_schedulers,
)
from app.callbacks.metrics_callback import MetricsCallback
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob
from app.models.training_metric import TrainingMetric
from app.schemas.training import TrainingStartRequest
from app.services.model_run import _build_training_callbacks, _training_loop_ctx


def _create_simple_model() -> tf.keras.Model:
    """Helper creating a simple Keras Sequential model without deprecated arguments."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(1),
        ]
    )


def test_extract_lr_none_optimizer():
    """Extracting lr from None safely returns None."""
    assert _extract_lr(None) is None


def test_extract_lr_from_keras_optimizer():
    """Extracting lr from standard Keras optimizer returns float."""
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    lr = _extract_lr(opt)
    assert lr is not None
    assert np.isclose(lr, 0.005, atol=1e-6)


def test_get_current_learning_rate():
    """Tests safe extraction of learning rate from Keras models."""
    model = _create_simple_model()
    # Uncompiled model has no optimizer, so should return None
    assert get_current_learning_rate(model) is None

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss="mse")
    lr = get_current_learning_rate(model)
    assert lr is not None
    assert round(lr, 3) == 0.005


def test_get_current_learning_rate_fallback():
    """Tests fallback to None when model or optimizer lacks valid learning rate."""
    assert get_current_learning_rate(None) is None

    class DummyModel:
        optimizer = None

    assert get_current_learning_rate(DummyModel()) is None


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
    model = _create_simple_model()
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

    model = _create_simple_model()
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
    unique_id = uuid.uuid4().hex[:8]
    model = ModelBasic(
        model_name=f"lr_metrics_model_{unique_id}",
        model_type=1,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = f"job-metrics-lr-{unique_id}"
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status="RUNNING",
    )
    db_session.add(job)
    db_session.commit()

    emit_mock = MagicMock()
    cb = MetricsCallback(
        job_id=job_id,
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

    metrics = db_session.exec(select(TrainingMetric).where(TrainingMetric.job_id == job_id)).all()
    metric_map = {m.metric_name: m.metric_value for m in metrics}
    assert "loss" in metric_map
    assert "lr" in metric_map
    assert np.isclose(metric_map["lr"], 0.002, atol=1e-6)

    # Verify Socket.IO emit contained lr
    emit_mock.assert_called_once()
    emitted_event = emit_mock.call_args[0][0]
    assert np.isclose(emitted_event["lr"], 0.002, atol=1e-6)


def test_build_lr_scheduler_callback_variants():
    """Tests factory creation of plateau, exponential, cosine, and step schedulers."""
    # 1. Reduce on plateau
    cbs_plateau = build_lr_scheduler_callback(
        "reduce_on_plateau",
        {"patience": 2, "factor": 0.2, "min_lr": 1e-5},
    )
    assert any(isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau) for cb in cbs_plateau)
    plateau_cb = next(cb for cb in cbs_plateau if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau))
    assert plateau_cb.patience == 2
    assert plateau_cb.factor == 0.2
    assert plateau_cb.min_lr == 1e-5

    # 2. Exponential decay
    cbs_exp = build_lr_scheduler_callback("exponential", {"decay_rate": 0.9}, initial_lr=0.01)
    assert any(isinstance(cb, tf.keras.callbacks.LearningRateScheduler) for cb in cbs_exp)

    # 3. Cosine annealing
    cbs_cosine = build_lr_scheduler_callback("cosine", {"total_epochs": 15}, initial_lr=0.01)
    assert any(isinstance(cb, tf.keras.callbacks.LearningRateScheduler) for cb in cbs_cosine)

    # 4. Step decay
    cbs_step = build_lr_scheduler_callback("step", {"step_size": 3, "gamma": 0.5}, initial_lr=0.01)
    assert any(isinstance(cb, tf.keras.callbacks.LearningRateScheduler) for cb in cbs_step)

    # 5. Default logger
    cbs_default = build_lr_scheduler_callback(None)
    assert len(cbs_default) == 1
    assert isinstance(cbs_default[0], MetricLearningRateLogger)


def test_unknown_scheduler_raises_value_error():
    """Tests that unknown scheduler names raise ValueError referencing supported options."""
    with pytest.raises(ValueError, match="Unsupported learning rate scheduler 'invalid_name'"):
        build_lr_scheduler_callback("invalid_name")


def test_scheduler_invalid_configs_raise_value_error():
    """Tests that out-of-range scheduler parameters raise ValueError with informative messages."""
    with pytest.raises(ValueError, match="factor must be between 0 and 1"):
        build_lr_scheduler_callback("reduce_on_plateau", {"factor": 1.5})

    with pytest.raises(ValueError, match="patience must be at least 1"):
        build_lr_scheduler_callback("reduce_on_plateau", {"patience": 0})

    with pytest.raises(ValueError, match="decay_rate must be between 0 and 1"):
        build_lr_scheduler_callback("exponential", {"decay_rate": -0.1})

    with pytest.raises(ValueError, match="total_epochs must be at least 1"):
        build_lr_scheduler_callback("cosine", {"total_epochs": 0})

    with pytest.raises(ValueError, match="step_size must be at least 1"):
        build_lr_scheduler_callback("step", {"step_size": 0})


def test_exponential_and_cosine_rate_progression():
    """Tests exact numerical progression of learning rates for exponential and cosine schedulers."""
    initial_lr = 0.01

    # Exponential decay
    cbs_exp = build_lr_scheduler_callback("exponential", {"decay_rate": 0.9}, initial_lr=initial_lr)
    exp_cb = next(cb for cb in cbs_exp if isinstance(cb, tf.keras.callbacks.LearningRateScheduler))
    assert abs(exp_cb.schedule(0, initial_lr) - 0.01) < 1e-6
    assert abs(exp_cb.schedule(1, initial_lr) - 0.009) < 1e-6
    assert abs(exp_cb.schedule(2, initial_lr) - 0.0081) < 1e-6

    # Cosine annealing
    total_epochs = 10
    min_lr = 0.001
    cbs_cos = build_lr_scheduler_callback(
        "cosine",
        {"total_epochs": total_epochs, "min_lr": min_lr},
        initial_lr=initial_lr,
    )
    cos_cb = next(cb for cb in cbs_cos if isinstance(cb, tf.keras.callbacks.LearningRateScheduler))
    assert abs(cos_cb.schedule(0, initial_lr) - initial_lr) < 1e-6
    assert abs(cos_cb.schedule(total_epochs, initial_lr) - min_lr) < 1e-6
    # Midpoint should be halfway between initial_lr and min_lr
    midpoint_lr = min_lr + 0.5 * (initial_lr - min_lr)
    assert abs(cos_cb.schedule(5, initial_lr) - midpoint_lr) < 1e-6


def test_no_validation_monitor_switch():
    """Tests that ReduceLROnPlateau switches monitor from val_loss to loss when has_validation=False."""
    cbs = build_lr_scheduler_callback("reduce_on_plateau", {"has_validation": False, "monitor": "val_loss"})
    plateau_cb = next(cb for cb in cbs if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau))
    assert plateau_cb.monitor == "loss"


def test_metric_lr_logger_captures_epoch_lr():
    """Tests that MetricLearningRateLogger populates lr in epoch logs."""
    model = _create_simple_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02), loss="mse")

    logger_cb = MetricLearningRateLogger()
    X = np.ones((10, 2), dtype=np.float32)
    y = np.ones((10, 1), dtype=np.float32)

    history = model.fit(X, y, epochs=1, callbacks=[logger_cb], verbose=0)
    assert "lr" in history.history
    assert round(history.history["lr"][0], 3) == 0.02


def test_reduce_lr_on_plateau_execution():
    """Tests that stagnant validation loss triggers learning rate reduction using factory callback."""
    model = _create_simple_model()
    initial_lr = 0.1
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=initial_lr), loss="mse")

    cbs = build_lr_scheduler_callback(
        "reduce_on_plateau",
        {"monitor": "loss", "factor": 0.5, "patience": 1, "min_lr": 1e-5},
        initial_lr=initial_lr,
    )

    X = np.zeros((10, 2), dtype=np.float32)
    y = np.zeros((10, 1), dtype=np.float32)

    model.fit(X, y, epochs=4, callbacks=cbs, verbose=0)
    final_lr = get_current_learning_rate(model)
    assert final_lr is not None
    assert final_lr < initial_lr


def test_build_training_callbacks_attaches_scheduler():
    """_build_training_callbacks includes ReduceLROnPlateauCallback when enabled."""
    token = _training_loop_ctx.set(MagicMock())
    try:
        # When disabled
        callbacks_default = _build_training_callbacks(job_id="job-1", hyperparams={})
        assert len(callbacks_default) == 2  # MetricsCallback and CancellationCheckCallback

        # When enabled via reduce_lr_on_plateau
        callbacks_with_lr = _build_training_callbacks(
            job_id="job-1",
            hyperparams={"reduce_lr_on_plateau": True, "reduce_lr_factor": 0.1},
            has_validation=True,
        )
        assert len(callbacks_with_lr) == 3
        lr_cbs = [cb for cb in callbacks_with_lr if isinstance(cb, ReduceLROnPlateauCallback)]
        assert len(lr_cbs) == 1
        assert lr_cbs[0].factor == 0.1

        # When enabled via scheduler name
        cbs_cosine = _build_training_callbacks(
            job_id="job-2",
            hyperparams={"scheduler": "cosine", "scheduler_config": {"total_epochs": 10}},
            initial_lr=0.005,
            total_epochs=10,
        )
        assert len(cbs_cosine) >= 3
        assert any(isinstance(cb, tf.keras.callbacks.LearningRateScheduler) for cb in cbs_cosine)
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


def test_get_supported_lr_schedulers():
    """Tests that supported schedulers catalog includes expected canonical strategy names."""
    scheds = get_supported_lr_schedulers()
    assert "reduce_on_plateau" in scheds
    assert "exponential" in scheds
    assert "cosine" in scheds
    assert "step" in scheds
