"""Tests for dynamic learning rate schedulers, metric logging, and ReduceLROnPlateau."""

import numpy as np
import tensorflow as tf

from app.callbacks.lr_schedulers import (
    MetricLearningRateLogger,
    build_lr_scheduler_callback,
    get_current_learning_rate,
    get_supported_lr_schedulers,
)


def test_get_current_learning_rate():
    """Tests safe extraction of learning rate from Keras models."""
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    assert get_current_learning_rate(model) == 0.0

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss="mse")
    lr = get_current_learning_rate(model)
    assert round(lr, 3) == 0.005


def test_build_lr_scheduler_callback_variants():
    """Tests factory creation of plateau, exponential, and cosine schedulers."""
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

    # 4. Default logger
    cbs_default = build_lr_scheduler_callback(None)
    assert len(cbs_default) == 1
    assert isinstance(cbs_default[0], MetricLearningRateLogger)


def test_metric_lr_logger_captures_epoch_lr():
    """Tests that MetricLearningRateLogger populates lr keys in training logs."""
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02), loss="mse")

    logger_cb = MetricLearningRateLogger()
    X = np.ones((10, 2), dtype=np.float32)
    y = np.ones((10, 1), dtype=np.float32)

    history = model.fit(X, y, epochs=1, callbacks=[logger_cb], verbose=0)
    assert "lr" in history.history or "learning_rate" in history.history


def test_reduce_lr_on_plateau_execution():
    """Tests that stagnant validation loss triggers learning rate reduction."""
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    initial_lr = 0.1
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=initial_lr), loss="mse")

    plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
        verbose=0,
    )

    X = np.zeros((10, 2), dtype=np.float32)
    y = np.zeros((10, 1), dtype=np.float32)

    model.fit(X, y, epochs=4, callbacks=[plateau_cb], verbose=0)
    final_lr = get_current_learning_rate(model)
    assert final_lr < initial_lr


def test_get_supported_lr_schedulers():
    """Tests that supported schedulers catalog includes expected strategy names."""
    scheds = get_supported_lr_schedulers()
    assert "reduce_on_plateau" in scheds
    assert "exponential" in scheds
    assert "cosine" in scheds
    assert "step" in scheds


def test_step_decay_scheduler_execution():
    """Tests step decay scheduler callback drops learning rate predictably."""
    cbs_step = build_lr_scheduler_callback("step", {"step_size": 2, "gamma": 0.5}, initial_lr=0.08)
    assert len(cbs_step) == 2
    step_cb = next(cb for cb in cbs_step if isinstance(cb, tf.keras.callbacks.LearningRateScheduler))
    # Test schedule computation at epochs 0, 1, 2, 4
    assert step_cb.schedule(0, 0.08) == 0.08
    assert step_cb.schedule(1, 0.08) == 0.08
    assert step_cb.schedule(2, 0.08) == 0.04
    assert step_cb.schedule(4, 0.08) == 0.02


def test_get_current_learning_rate_fallback():
    """Tests fallback to 0.0 when model or optimizer lacks valid learning rate."""
    assert get_current_learning_rate(None) == 0.0

    class DummyModel:
        optimizer = None

    assert get_current_learning_rate(DummyModel()) == 0.0
