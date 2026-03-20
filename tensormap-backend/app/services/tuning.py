"""Hyperparameter tuning engine — grid search and random search."""

from __future__ import annotations

import asyncio
import itertools
import random
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sqlmodel import Session, select

from app.models.ml import ModelBasic
from app.services.model_run import (
    _helper_generate_file_location,
    _helper_generate_json_model_file_location,
)
from app.shared.constants import SOCKETIO_DL_NAMESPACE, SOCKETIO_LISTENER
from app.shared.logging_config import get_logger
from app.socketio_instance import sio

logger = get_logger(__name__)

_main_loop: asyncio.AbstractEventLoop | None = None


def _emit(data: dict) -> None:
    if _main_loop is not None and _main_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            sio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE),
            _main_loop,
        )
        try:
            future.result(timeout=5)
        except Exception:
            logger.warning("Tuning emit failed: %s", data)


def _resp(status_code: int, success: bool, message: str, data=None):
    return {"success": success, "message": message, "data": data}, status_code


# ── Search space helpers ──────────────────────────────────────────────────────


def _grid_trials(space: dict) -> list[dict]:
    """Expand a grid search space into all combinations."""
    keys = list(space.keys())
    values = list(space.values())
    return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)]


def _random_trials(space: dict, n_trials: int, seed: int = 42) -> list[dict]:
    """Sample n_trials random combinations from the search space."""
    rng = random.Random(seed)
    trials = []
    for _ in range(n_trials):
        trial = {k: rng.choice(v) for k, v in space.items()}
        trials.append(trial)
    return trials


# ── Single trial runner ───────────────────────────────────────────────────────


def _run_trial(
    model_name: str,
    db: Session,
    trial: dict,
    trial_num: int,
    total_trials: int,
) -> dict:
    """Train one trial and return its result dict."""
    config = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()

    # Load fresh model architecture each trial
    json_path = _helper_generate_json_model_file_location(model_name)
    with open(json_path) as f:
        model = tf.keras.models.model_from_json(f.read())

    optimizer_name = trial.get("optimizer", config.optimizer or "adam")
    lr = trial.get("learning_rate", 0.001)
    batch_size = int(trial.get("batch_size", config.batch_size or 32))
    epochs = int(trial.get("epochs", config.epochs or 10))

    # Build optimizer with learning rate
    optimizer_map = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "adagrad": tf.keras.optimizers.Adagrad,
        "adamw": tf.keras.optimizers.AdamW,
    }
    opt_cls = optimizer_map.get(optimizer_name, tf.keras.optimizers.Adam)
    optimizer = opt_cls(learning_rate=float(lr))

    if config.loss == "sparse_categorical_crossentropy":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss, metrics=[config.metric or "accuracy"])

    # Load data
    file_location = _helper_generate_file_location(db, file_id=config.file_id)
    features = pd.read_csv(file_location)
    features.dropna(inplace=True)
    features = features.sample(frac=1, random_state=42).reset_index(drop=True)
    X = features.drop(config.target_field, axis=1).values.astype(np.float32)
    y = features[config.target_field].values
    split_index = int(len(X) * config.training_split / 100)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    _emit(
        {
            "type": "tuning_trial_start",
            "trial": trial_num,
            "total_trials": total_trials,
            "params": trial,
            "message": f"Trial {trial_num}/{total_trials} — {trial}",
            "test": 0,
        }
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    val_loss = float(history.history["val_loss"][-1])
    train_loss = float(history.history["loss"][-1])
    metric_key = config.metric or "accuracy"
    val_metric = float(history.history.get(f"val_{metric_key}", [0])[-1])
    train_metric = float(history.history.get(metric_key, [0])[-1])

    result = {
        "trial": trial_num,
        "params": trial,
        "val_loss": round(val_loss, 6),
        "train_loss": round(train_loss, 6),
        "val_metric": round(val_metric, 6),
        "train_metric": round(train_metric, 6),
        "metric_name": metric_key,
    }

    _emit(
        {
            "type": "tuning_trial_end",
            "trial": trial_num,
            "total_trials": total_trials,
            "result": result,
            "message": (
                f"Trial {trial_num}/{total_trials} done — val_loss: {val_loss:.4f}, val_{metric_key}: {val_metric:.4f}"
            ),
            "test": 0,
        }
    )

    return result


# ── Public service functions ──────────────────────────────────────────────────


def run_tuning_service(
    db: Session,
    model_name: str,
    strategy: str,
    space: dict[str, list[Any]],
    n_trials: int = 10,
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple:
    """Run grid or random hyperparameter search and return all trial results."""
    global _main_loop
    _main_loop = loop

    config = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not config:
        return _resp(404, False, "Model not found")
    if config.file_id is None:
        return _resp(400, False, "Training configuration not set")

    if strategy == "grid":
        trials = _grid_trials(space)
    elif strategy == "random":
        trials = _random_trials(space, n_trials=n_trials)
    else:
        return _resp(400, False, f"Unknown strategy: {strategy!r}. Use 'grid' or 'random'")

    total = len(trials)
    _emit(
        {
            "type": "tuning_start",
            "strategy": strategy,
            "total_trials": total,
            "message": f"Starting {strategy} search — {total} trials",
            "test": 0,
        }
    )

    results = []
    try:
        for i, trial in enumerate(trials, start=1):
            result = _run_trial(model_name, db, trial, i, total)
            results.append(result)
    except Exception as e:
        logger.exception("Tuning failed at trial %d: %s", len(results) + 1, e)
        _emit({"type": "tuning_error", "message": str(e), "test": -1})
        return _resp(500, False, f"Tuning failed: {e}")

    # Best trial = lowest val_loss
    best = min(results, key=lambda r: r["val_loss"])

    _emit(
        {
            "type": "tuning_end",
            "best": best,
            "total_trials": total,
            "message": (f"Search complete — best val_loss: {best['val_loss']:.4f} with params: {best['params']}"),
            "test": 4,
        }
    )

    return _resp(
        200,
        True,
        "Hyperparameter search complete",
        {
            "strategy": strategy,
            "total_trials": total,
            "results": results,
            "best": best,
        },
    )
