"""Service layer for learning rate range test and optimal learning rate finding."""

import math
import os
from typing import Any

import pandas as pd
import tensorflow as tf
from sqlmodel import Session, select

from app.callbacks.lr_finder import LRFinderCallback
from app.exceptions import AppException
from app.models.ml import ModelBasic
from app.services.model_run import (
    _build_optimizer,
    _emits_logits,
    _helper_generate_file_location,
    _helper_generate_json_model_file_location,
    _prepare_training_data,
    resolve_loss,
)
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def find_optimal_learning_rate(
    model_name: str,
    db: Session,
    start_lr: float = 1e-6,
    end_lr: float = 1.0,
    num_batches: int = 50,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Execute an automated learning rate range test profiler.

    Runs an exponential learning rate sweep over minibatches, smooths losses,
    identifies the optimal learning rate via steepest descent, and safely
    restores model weights.
    """
    model_configs = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not model_configs:
        raise AppException(404, f"Model configuration not found: {model_name}")

    if model_configs.file_id is None:
        raise AppException(400, "Dataset not linked to model. Please link a dataset first.")

    file_location = _helper_generate_file_location(db, file_id=model_configs.file_id)
    if not os.path.isfile(file_location):
        raise AppException(404, f"Dataset file not found at: {file_location}")

    features = pd.read_csv(file_location)
    training_split = model_configs.training_split or 80.0

    try:
        x_training, y_training, _, _ = _prepare_training_data(
            features, model_configs.target_field, training_split=training_split
        )
    except ValueError as e:
        raise AppException(400, str(e)) from e

    model_json_path = _helper_generate_json_model_file_location(model_name=model_name)
    if not os.path.isfile(model_json_path):
        raise AppException(404, f"Model architecture file not found: {model_json_path}")

    with open(model_json_path) as f:
        json_string = f.read()

    model = tf.keras.models.model_from_json(json_string, custom_objects=None)

    optimizer_name = model_configs.optimizer
    optimizer = _build_optimizer(optimizer_name, lr=start_lr)
    loss_fn = resolve_loss(model_configs.loss, from_logits=_emits_logits(model))
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[model_configs.metric])

    effective_batch_size = batch_size or model_configs.batch_size or 32
    batches_per_epoch = max(1, len(x_training) // effective_batch_size)
    epochs_required = math.ceil(num_batches / batches_per_epoch)

    lr_finder_cb = LRFinderCallback(
        start_lr=start_lr,
        end_lr=end_lr,
        num_batches=num_batches,
    )

    logger.info(
        "Starting LR finder range test for model '%s': start_lr=%s, end_lr=%s, num_batches=%d",
        model_name,
        start_lr,
        end_lr,
        num_batches,
    )

    model.fit(
        x_training,
        y_training,
        epochs=epochs_required,
        batch_size=effective_batch_size,
        callbacks=[lr_finder_cb],
        verbose=0,
    )

    results = lr_finder_cb.get_results()
    logger.info(
        "LR finder range test finished: suggested_lr=%s, min_loss=%s",
        results.get("suggested_lr"),
        results.get("min_loss"),
    )
    return results
