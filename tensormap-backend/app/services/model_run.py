import asyncio
import os

import pandas as pd
import tensorflow as tf
from sqlmodel import Session, select

from app.config import get_settings
from app.models.data import DataFile, ImageProperties
from app.models.ml import ModelBasic
from app.shared.constants import (
    MODEL_GENERATION_LOCATION,
    MODEL_GENERATION_TYPE,
    SOCKETIO_DL_NAMESPACE,
    SOCKETIO_LISTENER,
)
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger
from app.socketio_instance import sio

logger = get_logger(__name__)

_main_loop: asyncio.AbstractEventLoop | None = None

# ── Socket.IO helpers ──────────────────────────────────────────────────────────


def _emit(data: dict) -> None:
    """Thread-safe Socket.IO emit into the running asyncio event loop."""
    if _main_loop is not None and _main_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            sio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE),
            _main_loop,
        )
        try:
            future.result(timeout=5)
        except Exception:
            logger.warning("Failed to emit Socket.IO event: %s", data)
    else:
        logger.warning("No running event loop for Socket.IO emit: %s", data)


def _model_result(message: str, test: int) -> None:
    """Emit a legacy text progress event (kept for backwards compatibility)."""
    _emit({"message": message, "test": test})


# ── Keras callbacks ────────────────────────────────────────────────────────────


class CustomProgressBar(tf.keras.callbacks.Callback):
    """Emits both legacy text events AND structured epoch metrics via Socket.IO."""

    def on_train_begin(self, logs=None):
        _emit({"type": "train_begin", "message": "Starting", "test": 0})

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        _emit(
            {
                "type": "epoch_begin",
                "message": f"Epoch {epoch + 1}/{self.params['epochs']}",
                "epoch": epoch + 1,
                "total_epochs": self.params["epochs"],
                "test": 0,
            }
        )

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        # Detect metric name dynamically
        metric_name = "accuracy" if "accuracy" in logs else "mse" if "mse" in logs else None
        val_metric_key = f"val_{metric_name}" if metric_name else None

        payload = {
            "type": "epoch_metrics",
            "epoch": epoch + 1,
            "total_epochs": self.params["epochs"],
            "loss": round(float(logs.get("loss", 0)), 6),
            "val_loss": round(float(logs["val_loss"]), 6) if "val_loss" in logs else None,
            "metric_name": metric_name,
            "metric": round(float(logs[metric_name]), 6) if metric_name and metric_name in logs else None,
            "val_metric": round(float(logs[val_metric_key]), 6) if val_metric_key and val_metric_key in logs else None,
            # Legacy text fields so existing Result panel still works
            "message": (
                f"Epoch {epoch + 1}/{self.params['epochs']} — "
                f"loss: {logs.get('loss', 0):.4f}"
                + (f" — {metric_name}: {logs[metric_name]:.4f}" if metric_name and metric_name in logs else "")
                + (f" — val_loss: {logs['val_loss']:.4f}" if "val_loss" in logs else "")
            ),
            "test": 0,
        }
        _emit(payload)

    def on_batch_end(self, batch: int, logs=None) -> None:
        logs = logs or {}
        progress = batch / self.params["steps"] if self.params["steps"] else 0
        progress_bar_width = 50
        arrow = ">" * int(progress * progress_bar_width)
        spaces = "=" * (progress_bar_width - len(arrow))
        if "mse" in logs:
            metric = f"MSE: {logs['mse']:.4f}"
        elif "accuracy" in logs:
            metric = f"Accuracy: {logs['accuracy']:.4f}"
        else:
            metric = ""
        _emit(
            {
                "type": "batch_end",
                "message": (
                    f"{batch + 1}/{self.params['steps']}  [{arrow}{spaces}] "
                    f"{int(progress * 100)}% - Loss: {logs['loss']:.4f} - {metric}"
                ),
                "test": 1,
            }
        )

    def on_test_begin(self, logs=None) -> None:
        _emit({"type": "test_begin", "message": "Evaluating...", "test": 2})

    def on_test_end(self, logs=None) -> None:
        logs = logs or {}
        if "mse" in logs:
            metric = f"MSE Loss- {logs['mse']:.4f}"
        elif "accuracy" in logs:
            metric = f"Accuracy- {logs['accuracy']:.4f}"
        else:
            metric = ""
        _emit({"type": "test_end", "message": f"Evaluation Results: {metric} Loss - {logs['loss']:.4f}", "test": 3})
        _emit({"type": "train_end", "message": "Finish", "test": 4})


# ── DB / file helpers ──────────────────────────────────────────────────────────


def _helper_generate_file_location(db: Session, file_id) -> str:
    upload_folder = get_settings().upload_folder
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if file.file_type == "zip":
        return upload_folder + "/" + file.file_name
    return upload_folder + "/" + file.file_name + "." + file.file_type


def _helper_generate_json_model_file_location(model_name: str) -> str:
    path = os.path.realpath(os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE))
    base_dir = os.path.realpath(MODEL_GENERATION_LOCATION)
    if not path.startswith(base_dir + os.sep) and path != base_dir:
        raise ValueError("Invalid model path: escapes model directory")
    return path


# ── Entry point ────────────────────────────────────────────────────────────────


def model_run(model_name: str, db: Session, loop: asyncio.AbstractEventLoop | None = None) -> None:
    global _main_loop
    _main_loop = loop
    try:
        _run(model_name, db)
    except Exception as e:
        logger.exception("Training failed for model '%s': %s", model_name, str(e))
        _emit({"type": "error", "message": f"Training failed: {e}", "test": -1})
        raise


def _run(model_name: str, db: Session) -> None:
    model_configs = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()

    if model_configs.model_type == ProblemType.IMAGE_CLASSIFICATION:
        image_properties = db.exec(select(ImageProperties).where(ImageProperties.id == model_configs.file_id)).first()
        image_size = (image_properties.image_size, image_properties.image_size)
        batch_size = image_properties.batch_size
        color_mode = image_properties.color_mode
        label_mode = image_properties.label_mode
        directory = _helper_generate_file_location(db, file_id=model_configs.file_id)
        validation_split = 1 - (model_configs.training_split / 100)
        train_data = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode=label_mode,
        )
        test_data = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode=label_mode,
        )
    else:
        file_location = _helper_generate_file_location(db, file_id=model_configs.file_id)
        features = pd.read_csv(file_location)
        features.dropna(inplace=True)
        features = features.sample(frac=1, random_state=42).reset_index(drop=True)
        X = features.drop(model_configs.target_field, axis=1)
        y = features[model_configs.target_field]
        split_index = int(len(X) * model_configs.training_split / 100)
        x_training = X[:split_index]
        y_training = y[:split_index]
        x_testing = X[split_index:]
        y_testing = y[split_index:]
        batch_size = model_configs.batch_size if model_configs.batch_size is not None else 32

    with open(_helper_generate_json_model_file_location(model_name=model_name)) as f:
        json_string = f.read()
    model = tf.keras.models.model_from_json(json_string, custom_objects=None)
    model.summary()

    if model_configs.loss == "sparse_categorical_crossentropy":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=model_configs.optimizer,
        loss=loss,
        metrics=[model_configs.metric],
    )

    cb = CustomProgressBar()
    if model_configs.model_type == ProblemType.IMAGE_CLASSIFICATION:
        model.fit(
            train_data,
            validation_data=test_data,
            epochs=model_configs.epochs,
            callbacks=[cb],
            verbose=0,
        )
        model.evaluate(test_data, callbacks=[cb], verbose=0)
    else:
        model.fit(
            x_training,
            y_training,
            epochs=model_configs.epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[cb],
            verbose=0,
        )
        model.evaluate(x_testing, y_testing, callbacks=[cb], verbose=0)
