import asyncio
import contextlib
import contextvars
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sqlmodel import Session, select

from app.callbacks import CancellationCheckCallback, MetricsCallback
from app.config import get_settings
from app.models.data import DataFile, ImageProperties
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.shared.constants import (
    MODEL_GENERATION_LOCATION,
    MODEL_GENERATION_TYPE,
    SOCKETIO_DL_NAMESPACE,
    SOCKETIO_LISTENER,
)
from app.shared.enums import LossFunction, ProblemType
from app.shared.logging_config import get_logger
from app.shared.training_events import training_event
from app.socketio_instance import sio

logger = get_logger(__name__)


class ModelRunError(Exception):
    """Raised when model training cannot proceed due to missing data."""

    pass


def _validate_training_params(batch_size, epochs, training_split):
    """Validate training hyper-parameters before use (Issue #4).

    Raises ValueError with a clear message instead of letting TensorFlow
    crash with a cryptic error on bad values.
    """
    errors = []
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
        errors.append(f"batch_size must be a positive integer, got {batch_size!r}")
    if epochs is None or not isinstance(epochs, int) or epochs <= 0:
        errors.append(f"epochs must be a positive integer, got {epochs!r}")
    if training_split is None:
        errors.append("training_split must be specified")
    else:
        try:
            ts = float(training_split)
        except (TypeError, ValueError):
            errors.append(f"training_split must be a number, got {training_split!r}")
        else:
            if not (0 < ts < 100):
                errors.append(f"training_split must be between 0 and 100, got {ts}")
    if errors:
        raise ValueError("Invalid training parameters: " + "; ".join(errors))


def compute_balanced_class_weights(y: Any) -> dict[int, float] | None:
    """Compute balanced class weights inversely proportional to class frequencies.

    Formula: weight_c = N_samples / (n_classes * count(y == c))
    Supports 1D integer/float arrays, 1D string/object label series/arrays,
    and 2D one-hot encoded arrays/DataFrames. Returns dict mapping class index (int) to weight (float).
    """
    if y is None:
        return None

    # Extract numpy array from pandas Series/DataFrame or sequence
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)

    if y_arr.size == 0:
        return None

    # Handle 2D one-hot encoded targets (e.g. from pd.get_dummies or to_categorical)
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        counts = np.sum(y_arr, axis=0)
        n_samples = float(y_arr.shape[0])
        n_classes = float(y_arr.shape[1])
        if n_classes <= 1 or n_samples == 0:
            return None
        weights: dict[int, float] = {}
        for c in range(int(n_classes)):
            cnt = float(counts[c])
            if cnt > 0:
                weights[c] = round(float(n_samples / (n_classes * cnt)), 4)
        return weights

    # Flatten 1D or (N, 1) array
    y_flat = y_arr.ravel()
    n_samples = float(len(y_flat))
    if n_samples == 0:
        return None

    # Check if labels are non-numeric (strings or objects)
    if y_flat.dtype.kind in {"U", "S", "O"}:
        unique_classes, inverse_indices = np.unique(y_flat, return_inverse=True)
        classes = np.arange(len(unique_classes))
        counts = np.bincount(inverse_indices)
    else:
        classes, counts = np.unique(y_flat, return_counts=True)

    n_classes = float(len(classes))
    if n_classes <= 1:
        return None

    weights = {}
    for c, count in zip(classes, counts, strict=False):
        if count > 0:
            try:
                class_idx = int(c)
            except (ValueError, TypeError):
                continue
            weights[class_idx] = round(float(n_samples / (n_classes * float(count))), 4)

    return weights if weights else None


# Keras loss classes keyed by the name stored on ModelBasic.loss.
_LOSS_CLASSES: dict[str, type[tf.keras.losses.Loss]] = {
    LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY: tf.keras.losses.SparseCategoricalCrossentropy,
    LossFunction.CATEGORICAL_CROSSENTROPY: tf.keras.losses.CategoricalCrossentropy,
    LossFunction.BINARY_CROSSENTROPY: tf.keras.losses.BinaryCrossentropy,
    LossFunction.MEAN_SQUARED_ERROR: tf.keras.losses.MeanSquaredError,
    LossFunction.MEAN_ABSOLUTE_ERROR: tf.keras.losses.MeanAbsoluteError,
    LossFunction.HUBER: tf.keras.losses.Huber,
}
# Cross-entropy losses need to know whether the output layer already applies
# softmax/sigmoid; the regression losses take no such flag.
_LOGIT_LOSSES = frozenset(
    {
        LossFunction.SPARSE_CATEGORICAL_CROSSENTROPY,
        LossFunction.CATEGORICAL_CROSSENTROPY,
        LossFunction.BINARY_CROSSENTROPY,
    }
)


def _emits_logits(model: Any) -> bool:
    """True unless the model's output layer already applies softmax or sigmoid.

    Accepts an instantiated ``tf.keras.Model``, a serialized model JSON string,
    or a parsed model dictionary. Single source of truth shared between training
    and code generation.
    """
    if isinstance(model, (dict, str)):
        data = json.loads(model) if isinstance(model, str) else model
        config = data.get("config", {}) if isinstance(data, dict) else {}
        layers = config.get("layers", [])
        if not layers:
            return True

        output_layers = config.get("output_layers")
        if output_layers and isinstance(output_layers, list) and output_layers[0]:
            target_name = output_layers[0][0]
            output_layer = next(
                (
                    layer
                    for layer in layers
                    if layer.get("config", {}).get("name") == target_name or layer.get("name") == target_name
                ),
                layers[-1],
            )
        else:
            output_layer = layers[-1]

        class_name = str(output_layer.get("class_name", "")).lower()
        if class_name in ("softmax", "sigmoid"):
            return False

        layer_config = output_layer.get("config", {})
        act = layer_config.get("activation")
        act_name = str(act).lower() if act else ""
        return act_name not in ("softmax", "sigmoid")

    output_layer = model.layers[-1]
    activation = getattr(output_layer, "activation", None)
    name = getattr(activation, "__name__", type(output_layer).__name__).lower()
    return name not in ("softmax", "sigmoid")


def resolve_loss(name: str | None, from_logits: bool) -> tf.keras.losses.Loss:
    """Build the Keras loss for a stored loss name.

    Raises ValueError for names outside ``LossFunction`` instead of silently
    compiling with mean squared error.
    """
    loss_cls = _LOSS_CLASSES.get(name)
    if loss_cls is None:
        valid = ", ".join(_LOSS_CLASSES)
        raise ValueError(f"Unsupported loss function: {name!r}. Valid: {valid}")
    if name in _LOGIT_LOSSES:
        return loss_cls(from_logits=from_logits)
    return loss_cls()


def _build_optimizer(optimizer_name: Any, lr: float | None = None) -> Any:
    """Build a Keras optimizer, configuring learning rate if specified.

    Returns the optimizer name string or configured Keras optimizer instance.
    """
    if not isinstance(optimizer_name, str):
        return optimizer_name or "adam"
    opt_name = optimizer_name.lower()
    if lr is not None:
        try:
            return tf.keras.optimizers.get({"class_name": opt_name, "config": {"learning_rate": float(lr)}})
        except Exception:
            opt = tf.keras.optimizers.get(opt_name)
            if hasattr(opt, "learning_rate"):
                with contextlib.suppress(Exception):
                    opt.learning_rate.assign(float(lr))
            return opt
    return opt_name


# Context variable holding the event loop for each concurrent training task,
# so concurrent requests do not overwrite each other's loop reference (Issue #341).
_training_loop_ctx: contextvars.ContextVar[asyncio.AbstractEventLoop | None] = contextvars.ContextVar(
    "_training_loop_ctx", default=None
)


class CustomProgressBar(tf.keras.callbacks.Callback):
    """Keras callback that emits training progress to the frontend via Socket.IO."""

    def __init__(self) -> None:
        super().__init__()
        self._is_training = False

    def on_train_begin(self, logs: dict = None) -> None:
        """Mark start of training so epoch validation does not trigger terminal evaluation emits."""
        self._is_training = True

    def on_train_end(self, logs: dict = None) -> None:
        """Mark end of training so post-training model evaluation is processed."""
        self._is_training = False

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        """Emit the start of a new epoch."""
        _model_result(f"Epoch {epoch + 1}/{self.params['epochs']}", 0)

    def on_batch_end(self, batch: int, logs: dict = None) -> None:
        """Emit batch-level progress with loss and metric values."""
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

        _model_result(
            f"{batch + 1}/{self.params['steps']}  [{arrow}{spaces}] "
            f"{int(progress * 100)}% - Loss: {logs['loss']:.4f} - {metric}",
            1,
        )

    def on_test_begin(self, logs: dict = None) -> None:
        """Emit the start of model evaluation."""
        if self._is_training:
            return
        _model_result("Evaluating...", 2)

    def on_test_end(self, logs: dict = None) -> None:
        """Emit final evaluation metrics."""
        if self._is_training:
            return
        if "mse" in logs:
            metric = f"MSE Loss- {logs['mse']:.4f}"
        elif "accuracy" in logs:
            metric = f"Accuracy- {logs['accuracy']:.4f}"
        else:
            metric = ""
        _model_result(f"Evaluation Results: {metric} Loss - {logs['loss']:.4f}", 3)
        _model_result("Finish", 4)


def _model_result(message: str, test: int) -> None:
    """Emit a Socket.IO event with a training progress message.

    Reads the per-task event loop from a ContextVar so concurrent training
    requests stay isolated.
    """
    data = {"message": message, "test": test}
    loop = _training_loop_ctx.get()
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            sio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE),
            loop,
        )
        try:
            future.result(timeout=5)
        except Exception:
            logger.warning("Failed to emit Socket.IO event: %s", message)
    else:
        logger.warning("No running event loop for Socket.IO emit: %s", message)


def _helper_generate_file_location(db: Session, file_id) -> str:
    """Resolve the on-disk path for a dataset file by its DB ID."""
    upload_folder = get_settings().upload_folder
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if file is None:
        raise ModelRunError(f"Dataset file not found (id={file_id})")
    if file.file_type == "zip":
        return upload_folder + "/" + file.disk_name.rsplit(".", 1)[0]
    return upload_folder + "/" + file.disk_name


def _helper_generate_json_model_file_location(model_name: str) -> str:
    """Construct the path to the model's JSON file, validating against path traversal."""
    path = os.path.realpath(os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE))
    base_dir = os.path.realpath(MODEL_GENERATION_LOCATION)
    if not path.startswith(base_dir + os.sep) and path != base_dir:
        raise ValueError("Invalid model path: escapes model directory")
    return path


def _build_training_callbacks(
    job_id: str | None,
    hyperparams: dict | None = None,
    has_validation: bool = True,
) -> list[tf.keras.callbacks.Callback]:
    """Return the Keras callbacks for a run.

    Without a job_id (legacy/direct calls) we keep the text-based
    ``CustomProgressBar``. With a job_id we use the persistent ``MetricsCallback``
    (DB + room-scoped Socket.IO) plus ``CancellationCheckCallback``.
    When reduce_lr_on_plateau is enabled in hyperparams, attaches
    ``ReduceLROnPlateauCallback`` (Issue #558).
    """
    if job_id is None:
        return [CustomProgressBar()]

    from app.callbacks.lr_schedulers import create_lr_scheduler_callback
    from app.services.training_service import make_session

    loop = _training_loop_ctx.get()
    callbacks: list[tf.keras.callbacks.Callback] = [
        MetricsCallback(job_id, make_session, sio, loop),
        CancellationCheckCallback(job_id, make_session),
    ]

    lr_cb = create_lr_scheduler_callback(
        job_id=job_id,
        hyperparams=hyperparams,
        has_validation=has_validation,
        sio_instance=sio,
        loop=loop,
    )
    if lr_cb is not None:
        callbacks.append(lr_cb)

    return callbacks


def model_run(
    model_name: str,
    db: Session,
    loop: asyncio.AbstractEventLoop | None = None,
    job_id: str | None = None,
) -> None:
    """Load, compile, and train a Keras model, emitting progress via Socket.IO.

    Stores the caller's event loop in a ContextVar so concurrent requests
    each see their own loop instead of racing on a shared global (Issue #341).

    When ``job_id`` is given, training progress is persisted to the
    ``training_metric`` table and the job's lifecycle (running/completed/failed)
    is recorded; on failure the job is marked FAILED with the error message.
    """
    token = _training_loop_ctx.set(loop)
    try:
        _run(model_name, db, job_id=job_id)
    except Exception as e:
        logger.exception("Training failed for model '%s': %s", model_name, str(e))
        if job_id is not None:
            from app.callbacks.metrics_callback import schedule_room_emit
            from app.services.training_service import make_session, update_job_status

            try:
                with make_session() as session:
                    update_job_status(job_id, TrainingStatus.FAILED, session, error_message=str(e))
                schedule_room_emit(
                    sio,
                    loop,
                    job_id,
                    training_event(
                        job_id,
                        "status",
                        status="failed",
                        phase=None,
                        epoch=None,
                        batch=None,
                        steps=None,
                        error=str(e),
                    ),
                )
            except Exception:  # noqa: BLE001 - never mask the original training error
                logger.exception("Failed to mark job %s as failed", job_id)
        else:
            _model_result(f"Training failed: {e}", -1)
        raise
    finally:
        _training_loop_ctx.reset(token)


def _prepare_training_data(features, target_field, training_split):
    """Prepare training data with transparent logging of dropped rows (Issue #5).

    Instead of silently dropping rows with missing values, this function logs
    the count and emits a Socket.IO warning so the user knows their dataset
    was trimmed before training.
    """
    original_rows = len(features)
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    has_inf = (
        np.isinf(features[numeric_cols]).any(axis=1)
        if len(numeric_cols) > 0
        else pd.Series(False, index=features.index)
    )
    has_missing = features.isnull().any(axis=1)
    rows_with_invalid = int((has_missing | has_inf).sum())

    if rows_with_invalid > 0:
        pct = (rows_with_invalid / original_rows) * 100 if original_rows else 0
        msg = (
            f"Warning: Removed {rows_with_invalid} rows ({pct:.1f}%) with missing or non-finite values before training"
        )
        logger.warning(msg)
        _model_result(msg, -1)

    features = features[~(has_missing | has_inf)]
    logger.info("Data preparation: %d -> %d rows retained", original_rows, len(features))

    if target_field not in features.columns:
        raise ValueError(f"Target field '{target_field}' not found in dataset after cleaning")

    # Shuffle to prevent issues with ordered datasets
    features = features.sample(frac=1, random_state=42).reset_index(drop=True)

    X = features.drop(target_field, axis=1)
    y = features[target_field]
    # Encode string labels to integers for categorical targets
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Categorical(y).codes

    split_index = int(len(X) * float(training_split) / 100)
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:]


def _should_skip_evaluation(callbacks: list[tf.keras.callbacks.Callback], model: tf.keras.Model) -> bool:
    """Return True if training was cancelled or stopped prematurely, skipping evaluation."""
    for cb in callbacks:
        if isinstance(cb, CancellationCheckCallback) and (getattr(cb, "cancelled", False) or cb.check_cancelled()):
            return True
    return bool(getattr(model, "stop_training", False))


def _run(model_name: str, db: Session, job_id: str | None = None) -> None:
    model_configs = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if model_configs is None:
        raise ModelRunError(f"Model configuration not found: {model_name}")

    # Resolve effective hyperparameters: job overrides take precedence over saved model configs
    job = db.get(TrainingJob, job_id) if job_id else None
    hyperparams = job.hyperparams if job and job.hyperparams else {}

    # Ensure reproducibility snapshot is captured for jobs starting training (Issue #520)
    if job is not None and job.reproducibility is None:
        try:
            from app.services.reproducibility import capture_reproducibility_snapshot

            job.reproducibility = capture_reproducibility_snapshot(
                session=db,
                model=model_configs,
                hyperparams=hyperparams,
            )
            db.add(job)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not record reproducibility snapshot for job %s: %s", job_id, exc)

    batch_size = hyperparams.get("batch_size") or model_configs.batch_size
    epochs = hyperparams.get("epochs") or model_configs.epochs
    optimizer_name = hyperparams.get("optimizer") or model_configs.optimizer
    lr = hyperparams.get("lr")
    training_split = model_configs.training_split

    # Issue #4: validate training params early for clear error messages
    _validate_training_params(
        batch_size=batch_size,
        epochs=epochs,
        training_split=training_split,
    )

    if model_configs.model_type == ProblemType.IMAGE_CLASSIFICATION:
        image_properties = db.exec(select(ImageProperties).where(ImageProperties.id == model_configs.file_id)).first()
        if image_properties is None:
            raise ModelRunError(f"Image properties not found for file_id={model_configs.file_id}")
        image_size = (image_properties.image_size, image_properties.image_size)
        effective_batch_size = batch_size or image_properties.batch_size or 32
        color_mode = image_properties.color_mode
        label_mode = image_properties.label_mode

        directory = _helper_generate_file_location(db, file_id=model_configs.file_id)
        logger.debug(
            "Image classification params - directory: %s, image_size: %s, "
            "batch_size: %s, color_mode: %s, label_mode: %s",
            directory,
            image_size,
            effective_batch_size,
            color_mode,
            label_mode,
        )
        validation_split = 1 - (float(training_split) / 100)
        train_data = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=effective_batch_size,
            color_mode=color_mode,
            label_mode=label_mode,
        )
        test_data = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=effective_batch_size,
            color_mode=color_mode,
            label_mode=label_mode,
        )
    else:
        file_location = _helper_generate_file_location(db, file_id=model_configs.file_id)
        features = pd.read_csv(file_location)

        # Issue #5: use _prepare_training_data for transparent missing-value handling
        x_training, y_training, x_testing, y_testing = _prepare_training_data(
            features, model_configs.target_field, training_split
        )

        effective_batch_size = batch_size if batch_size is not None else 32

    with open(_helper_generate_json_model_file_location(model_name=model_name)) as f:
        json_string = f.read()
    model = tf.keras.models.model_from_json(json_string, custom_objects=None)
    model.summary()
    optimizer = _build_optimizer(optimizer_name, lr)
    model.compile(
        optimizer=optimizer,
        loss=resolve_loss(model_configs.loss, from_logits=_emits_logits(model)),
        metrics=[model_configs.metric],
    )

    class_weights = None
    if getattr(model_configs, "class_weight_mode", None) == "balanced":
        try:
            if model_configs.model_type == ProblemType.CLASSIFICATION:
                y_arr = y_training.values if hasattr(y_training, "values") else np.array(y_training)
                if y_arr.ndim > 1 and y_arr.shape[1] > 1:
                    y_arr = np.argmax(y_arr, axis=1)
                elif y_arr.ndim > 1:
                    y_arr = y_arr.ravel()
                class_weights = compute_balanced_class_weights(y_arr)
            elif model_configs.model_type == ProblemType.IMAGE_CLASSIFICATION:
                class_dirs = sorted(
                    [
                        d
                        for d in os.listdir(directory)
                        if os.path.isdir(os.path.join(directory, d)) and not d.startswith(".")
                    ]
                )
                if len(class_dirs) > 1:
                    class_counts = []
                    for c_name in class_dirs:
                        c_path = os.path.join(directory, c_name)
                        count = sum(1 for f in os.scandir(c_path) if f.is_file() and not f.name.startswith("."))
                        class_counts.append(count)
                    total_samples = float(sum(class_counts))
                    n_classes = float(len(class_dirs))
                    if total_samples > 0 and n_classes > 1:
                        class_weights = {
                            i: round(float(total_samples / (n_classes * cnt)), 4)
                            for i, cnt in enumerate(class_counts)
                            if cnt > 0
                        }
            if class_weights is None:
                raise ModelRunError(
                    f"Cannot compute balanced class weights for model '{model_name}': "
                    "dataset must contain at least two classes with positive sample counts."
                )
            logger.info("Using balanced class weights for model '%s': %s", model_name, class_weights)
        except ModelRunError:
            raise
        except Exception as e:
            logger.exception("Failed to compute balanced class weights for model '%s'", model_name)
            raise ModelRunError(f"Failed to compute balanced class weights: {e}") from e

    if model_configs.model_type == ProblemType.IMAGE_CLASSIFICATION:
        callbacks = _build_training_callbacks(job_id, hyperparams=hyperparams, has_validation=True)
        model.fit(
            train_data,
            validation_data=test_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0,
        )
        if _should_skip_evaluation(callbacks, model):
            logger.info(
                "Training cancelled or stopped early; skipping evaluation for image model '%s' (job %s)",
                model_name,
                job_id,
            )
        else:
            model.evaluate(test_data, callbacks=callbacks, verbose=0)
    else:
        validation_data = (
            (x_testing, y_testing)
            if (
                x_testing is not None
                and y_testing is not None
                and (not hasattr(x_testing, "__len__") or len(x_testing) > 0)
            )
            else None
        )
        callbacks = _build_training_callbacks(
            job_id,
            hyperparams=hyperparams,
            has_validation=validation_data is not None,
        )
        model.fit(
            x_training,
            y_training,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=effective_batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0,
        )
        if _should_skip_evaluation(callbacks, model):
            logger.info(
                "Training cancelled or stopped early; skipping evaluation for model '%s' (job %s)",
                model_name,
                job_id,
            )
        else:
            if validation_data is not None or not hasattr(x_testing, "__len__") or len(x_testing) > 0:
                model.evaluate(x_testing, y_testing, callbacks=callbacks, verbose=0)
