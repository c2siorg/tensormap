"""Post-training interpretability and analysis service."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sqlmodel import Session, select

from app.models.ml import ModelBasic
from app.services.model_run import (
    _helper_generate_file_location,
    _helper_generate_json_model_file_location,
)
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _resp(status_code: int, success: bool, message: str, data=None):
    return {"success": success, "message": message, "data": data}, status_code


def _load_model_and_data(
    db: Session, model_name: str
) -> tuple[tf.keras.Model, np.ndarray, np.ndarray, ModelBasic] | tuple[None, None, None, None]:
    """Load Keras model + test split from DB config. Returns (model, X_test, y_test, config)."""
    config = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not config:
        return None, None, None, None

    # Load model architecture
    json_path = _helper_generate_json_model_file_location(model_name)
    with open(json_path) as f:
        model = tf.keras.models.model_from_json(f.read())

    # Load and split data (tabular only for now)
    file_location = _helper_generate_file_location(db, file_id=config.file_id)
    features = pd.read_csv(file_location)
    features.dropna(inplace=True)
    features = features.sample(frac=1, random_state=42).reset_index(drop=True)

    X = features.drop(config.target_field, axis=1)
    y = features[config.target_field]
    split_index = int(len(X) * config.training_split / 100)
    X_test = X[split_index:].values.astype(np.float32)
    y_test = y[split_index:].values

    # Compile so predict works
    if config.loss == "sparse_categorical_crossentropy":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=config.optimizer or "adam", loss=loss, metrics=[config.metric or "accuracy"])

    return model, X_test, y_test, config


# ── Confusion matrix ──────────────────────────────────────────────────────────


def confusion_matrix_service(db: Session, model_name: str) -> tuple:
    """Compute confusion matrix for a classification model."""
    model, X_test, y_test, config = _load_model_and_data(db, model_name)
    if model is None:
        return _resp(404, False, "Model not found")
    if config.model_type not in (ProblemType.CLASSIFICATION, ProblemType.IMAGE_CLASSIFICATION):
        return _resp(400, False, "Confusion matrix is only available for classification models")

    try:
        preds = model.predict(X_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(set(y_test.tolist()))
        return _resp(
            200,
            True,
            "Confusion matrix computed",
            {
                "matrix": cm.tolist(),
                "labels": [str(lbl) for lbl in labels],
            },
        )
    except Exception as e:
        logger.exception("Confusion matrix failed: %s", e)
        return _resp(500, False, f"Failed to compute confusion matrix: {e}")


# ── Classification report ─────────────────────────────────────────────────────


def classification_report_service(db: Session, model_name: str) -> tuple:
    """Return per-class precision, recall, f1-score."""
    model, X_test, y_test, config = _load_model_and_data(db, model_name)
    if model is None:
        return _resp(404, False, "Model not found")
    if config.model_type not in (ProblemType.CLASSIFICATION, ProblemType.IMAGE_CLASSIFICATION):
        return _resp(400, False, "Classification report is only available for classification models")

    try:
        preds = model.predict(X_test, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        return _resp(200, True, "Classification report computed", {"report": report})
    except Exception as e:
        logger.exception("Classification report failed: %s", e)
        return _resp(500, False, f"Failed to compute classification report: {e}")


# ── Feature importance (permutation) ─────────────────────────────────────────


def feature_importance_service(db: Session, model_name: str) -> tuple:
    """Compute permutation-based feature importance on the test split."""
    model, X_test, y_test, config = _load_model_and_data(db, model_name)
    if model is None:
        return _resp(404, False, "Model not found")

    try:
        file_location = _helper_generate_file_location(db, file_id=config.file_id)
        features = pd.read_csv(file_location)
        features.dropna(inplace=True)
        feature_names = [c for c in features.columns if c != config.target_field]

        # Baseline loss
        baseline_loss = float(model.evaluate(X_test, y_test, verbose=0)[0])

        importances = []
        for i, name in enumerate(feature_names):
            X_permuted = X_test.copy()
            rng = np.random.default_rng(42)
            X_permuted[:, i] = rng.permutation(X_permuted[:, i])
            permuted_loss = float(model.evaluate(X_permuted, y_test, verbose=0)[0])
            importances.append(
                {
                    "feature": name,
                    "importance": round(permuted_loss - baseline_loss, 6),
                }
            )

        importances.sort(key=lambda x: x["importance"], reverse=True)
        return _resp(
            200,
            True,
            "Feature importance computed",
            {
                "baseline_loss": round(baseline_loss, 6),
                "importances": importances,
            },
        )
    except Exception as e:
        logger.exception("Feature importance failed: %s", e)
        return _resp(500, False, f"Failed to compute feature importance: {e}")


# ── Prediction explorer ───────────────────────────────────────────────────────


def prediction_explorer_service(db: Session, model_name: str, n_samples: int = 20) -> tuple:
    """Return actual vs predicted values for a sample of the test set."""
    model, X_test, y_test, config = _load_model_and_data(db, model_name)
    if model is None:
        return _resp(404, False, "Model not found")

    try:
        preds = model.predict(X_test, verbose=0)
        is_classification = config.model_type in (ProblemType.CLASSIFICATION, ProblemType.IMAGE_CLASSIFICATION)

        n = min(n_samples, len(y_test))
        indices = np.random.default_rng(42).choice(len(y_test), n, replace=False)

        rows = []
        for idx in indices:
            actual = int(y_test[idx]) if is_classification else float(round(float(y_test[idx]), 4))
            if is_classification:
                pred_class = int(np.argmax(preds[idx]))
                confidence = float(round(float(np.max(tf.nn.softmax(preds[idx]).numpy())), 4))
                rows.append(
                    {
                        "index": int(idx),
                        "actual": actual,
                        "predicted": pred_class,
                        "confidence": confidence,
                        "correct": actual == pred_class,
                    }
                )
            else:
                predicted = float(round(float(preds[idx][0]), 4))
                rows.append(
                    {
                        "index": int(idx),
                        "actual": actual,
                        "predicted": predicted,
                        "error": float(round(abs(actual - predicted), 4)),
                    }
                )

        return _resp(
            200,
            True,
            "Predictions computed",
            {
                "samples": rows,
                "is_classification": is_classification,
            },
        )
    except Exception as e:
        logger.exception("Prediction explorer failed: %s", e)
        return _resp(500, False, f"Failed to compute predictions: {e}")
