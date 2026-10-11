"""Post-training interpretability analysis service.

This module provides cached analysis results for trained models including:
- Confusion matrices and classification reports
- Feature importance via permutation
- Regression analysis (residuals, MAE, MSE)

Computations run via asyncio.to_thread to avoid blocking the event loop.
Results are cached in training_job.analysis_cache JSON column.
"""

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sqlmodel import Session

from app.shared.logging_config import get_logger

logger = get_logger(__name__)

EXPORTS_BASE = Path("./exports")


def _resolve_classification_predictions(y_pred_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve predicted class labels and probability distribution from raw model output.

    Handles multi-class models (softmax/logits over K classes, shape (N, K))
    as well as binary classification models with a single output unit (sigmoid/logit,
    shape (N, 1) or (N,)).

    Returns:
        Tuple of (y_pred_classes, y_pred_probs):
        - y_pred_classes: 1D array of predicted class integers (shape (N,))
        - y_pred_probs: 2D array of class probabilities (shape (N, K))
    """
    if y_pred_raw.ndim == 1 or (y_pred_raw.ndim == 2 and y_pred_raw.shape[1] == 1):
        p1 = np.clip(y_pred_raw.flatten(), 0.0, 1.0)
        p0 = 1.0 - p1
        y_pred_probs = np.column_stack([p0, p1])
        y_pred_classes = (p1 >= 0.5).astype(int)
    else:
        y_pred_probs = y_pred_raw
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
    return y_pred_classes, y_pred_probs


def _resolve_ground_truth_classes(y_true: np.ndarray) -> np.ndarray:
    """Resolve ground truth class integer labels from y_true array.

    Handles 1D label vectors (N,), 2D single-column vectors (N, 1),
    and 2D one-hot encoded label matrices (N, K).
    """
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        return y_true.flatten().astype(int)
    return np.argmax(y_true, axis=1).astype(int)


class AnalysisCache:
    """Manages caching of analysis results in training_job.analysis_cache JSON column.

    The analysis_cache column stores results keyed by analysis type:
    {
        "confusion_matrix": {...},
        "feature_importance": {...},
        "predictions": {...}
    }
    """

    def get_cached(self, job_id: str, analysis_type: str, session: Session) -> dict | None:
        """Returns cached analysis result or None if not cached.

        Args:
            job_id: Training job ID
            analysis_type: Type of analysis (confusion_matrix, feature_importance, predictions)
            session: Database session

        Returns:
            Cached result dict or None if not found
        """
        from app.models.training_job import TrainingJob

        job = session.get(TrainingJob, job_id)
        if not job or not job.analysis_cache:
            return None

        return job.analysis_cache.get(analysis_type)

    def set_cached(self, job_id: str, analysis_type: str, result: dict, session: Session) -> None:
        """Stores analysis result in training_job.analysis_cache.

        Args:
            job_id: Training job ID
            analysis_type: Type of analysis (confusion_matrix, feature_importance, predictions)
            result: Analysis result to cache
            session: Database session
        """
        from app.models.training_job import TrainingJob

        job = session.get(TrainingJob, job_id)
        if not job:
            logger.warning(f"Cannot cache analysis for non-existent job {job_id}")
            return

        if job.analysis_cache is None:
            job.analysis_cache = {}

        # Update cache
        job.analysis_cache[analysis_type] = result

        # Mark as modified for SQLAlchemy to detect the change
        from sqlalchemy.orm import attributes

        attributes.flag_modified(job, "analysis_cache")

        session.add(job)
        session.commit()
        logger.info(f"Cached {analysis_type} analysis for job {job_id}")


class InterpretabilityService:
    """Computes interpretability analyses for trained models.

    Supports classification (confusion matrix, classification report),
    regression (residuals, MAE, MSE), and feature importance (permutation).
    All heavy computations run in thread pools via asyncio.to_thread.
    """

    def __init__(self):
        self.cache = AnalysisCache()

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_test_data(self, job_id: str, session: Session) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Loads test split of the dataset used for this training job.

        Finds data_file via: training_job → model_basic → data_file
        Applies same preprocessing that was used during training
        (shuffle with random_state=42, split at training_split / 100).

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Tuple of (X_test, y_test, class_names)
        """
        from app.config import get_settings
        from app.models.data import DataFile
        from app.models.ml import ModelBasic
        from app.models.training_job import TrainingJob

        job = session.get(TrainingJob, job_id)
        if not job:
            raise ValueError(f"Training job not found: {job_id}")

        model = session.get(ModelBasic, job.model_id)
        if not model:
            raise ValueError(f"Model not found for job: {job_id}")

        if not model.file_id:
            raise ValueError(f"No dataset linked to model {model.model_name}")

        data_file = session.get(DataFile, model.file_id)
        if not data_file:
            raise ValueError(f"Data file not found for model {model.model_name}")

        # Build file path (same logic as model_run._helper_generate_file_location)
        upload_folder = get_settings().upload_folder
        file_path = f"{upload_folder}/{data_file.disk_name}"

        features = pd.read_csv(file_path)

        target_field = model.target_field
        if target_field not in features.columns:
            raise ValueError(f"Target field '{target_field}' not found in dataset")

        # Replicate exactly the same preprocessing as model_run._prepare_training_data
        features = features.dropna()
        features = features.sample(frac=1, random_state=42).reset_index(drop=True)

        X = features.drop(target_field, axis=1)
        y = features[target_field]

        # Track original class names before encoding
        if not pd.api.types.is_numeric_dtype(y):
            class_names = list(pd.Categorical(y).categories)
            y = pd.Categorical(y).codes
        else:
            class_names = [str(c) for c in sorted(y.unique())]

        training_split = model.training_split if model.training_split else 80
        split_index = int(len(X) * float(training_split) / 100)

        X_test = X[split_index:].values.astype(np.float32)
        y_test = y[split_index:].values if hasattr(y, "values") else y[split_index:]

        return X_test, y_test, class_names

    def _load_test_data_with_features(self, job_id: str, session: Session) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Loads test split with feature names instead of class names.

        Same data loading as _load_test_data but returns feature column names
        for use with permutation feature importance.

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Tuple of (X_test, y_test, feature_names)
        """
        from app.config import get_settings
        from app.models.data import DataFile
        from app.models.ml import ModelBasic
        from app.models.training_job import TrainingJob

        job = session.get(TrainingJob, job_id)
        if not job:
            raise ValueError(f"Training job not found: {job_id}")

        model = session.get(ModelBasic, job.model_id)
        if not model:
            raise ValueError(f"Model not found for job: {job_id}")

        if not model.file_id:
            raise ValueError(f"No dataset linked to model {model.model_name}")

        data_file = session.get(DataFile, model.file_id)
        if not data_file:
            raise ValueError(f"Data file not found for model {model.model_name}")

        upload_folder = get_settings().upload_folder
        file_path = f"{upload_folder}/{data_file.disk_name}"

        features = pd.read_csv(file_path)

        target_field = model.target_field
        if target_field not in features.columns:
            raise ValueError(f"Target field '{target_field}' not found in dataset")

        features = features.dropna()
        features = features.sample(frac=1, random_state=42).reset_index(drop=True)

        X = features.drop(target_field, axis=1)
        y = features[target_field]

        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Categorical(y).codes

        feature_names = list(X.columns)

        training_split = model.training_split if model.training_split else 80
        split_index = int(len(X) * float(training_split) / 100)

        X_test = X[split_index:].values.astype(np.float32)
        y_test = y[split_index:].values if hasattr(y, "values") else y[split_index:]

        return X_test, y_test, feature_names

    def _load_test_data_complete(
        self, job_id: str, session: Session
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Loads test split with both class names and feature names in a single pass.

        Avoids duplicate CSV loading and preprocessing by returning all metadata
        at once. Use this when you need both class_names and feature_names.

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Tuple of (X_test, y_test, class_names, feature_names)
        """
        from app.config import get_settings
        from app.models.data import DataFile
        from app.models.ml import ModelBasic
        from app.models.training_job import TrainingJob

        job = session.get(TrainingJob, job_id)
        if not job:
            raise ValueError(f"Training job not found: {job_id}")

        model = session.get(ModelBasic, job.model_id)
        if not model:
            raise ValueError(f"Model not found for job: {job_id}")

        if not model.file_id:
            raise ValueError(f"No dataset linked to model {model.model_name}")

        data_file = session.get(DataFile, model.file_id)
        if not data_file:
            raise ValueError(f"Data file not found for model {model.model_name}")

        upload_folder = get_settings().upload_folder
        file_path = f"{upload_folder}/{data_file.disk_name}"

        features = pd.read_csv(file_path)

        target_field = model.target_field
        if target_field not in features.columns:
            raise ValueError(f"Target field '{target_field}' not found in dataset")

        features = features.dropna()
        features = features.sample(frac=1, random_state=42).reset_index(drop=True)

        X = features.drop(target_field, axis=1)
        y = features[target_field]

        # Track original class names before encoding
        if not pd.api.types.is_numeric_dtype(y):
            class_names = list(pd.Categorical(y).categories)
            y = pd.Categorical(y).codes
        else:
            class_names = [str(c) for c in sorted(y.unique())]

        feature_names = list(X.columns)

        training_split = model.training_split if model.training_split else 80
        split_index = int(len(X) * float(training_split) / 100)

        X_test = X[split_index:].values.astype(np.float32)
        y_test = y[split_index:].values if hasattr(y, "values") else y[split_index:]

        return X_test, y_test, class_names, feature_names

    # ------------------------------------------------------------------
    # Confusion matrix + classification report
    # ------------------------------------------------------------------

    async def compute_confusion_matrix_async(self, job_id: str, session: Session) -> dict:
        """Async wrapper — runs blocking sklearn computation in thread pool.

        Returns cached result if available.

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Dictionary with confusion matrix and classification metrics
        """
        cache = self.cache.get_cached(job_id, "confusion_matrix", session)
        if cache:
            return cache

        result = await asyncio.to_thread(self._compute_confusion_matrix_sync, job_id, session)
        self.cache.set_cached(job_id, "confusion_matrix", result, session)
        return result

    def _compute_confusion_matrix_sync(self, job_id: str, session: Session) -> dict:
        """Compute confusion matrix and classification report (blocking).

        Loads the trained model from exports, runs predictions on the test split,
        and computes sklearn metrics.

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Dictionary with confusion_matrix, classification_report, class_names,
            overall_accuracy, n_samples, and analysis_type
        """
        import tensorflow as tf

        model_path = EXPORTS_BASE / job_id / "model.keras"
        model = tf.keras.models.load_model(model_path)

        X_test, y_test, class_names = self._load_test_data(job_id, session)

        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes, _ = _resolve_classification_predictions(y_pred)
        y_true_classes = _resolve_ground_truth_classes(y_test)

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        report = classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "class_names": class_names,
            "overall_accuracy": float(report["accuracy"]),
            "n_samples": len(y_true_classes),
            "analysis_type": "classification",
        }

    # ------------------------------------------------------------------
    # Regression analysis
    # ------------------------------------------------------------------

    async def compute_regression_analysis_async(self, job_id: str, session: Session) -> dict:
        """Async wrapper for regression analysis.

        Returns cached result if available.

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Dictionary with residuals, predictions, actuals, MAE, MSE
        """
        cache = self.cache.get_cached(job_id, "regression_analysis", session)
        if cache:
            return cache

        result = await asyncio.to_thread(self._compute_regression_analysis_sync, job_id, session)
        self.cache.set_cached(job_id, "regression_analysis", result, session)
        return result

    def _compute_regression_analysis_sync(self, job_id: str, session: Session) -> dict:
        """Compute regression analysis metrics (blocking).

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Dictionary with residuals, y_pred, y_true, mae, mse, analysis_type
        """
        import tensorflow as tf

        model_path = EXPORTS_BASE / job_id / "model.keras"
        model = tf.keras.models.load_model(model_path)

        X_test, y_test, _ = self._load_test_data(job_id, session)

        raw_pred = model.predict(X_test, verbose=0)
        # For regression models the output is (n_samples, 1) → flatten to 1D.
        # Guard against classification models used in regression context.
        if raw_pred.ndim == 2 and raw_pred.shape[1] == 1:
            y_pred = raw_pred.flatten()
        elif raw_pred.ndim == 2:
            # Multi-output: use first column as regression output
            y_pred = raw_pred[:, 0]
        else:
            y_pred = raw_pred.flatten()

        y_true = y_test.astype(float)
        # Ensure shapes match
        y_pred = y_pred[: len(y_true)]

        residuals = (y_pred - y_true).tolist()
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))

        return {
            "residuals": residuals,
            "y_pred": y_pred.tolist(),
            "y_true": y_true.tolist(),
            "mae": mae,
            "mse": mse,
            "analysis_type": "regression",
        }

    # ------------------------------------------------------------------
    # Permutation feature importance (202 polling pattern)
    # ------------------------------------------------------------------

    async def compute_feature_importance_async(self, job_id: str, session: Session) -> dict | None:
        """Returns None if computation is in progress (202 case).

        Returns dict if cached.
        Fires background asyncio.to_thread computation if not started.

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Feature importance dict if cached/computed, None if still computing
        """
        cache = self.cache.get_cached(job_id, "feature_importance", session)
        if cache:
            return cache

        # Check if already computing (use a "computing" sentinel in cache)
        sentinel = self.cache.get_cached(job_id, "feature_importance_status", session)
        if sentinel and sentinel.get("status") == "computing":
            return None  # Still computing → 202

        # Mark as computing
        self.cache.set_cached(job_id, "feature_importance_status", {"status": "computing"}, session)

        # Fire background computation
        asyncio.create_task(self._compute_and_cache_feature_importance(job_id, session))
        return None  # Will be 202

    async def _compute_and_cache_feature_importance(self, job_id: str, session: Session) -> None:
        """Background task that computes feature importance and caches the result.

        Args:
            job_id: Training job ID
            session: Database session
        """
        try:
            result = await asyncio.to_thread(self._compute_feature_importance_sync, job_id, session)
            self.cache.set_cached(job_id, "feature_importance", result, session)
            # Clear the computing sentinel
            self.cache.set_cached(job_id, "feature_importance_status", {"status": "completed"}, session)
            logger.info(f"Feature importance computation completed for job {job_id}")
        except Exception:
            logger.exception(f"Feature importance computation failed for job {job_id}")
            self.cache.set_cached(job_id, "feature_importance_status", {"status": "failed"}, session)

    def _compute_feature_importance_sync(self, job_id: str, session: Session) -> dict:
        """Compute permutation feature importance (blocking).

        Guardrails:
        - Sample capped at min(1000, len(X_test))
        - Feature count capped at 20 (top by variance)

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Dictionary with features, importances_mean, importances_std,
            n_samples_used, n_repeats, analysis_type
        """
        import tensorflow as tf
        from sklearn.base import BaseEstimator, ClassifierMixin
        from sklearn.inspection import permutation_importance

        model_path = EXPORTS_BASE / job_id / "model.keras"
        keras_model = tf.keras.models.load_model(model_path)

        X_test, y_test, feature_names = self._load_test_data_with_features(job_id, session)

        # Guardrails
        MAX_SAMPLES = 1000
        MAX_FEATURES = 20
        n_samples = min(MAX_SAMPLES, len(X_test))
        X_sample = X_test[:n_samples]
        y_sample = _resolve_ground_truth_classes(y_test[:n_samples])

        if X_sample.shape[1] > MAX_FEATURES:
            # Select top 20 features by variance
            variances = np.var(X_sample, axis=0)
            top_indices = np.argsort(variances)[-MAX_FEATURES:]
            top_indices = np.sort(top_indices)  # Keep original order
            X_sample = X_sample[:, top_indices]
            feature_names = [feature_names[i] for i in top_indices]

        # Wrap Keras model in a sklearn-compatible estimator so
        # permutation_importance can call .predict() and .score()
        class _KerasEstimatorWrapper(BaseEstimator, ClassifierMixin):
            """Minimal sklearn wrapper around a trained Keras model."""

            def __init__(self, model):
                self.model = model
                self.classes_ = np.unique(y_sample)

            def fit(self, X, y=None):
                return self  # Already trained

            def predict(self, X):
                preds = self.model.predict(X, verbose=0)
                pred_classes, _ = _resolve_classification_predictions(preds)
                return pred_classes

        estimator = _KerasEstimatorWrapper(keras_model)

        result = permutation_importance(
            estimator,
            X_sample,
            y_sample,
            n_repeats=10,
            random_state=42,
            scoring="accuracy",
        )

        return {
            "features": feature_names,
            "importances_mean": result.importances_mean.tolist(),
            "importances_std": result.importances_std.tolist(),
            "analysis_type": "feature_importance",
            "n_samples_used": n_samples,
            "n_repeats": 10,
        }

    # ------------------------------------------------------------------
    # Predictions explorer (Week 10)
    # ------------------------------------------------------------------

    async def get_predictions_async(
        self, job_id: str, offset: int, limit: int, filter_correct: bool | None, session: Session
    ) -> dict:
        """Returns paginated predictions for the prediction explorer.

        Computes all predictions once and caches them. Subsequent calls
        serve from cache. This avoids running model.predict() on every
        page request.

        Args:
            job_id: Training job ID
            offset: Pagination offset (0-indexed)
            limit: Number of predictions to return per page
            filter_correct: Filter by correctness
                - True: only correct predictions
                - False: only incorrect predictions
                - None: all predictions
            session: Database session

        Returns:
            {
                "total": 150,
                "offset": 0,
                "limit": 25,
                "predictions": [
                    {
                        "index": 0,
                        "actual_class": 0,
                        "actual_class_name": "Setosa",
                        "predicted_class": 0,
                        "predicted_class_name": "Setosa",
                        "confidence": 0.97,
                        "probabilities": [0.97, 0.02, 0.01],
                        "features": {"sepal_length": 5.1, "sepal_width": 3.5, ...},
                        "is_correct": true
                    },
                    ...
                ]
            }
        """
        # Check cache first
        cached = self.cache.get_cached(job_id, "predictions", session)
        if not cached:
            # Compute all predictions and cache (offload to thread pool)
            cached = await asyncio.to_thread(self._compute_all_predictions_sync, job_id, session)
            self.cache.set_cached(job_id, "predictions", cached, session)

        # Filter
        all_predictions = cached["predictions"]
        if filter_correct is True:
            filtered = [p for p in all_predictions if p["is_correct"]]
        elif filter_correct is False:
            filtered = [p for p in all_predictions if not p["is_correct"]]
        else:
            filtered = all_predictions

        # Paginate
        paginated = filtered[offset : offset + limit]

        return {
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "predictions": paginated,
        }

    def _compute_all_predictions_sync(self, job_id: str, session: Session) -> dict:
        """Compute predictions for all test samples.

        Args:
            job_id: Training job ID
            session: Database session

        Returns:
            Dictionary with all predictions (not paginated)
        """
        import tensorflow as tf

        model_path = EXPORTS_BASE / job_id / "model.keras"
        model = tf.keras.models.load_model(model_path)

        # Load test data with both class names and feature names in a single pass
        X_test, y_test, class_names, feature_names = self._load_test_data_complete(job_id, session)

        # Run predictions
        y_pred_raw = model.predict(X_test, verbose=0)
        y_pred_classes, y_pred_probs = _resolve_classification_predictions(y_pred_raw)
        y_true_classes = _resolve_ground_truth_classes(y_test)

        # Build prediction records
        predictions = []
        for i in range(len(X_test)):
            actual_class = int(y_true_classes[i])
            predicted_class = int(y_pred_classes[i])
            probabilities = y_pred_probs[i].tolist()
            confidence = float(max(probabilities))

            # Build feature dict
            features = {feature_names[j]: float(X_test[i][j]) for j in range(len(feature_names))}

            predictions.append(
                {
                    "index": i,
                    "actual_class": actual_class,
                    "actual_class_name": class_names[actual_class]
                    if actual_class < len(class_names)
                    else str(actual_class),
                    "predicted_class": predicted_class,
                    "predicted_class_name": class_names[predicted_class]
                    if predicted_class < len(class_names)
                    else str(predicted_class),
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "features": features,
                    "is_correct": actual_class == predicted_class,
                }
            )

        # Sort by confidence DESC (most confident first)
        predictions.sort(key=lambda p: p["confidence"], reverse=True)

        return {"predictions": predictions}
