"""Holdout test dataset evaluation and batch prediction service.

Enables benchmarking trained models against unseen holdout datasets, computing
full evaluation scorecards (Accuracy, Precision, Recall, F1, Log Loss for
classification; MSE, RMSE, MAE, R², MAPE for regression), diagnosing top error
samples, and generating batch predictions with downloadable CSV exports.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sqlmodel import Session

from app.config import get_settings
from app.exceptions import AppException
from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.schemas.evaluation import (
    BatchEvaluationResponse,
    BatchPredictionResponse,
    TopErrorSample,
)
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

EXPORTS_BASE = Path("./exports")


def _resolve_classification_predictions(y_pred_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve predicted class labels and probability distribution from raw model output.

    Handles multi-class models (softmax/logits over K classes, shape (N, K))
    as well as binary classification models with a single output unit (sigmoid/logit,
    shape (N, 1) or (N,)).
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
    """Resolve ground truth class integer labels from y_true array."""
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        return y_true.flatten().astype(int)
    return np.argmax(y_true, axis=1).astype(int)


class EvaluationService:
    """Service providing dataset evaluation, error analysis, and batch predictions."""

    def _get_job_and_model(self, job_id: str, session: Session) -> tuple[TrainingJob, ModelBasic, DataFile]:
        """Fetch and validate job, model, and training data file from database."""
        job = session.get(TrainingJob, job_id)
        if not job:
            raise AppException(404, "Training job not found")

        if job.status != TrainingStatus.COMPLETED:
            raise AppException(
                400,
                f"Evaluation is only available for completed jobs. Current status: {job.status.value}",
            )

        model = session.get(ModelBasic, job.model_id)
        if not model:
            raise AppException(404, "Model not found for this training job")

        if not model.file_id:
            raise AppException(404, "No training dataset associated with this model")

        data_file = session.get(DataFile, model.file_id)
        if not data_file:
            raise AppException(404, "Training dataset file record not found")

        return job, model, data_file

    def _resolve_problem_type_str(self, model_type: int | None) -> str:
        """Map integer ProblemType enum to canonical string."""
        if model_type == ProblemType.REGRESSION:
            return "regression"
        if model_type == ProblemType.IMAGE_CLASSIFICATION:
            return "image_classification"
        return "classification"

    def _load_model(self, job_id: str):
        """Loads trained Keras model from disk."""
        model_path = EXPORTS_BASE / job_id / "model.keras"
        if not model_path.exists():
            raise AppException(404, f"Trained model artifact not found for job {job_id}")

        import tensorflow as tf

        try:
            return tf.keras.models.load_model(model_path)
        except Exception as err:
            logger.exception("Failed to load model from %s", model_path)
            raise AppException(500, f"Failed to load trained model: {err}") from err

    def _get_training_feature_names_and_classes(
        self, training_data_file: DataFile, target_field: str | None
    ) -> tuple[list[str], list[str]]:
        """Inspects the training dataset to extract expected feature columns and class names."""
        upload_folder = get_settings().upload_folder
        file_path = Path(upload_folder) / training_data_file.disk_name
        if not file_path.exists():
            raise AppException(404, f"Training dataset file not found at {file_path}")

        try:
            df = pd.read_csv(file_path).dropna()
        except pd.errors.EmptyDataError as err:
            raise AppException(400, "Training dataset is empty") from err
        except Exception as err:
            raise AppException(400, f"Error reading training dataset: {err}") from err

        if df.empty:
            raise AppException(400, "Training dataset is empty")

        non_target_cols = [c for c in df.columns if c != target_field]
        feature_names: list[str] = []
        for col in non_target_cols:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.dropna().shape[0] > 0:
                feature_names.append(col)

        class_names: list[str] = []
        if target_field and target_field in df.columns:
            target_series = df[target_field]
            if not pd.api.types.is_numeric_dtype(target_series):
                class_names = [str(c) for c in pd.Categorical(target_series).categories]
            else:
                unique_vals = sorted(target_series.unique())
                class_names = [str(int(c)) if isinstance(c, float) and c.is_integer() else str(c) for c in unique_vals]

        return feature_names, class_names

    def _load_dataset_file(self, file_id: UUID, session: Session) -> tuple[DataFile, pd.DataFrame]:
        """Loads and returns dataset file record and DataFrame."""
        data_file = session.get(DataFile, file_id)
        if not data_file:
            raise AppException(404, f"Dataset file {file_id} not found")

        upload_folder = get_settings().upload_folder
        file_path = Path(upload_folder) / data_file.disk_name
        if not file_path.exists():
            raise AppException(404, f"Dataset file not found on disk at {file_path}")

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError as err:
            raise AppException(400, "Dataset file is empty") from err
        except Exception as err:
            raise AppException(400, f"Failed to parse CSV dataset: {err}") from err

        if df.empty:
            raise AppException(400, "Dataset file is empty")

        return data_file, df

    def evaluate_holdout_dataset(
        self,
        job_id: str,
        file_id: UUID,
        target_field: str | None,
        batch_size: int,
        session: Session,
    ) -> BatchEvaluationResponse:
        """Evaluates a trained model against an uploaded holdout dataset."""
        _, model, training_data_file = self._get_job_and_model(job_id, session)
        problem_type = self._resolve_problem_type_str(model.model_type)

        if problem_type == "image_classification":
            raise AppException(400, "Holdout evaluation is currently supported for tabular models.")

        effective_target = target_field or model.target_field
        feature_names, train_class_names = self._get_training_feature_names_and_classes(
            training_data_file, model.target_field
        )

        _, test_df = self._load_dataset_file(file_id, session)

        if not effective_target or effective_target not in test_df.columns:
            raise AppException(
                400,
                f"Target column '{effective_target}' not found in the evaluation dataset.",
            )

        missing_features = [col for col in feature_names if col not in test_df.columns]
        if missing_features:
            raise AppException(
                400,
                f"Evaluation dataset is missing required feature columns: {', '.join(sorted(missing_features))}",
            )

        # Drop rows with NaNs in required columns
        valid_cols = feature_names + [effective_target]
        clean_df = test_df.dropna(subset=valid_cols).copy()
        if clean_df.empty:
            raise AppException(
                400,
                "No valid rows remaining in evaluation dataset after removing missing values.",
            )

        # Vectorize features
        try:
            X_numeric = clean_df[feature_names].apply(pd.to_numeric, errors="coerce")
            if X_numeric.isna().any().any():
                clean_df = clean_df.loc[~X_numeric.isna().any(axis=1)]
                X_numeric = X_numeric.dropna()
            X_mat = X_numeric.values.astype(np.float32)
        except Exception as err:
            raise AppException(400, f"Failed to convert feature matrix to numeric values: {err}") from err

        if len(X_mat) == 0:
            raise AppException(400, "Evaluation dataset contains no valid numeric rows for features.")

        keras_model = self._load_model(job_id)
        raw_preds = keras_model.predict(X_mat, batch_size=batch_size, verbose=0)

        raw_targets = clean_df[effective_target].values
        top_errors: list[TopErrorSample] = []
        metrics: dict[str, Any] = {}

        if problem_type == "classification":
            pred_classes, pred_probs = _resolve_classification_predictions(raw_preds)

            # Determine class mapping
            class_names = train_class_names
            if not class_names:
                unique_targets = sorted(clean_df[effective_target].unique())
                class_names = [str(u) for u in unique_targets]

            # Encode actual classes
            if not pd.api.types.is_numeric_dtype(clean_df[effective_target]):
                cat_targets = pd.Categorical(clean_df[effective_target], categories=class_names)
                y_true_classes = cat_targets.codes
            else:
                try:
                    y_true_classes = np.array(
                        [
                            class_names.index(str(int(v)))
                            if isinstance(v, (int, float)) and float(v).is_integer()
                            else class_names.index(str(v))
                            for v in raw_targets
                        ]
                    )
                except ValueError:
                    # Fallback to direct numeric conversion
                    y_true_classes = _resolve_ground_truth_classes(raw_targets)

            # Drop any invalid target rows (-1 from unknown category)
            valid_mask = y_true_classes >= 0
            if not np.all(valid_mask):
                y_true_classes = y_true_classes[valid_mask]
                pred_classes = pred_classes[valid_mask]
                pred_probs = pred_probs[valid_mask]
                clean_df = clean_df.iloc[valid_mask]

            acc = float(accuracy_score(y_true_classes, pred_classes))
            prec = float(precision_score(y_true_classes, pred_classes, average="macro", zero_division=0))
            rec = float(recall_score(y_true_classes, pred_classes, average="macro", zero_division=0))
            f1_macro = float(f1_score(y_true_classes, pred_classes, average="macro", zero_division=0))
            f1_weighted = float(f1_score(y_true_classes, pred_classes, average="weighted", zero_division=0))

            loss_val: float | None = None
            try:
                if len(class_names) == pred_probs.shape[1] and len(np.unique(y_true_classes)) > 1:
                    loss_val = float(round(log_loss(y_true_classes, pred_probs), 4))
            except Exception:
                loss_val = None

            cm = confusion_matrix(y_true_classes, pred_classes).tolist()

            metrics = {
                "accuracy": round(acc, 4),
                "precision_macro": round(prec, 4),
                "recall_macro": round(rec, 4),
                "f1_macro": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4),
                "log_loss": loss_val,
                "confusion_matrix": cm,
                "class_names": class_names,
            }

            # Top misclassification errors (highest confidence on wrong class)
            error_indices = np.where(y_true_classes != pred_classes)[0]
            if len(error_indices) > 0:
                error_confidences = [float(np.max(pred_probs[i])) for i in error_indices]
                sorted_error_order = np.argsort(error_confidences)[::-1][:10]

                for order_idx in sorted_error_order:
                    orig_idx = int(error_indices[order_idx])
                    row_data = clean_df.iloc[orig_idx]
                    feat_dict = {f: float(row_data[f]) for f in feature_names if f in row_data}
                    actual_cls = int(y_true_classes[orig_idx])
                    pred_cls = int(pred_classes[orig_idx])
                    act_name = class_names[actual_cls] if actual_cls < len(class_names) else str(actual_cls)
                    prd_name = class_names[pred_cls] if pred_cls < len(class_names) else str(pred_cls)
                    conf = float(np.max(pred_probs[orig_idx]))

                    top_errors.append(
                        TopErrorSample(
                            index=orig_idx,
                            features=feat_dict,
                            actual_label=act_name,
                            predicted_label=prd_name,
                            confidence=round(conf, 4),
                            loss_or_residual=round(conf, 4),
                        )
                    )

        else:
            # Regression evaluation
            y_pred = raw_preds.flatten().astype(float)
            y_true = clean_df[effective_target].values.astype(float)

            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true, y_pred))

            r2_val: float | None = None
            if len(y_true) > 1 and np.var(y_true) > 0:
                r2_val = float(round(r2_score(y_true, y_pred), 4))

            # Mean Absolute Percentage Error (avoid division by 0)
            non_zero_mask = np.abs(y_true) > 1e-6
            mape_val: float | None = None
            if np.any(non_zero_mask):
                mape_val = float(
                    round(
                        np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))
                        * 100.0,
                        2,
                    )
                )

            max_err = float(np.max(np.abs(y_true - y_pred)))

            metrics = {
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": r2_val,
                "mape": mape_val,
                "max_error": round(max_err, 4),
            }

            # Top residual errors
            residuals = np.abs(y_true - y_pred)
            sorted_residual_indices = np.argsort(residuals)[::-1][:10]

            for orig_idx in sorted_residual_indices:
                row_data = clean_df.iloc[orig_idx]
                feat_dict = {f: float(row_data[f]) for f in feature_names if f in row_data}
                act_val = float(y_true[orig_idx])
                prd_val = float(y_pred[orig_idx])
                res_val = float(residuals[orig_idx])

                top_errors.append(
                    TopErrorSample(
                        index=int(orig_idx),
                        features=feat_dict,
                        actual_label=str(round(act_val, 4)),
                        predicted_label=str(round(prd_val, 4)),
                        confidence=None,
                        loss_or_residual=round(res_val, 4),
                    )
                )

        return BatchEvaluationResponse(
            job_id=job_id,
            file_id=str(file_id),
            problem_type=problem_type,
            total_samples=len(clean_df),
            evaluated_at=datetime.now(UTC).isoformat(),
            metrics=metrics,
            top_errors=top_errors,
        )

    def run_batch_prediction(
        self,
        job_id: str,
        file_id: UUID,
        batch_size: int,
        session: Session,
    ) -> tuple[pd.DataFrame, BatchPredictionResponse]:
        """Runs batch inference across dataset rows and returns scored DataFrame and response."""
        _, model, training_data_file = self._get_job_and_model(job_id, session)
        problem_type = self._resolve_problem_type_str(model.model_type)

        if problem_type == "image_classification":
            raise AppException(400, "Batch prediction is currently supported for tabular models.")

        feature_names, train_class_names = self._get_training_feature_names_and_classes(
            training_data_file, model.target_field
        )

        _, df = self._load_dataset_file(file_id, session)

        missing_features = [col for col in feature_names if col not in df.columns]
        if missing_features:
            raise AppException(
                400,
                f"Prediction dataset is missing required feature columns: {', '.join(sorted(missing_features))}",
            )

        scored_df = df.copy()
        X_numeric = scored_df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_mat = X_numeric.values.astype(np.float32)

        keras_model = self._load_model(job_id)
        raw_preds = keras_model.predict(X_mat, batch_size=batch_size, verbose=0)

        if problem_type == "classification":
            pred_classes, pred_probs = _resolve_classification_predictions(raw_preds)
            class_names = train_class_names or [f"Class {i}" for i in range(pred_probs.shape[1])]

            pred_labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c in pred_classes]
            confidences = [round(float(np.max(p)), 4) for p in pred_probs]

            scored_df["predicted_class"] = pred_labels
            scored_df["confidence"] = confidences

            for class_idx, name in enumerate(class_names):
                if class_idx < pred_probs.shape[1]:
                    scored_df[f"prob_{name}"] = [round(float(p[class_idx]), 4) for p in pred_probs]
        else:
            y_pred = [round(float(val), 4) for val in raw_preds.flatten()]
            scored_df["predicted_value"] = y_pred

        preview_records = scored_df.head(20).to_dict(orient="records")
        # Sanitize NaN/Inf for JSON
        for record in preview_records:
            for k, v in record.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    record[k] = None

        response = BatchPredictionResponse(
            job_id=job_id,
            file_id=str(file_id),
            problem_type=problem_type,
            total_rows=len(scored_df),
            columns=list(scored_df.columns),
            preview_rows=preview_records,
        )

        return scored_df, response

    def export_batch_predictions_csv(
        self,
        job_id: str,
        file_id: UUID,
        batch_size: int,
        session: Session,
    ) -> str:
        """Runs batch prediction and exports the full scored DataFrame as a CSV string."""
        scored_df, _ = self.run_batch_prediction(job_id, file_id, batch_size, session)
        return scored_df.to_csv(index=False)
