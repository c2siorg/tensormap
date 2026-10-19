"""Service for interactive real-time model inference and schema inspection.

Features:
- Thread-safe LRU model memory cache to avoid expensive disk reloading
- Feature schema extraction (bounds, distributions, steps) to drive UI sliders
- Sample test dataset record generator with ground truth targets
- Microsecond/low-latency model prediction for classification and regression
"""

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.config import get_settings
from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.schemas.inference import (
    ClassProbability,
    FeatureSchemaItem,
    InferenceResponse,
    ModelFeatureSchemaResponse,
    SampleTestRecord,
)
from app.services.interpretability import (
    EXPORTS_BASE,
    _resolve_classification_predictions,
)
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


class ModelInferenceCache:
    """Thread-safe LRU in-memory cache for loaded Keras models.

    Prevents reloading models from disk on every single slider change or inference request,
    reducing inference latency from ~1-2 seconds down to 5-15 milliseconds.
    """

    def __init__(self, maxsize: int = 5):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get_or_load(self, job_id: str, model_path: Path) -> Any:
        with self._lock:
            if job_id in self._cache:
                self._cache.move_to_end(job_id)
                return self._cache[job_id]

        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")

        # Lazy import TensorFlow
        import tensorflow as tf

        logger.info(f"Loading model into inference cache for job {job_id}")
        model = tf.keras.models.load_model(model_path)

        with self._lock:
            if job_id in self._cache:
                return self._cache[job_id]
            if len(self._cache) >= self._maxsize:
                evicted_job_id, _ = self._cache.popitem(last=False)
                logger.info(f"Evicted model {evicted_job_id} from inference cache")
            self._cache[job_id] = model
            return model

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class InferenceService:
    """Provides feature schema inspection and live inference execution."""

    def __init__(self):
        self.model_cache = ModelInferenceCache(maxsize=5)

    def _get_job_and_model(self, job_id: str, session: Session) -> tuple[TrainingJob, ModelBasic, DataFile]:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise ValueError(f"Training job not found: {job_id}")

        if job.status != TrainingStatus.COMPLETED:
            raise ValueError(f"Inference is only available for completed jobs (current status: {job.status})")

        model = session.get(ModelBasic, job.model_id)
        if not model:
            raise ValueError(f"Model not found for job: {job_id}")

        if not model.file_id:
            raise ValueError(f"No dataset linked to model {model.model_name}")

        data_file = session.get(DataFile, model.file_id)
        if not data_file:
            raise ValueError(f"Data file not found for model {model.model_name}")

        return job, model, data_file

    def _resolve_problem_type_str(self, model_type: int | None) -> str:
        if model_type == ProblemType.REGRESSION:
            return "regression"
        if model_type == ProblemType.IMAGE_CLASSIFICATION:
            return "image_classification"
        return "classification"

    def get_feature_schema(self, job_id: str, session: Session) -> ModelFeatureSchemaResponse:
        """Inspects the dataset and returns feature metadata, bounds, and test sample rows."""
        _, model, data_file = self._get_job_and_model(job_id, session)

        problem_type_str = self._resolve_problem_type_str(model.model_type)
        if problem_type_str == "image_classification":
            raise ValueError("Interactive tabular inference is not available for image classification models.")

        upload_folder = get_settings().upload_folder
        file_path = Path(upload_folder) / data_file.disk_name

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist at {file_path}")

        features_df = pd.read_csv(file_path).dropna()
        target_field = model.target_field or ""

        if target_field and target_field in features_df.columns:
            y_series = features_df[target_field]
            X_df = features_df.drop(columns=[target_field])
        else:
            y_series = None
            X_df = features_df

        # Resolve class names if classification
        class_names: list[str] | None = None
        if problem_type_str == "classification" and y_series is not None:
            if not pd.api.types.is_numeric_dtype(y_series):
                class_names = [str(c) for c in pd.Categorical(y_series).categories]
            else:
                class_names = [str(c) for c in sorted(y_series.unique())]

        # Extract schema items for numerical features
        schema_items: list[FeatureSchemaItem] = []
        for col in X_df.columns:
            col_series = pd.to_numeric(X_df[col], errors="coerce").dropna()
            if col_series.empty:
                continue

            c_min = float(col_series.min())
            c_max = float(col_series.max())
            c_mean = float(col_series.mean())
            c_median = float(col_series.median())

            # Detect whether column is integer-like
            is_int = bool(np.all(np.equal(np.mod(col_series.values, 1), 0)))
            dtype_str = "int" if is_int else "float"

            if is_int:
                step = 1.0
            else:
                span = c_max - c_min
                step = round(max(span / 100.0, 0.01), 3) if span > 0 else 0.1

            schema_items.append(
                FeatureSchemaItem(
                    name=col,
                    dtype=dtype_str,
                    min=round(c_min, 4),
                    max=round(c_max, 4),
                    mean=round(c_mean, 4),
                    median=round(c_median, 4),
                    step=step,
                    default_value=round(c_median, 4),
                )
            )

        # Generate sample test records
        sample_records: list[SampleTestRecord] = []
        training_split = model.training_split if model.training_split else 80.0
        shuffled_df = features_df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(shuffled_df) * float(training_split) / 100.0)
        test_df = shuffled_df.iloc[split_idx:]

        num_samples = min(10, len(test_df))
        if num_samples > 0:
            sample_subset = test_df.head(num_samples)
            for idx, (_, row) in enumerate(sample_subset.iterrows()):
                feature_vals = {item.name: float(row[item.name]) for item in schema_items if item.name in row}
                actual_class: int | None = None
                actual_class_name: str | None = None
                actual_value: float | None = None

                if target_field and target_field in row:
                    raw_target = row[target_field]
                    if problem_type_str == "classification":
                        if class_names and str(raw_target) in class_names:
                            actual_class = class_names.index(str(raw_target))
                            actual_class_name = str(raw_target)
                        elif str(raw_target).isdigit() and int(raw_target) < len(class_names or []):
                            actual_class = int(raw_target)
                            actual_class_name = class_names[actual_class] if class_names else str(actual_class)
                        else:
                            try:
                                actual_class = int(float(raw_target))
                                actual_class_name = str(raw_target)
                            except ValueError:
                                actual_class = 0
                                actual_class_name = str(raw_target)
                    else:
                        try:
                            actual_value = round(float(raw_target), 4)
                        except (ValueError, TypeError):
                            actual_value = None

                sample_records.append(
                    SampleTestRecord(
                        index=idx,
                        features=feature_vals,
                        actual_class=actual_class,
                        actual_class_name=actual_class_name,
                        actual_value=actual_value,
                    )
                )

        return ModelFeatureSchemaResponse(
            job_id=job_id,
            target_field=target_field,
            problem_type=problem_type_str,
            features=schema_items,
            class_names=class_names,
            sample_records=sample_records,
        )

    def run_inference(self, job_id: str, inputs: dict[str, float], session: Session) -> InferenceResponse:
        """Executes inference on given input feature dictionary."""
        _, model, data_file = self._get_job_and_model(job_id, session)

        problem_type_str = self._resolve_problem_type_str(model.model_type)
        if problem_type_str == "image_classification":
            raise ValueError("Interactive tabular inference is not available for image classification models.")

        # Load model from memory cache or disk
        model_path = EXPORTS_BASE / job_id / "model.keras"
        keras_model = self.model_cache.get_or_load(job_id, model_path)

        # Determine feature names and class names
        upload_folder = get_settings().upload_folder
        file_path = Path(upload_folder) / data_file.disk_name
        features_df = pd.read_csv(file_path).dropna()

        target_field = model.target_field or ""
        if target_field and target_field in features_df.columns:
            y_series = features_df[target_field]
            X_df = features_df.drop(columns=[target_field])
        else:
            y_series = None
            X_df = features_df

        class_names: list[str] = []
        if problem_type_str == "classification" and y_series is not None:
            if not pd.api.types.is_numeric_dtype(y_series):
                class_names = [str(c) for c in pd.Categorical(y_series).categories]
            else:
                class_names = [str(c) for c in sorted(y_series.unique())]

        # Order inputs according to dataset columns
        feature_vector: list[float] = []
        for col in X_df.columns:
            if col in inputs:
                try:
                    val = float(inputs[col])
                except (ValueError, TypeError):
                    val = float(X_df[col].median())
                feature_vector.append(val)
            else:
                # Fallback to column median
                feature_vector.append(float(X_df[col].median()))

        input_array = np.array([feature_vector], dtype=np.float32)

        # Run model inference and measure latency
        t0 = time.perf_counter()
        raw_pred = keras_model.predict(input_array, verbose=0)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if problem_type_str == "classification":
            pred_classes, pred_probs = _resolve_classification_predictions(raw_pred)
            pred_class = int(pred_classes[0])
            pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"

            probs_row = pred_probs[0]
            num_classes = len(probs_row)
            class_prob_list: list[ClassProbability] = []
            for i in range(num_classes):
                name = class_names[i] if i < len(class_names) else f"Class {i}"
                prob_val = float(np.clip(probs_row[i], 0.0, 1.0))
                class_prob_list.append(
                    ClassProbability(
                        class_index=i,
                        class_name=name,
                        probability=round(prob_val, 4),
                    )
                )

            confidence = round(float(np.clip(probs_row[pred_class], 0.0, 1.0)), 4)

            return InferenceResponse(
                success=True,
                problem_type="classification",
                predicted_class=pred_class,
                predicted_label=pred_label,
                confidence=confidence,
                probabilities=class_prob_list,
                latency_ms=latency_ms,
                input_features={col: feature_vector[idx] for idx, col in enumerate(X_df.columns)},
            )
        else:
            # Regression
            pred_val = float(raw_pred.flatten()[0])
            return InferenceResponse(
                success=True,
                problem_type="regression",
                predicted_value=round(pred_val, 4),
                latency_ms=latency_ms,
                input_features={col: feature_vector[idx] for idx, col in enumerate(X_df.columns)},
            )
