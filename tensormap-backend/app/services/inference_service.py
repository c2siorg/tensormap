"""Service for interactive real-time model inference and schema inspection.

Features:
- Thread-safe LRU model memory cache with cached feature columns and medians
- Feature schema extraction (bounds, distributions, steps) to drive UI sliders
- Sample test dataset record generator with ground truth targets
- Microsecond/low-latency single-row model prediction for classification and regression
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlmodel import Session

from app.config import get_settings
from app.exceptions import AppException
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
from app.services.dataset_utils import prepare_tabular_dataset
from app.services.interpretability import (
    EXPORTS_BASE,
    _resolve_classification_predictions,
)
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CachedModelEntry:
    """In-memory cache entry containing the loaded Keras model and dataset metadata."""

    model: Any
    problem_type: str
    feature_names: list[str]
    class_names: list[str]
    feature_medians: dict[str, float]
    feature_dtypes: dict[str, str]
    lock: threading.Lock = field(default_factory=threading.Lock)


class ModelInferenceCache:
    """Thread-safe LRU in-memory cache for loaded Keras models and metadata.

    Caches the compiled model alongside feature column names, class labels, and medians to
    avoid expensive disk and CSV I/O on live predict requests, bringing single-row inference
    latency down to 1-5 milliseconds.
    """

    def __init__(self, maxsize: int = 5):
        self._cache: OrderedDict[str, CachedModelEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._loading_locks: dict[str, threading.Lock] = {}
        self._maxsize = maxsize

    def get_or_load(
        self,
        job_id: str,
        model_path: Path,
        file_path: Path | None = None,
        target_field: str | None = None,
        model_type: int | None = None,
    ) -> CachedModelEntry:
        """Retrieves an existing cached model or loads and caches it from disk."""
        with self._lock:
            if job_id in self._cache:
                self._cache.move_to_end(job_id)
                return self._cache[job_id]
            if job_id not in self._loading_locks:
                self._loading_locks[job_id] = threading.Lock()
            job_lock = self._loading_locks[job_id]

        with job_lock:
            with self._lock:
                if job_id in self._cache:
                    self._cache.move_to_end(job_id)
                    return self._cache[job_id]

            if not model_path.exists():
                raise FileNotFoundError(f"Trained model not found at {model_path}")

            # Lazy import TensorFlow
            import tensorflow as tf

            logger.info(f"Loading model into inference cache for job {job_id}")
            keras_model = tf.keras.models.load_model(model_path)

            # Precompute and cache dataset metadata once per job
            feature_names: list[str] = []
            class_names: list[str] = []
            feature_medians: dict[str, float] = {}
            feature_dtypes: dict[str, str] = {}

            if file_path and Path(file_path).exists():
                dataset_info = prepare_tabular_dataset(file_path, target_field=target_field)
                feature_names = dataset_info["feature_names"]
                class_names = dataset_info["class_names"] or []
                feature_medians = dataset_info["feature_medians"]
                feature_dtypes = dataset_info["feature_dtypes"]

            problem_type_str = "regression" if model_type == ProblemType.REGRESSION else "classification"

            entry = CachedModelEntry(
                model=keras_model,
                problem_type=problem_type_str,
                feature_names=feature_names,
                class_names=class_names,
                feature_medians=feature_medians,
                feature_dtypes=feature_dtypes,
            )

            with self._lock:
                if job_id in self._cache:
                    self._cache.move_to_end(job_id)
                    return self._cache[job_id]
                if len(self._cache) >= self._maxsize:
                    evicted_job_id, _ = self._cache.popitem(last=False)
                    logger.info(f"Evicted model {evicted_job_id} from inference cache")
                self._cache[job_id] = entry
                self._loading_locks.pop(job_id, None)
                return entry

    def clear(self) -> None:
        """Empties the cache."""
        with self._lock:
            self._cache.clear()
            self._loading_locks.clear()


class InferenceService:
    """Provides feature schema inspection and live inference execution."""

    def __init__(self):
        self.model_cache = ModelInferenceCache(maxsize=5)

    def _get_job_and_model(self, job_id: str, session: Session) -> tuple[TrainingJob, ModelBasic, DataFile]:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise AppException(404, f"Training job not found: {job_id}")

        if job.status != TrainingStatus.COMPLETED:
            raise AppException(
                400, f"Inference is only available for completed jobs (current status: {job.status.value})"
            )

        model = session.get(ModelBasic, job.model_id)
        if not model:
            raise AppException(404, f"Model not found for job: {job_id}")

        if not model.file_id:
            raise AppException(400, f"No dataset linked to model {model.model_name}")

        data_file = session.get(DataFile, model.file_id)
        if not data_file:
            raise AppException(404, f"Data file not found for model {model.model_name}")

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
            raise AppException(400, "Interactive tabular inference is not available for image classification models.")

        upload_folder = get_settings().upload_folder
        file_path = Path(upload_folder) / data_file.disk_name

        dataset_info = prepare_tabular_dataset(
            file_path=file_path,
            target_field=model.target_field,
            training_split=model.training_split or 80.0,
        )

        class_names = dataset_info["class_names"]
        schema_items: list[FeatureSchemaItem] = []
        for col in dataset_info["feature_names"]:
            schema_items.append(
                FeatureSchemaItem(
                    name=col,
                    dtype=dataset_info["feature_dtypes"][col],
                    min=round(dataset_info["feature_mins"][col], 4),
                    max=round(dataset_info["feature_maxs"][col], 4),
                    mean=round(dataset_info["feature_means"][col], 4),
                    median=round(dataset_info["feature_medians"][col], 4),
                    step=dataset_info["feature_steps"][col],
                    default_value=round(dataset_info["feature_medians"][col], 4),
                )
            )

        # Generate sample test records from test split
        raw_df = dataset_info["raw_df"]
        split_idx = dataset_info["split_idx"]
        test_df = raw_df.iloc[split_idx:]
        sample_records: list[SampleTestRecord] = []
        num_samples = min(10, len(test_df))
        target_field = model.target_field or ""

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
                        if isinstance(raw_target, (float, np.floating)) and float(raw_target).is_integer():
                            raw_str = str(int(raw_target))
                        else:
                            raw_str = str(raw_target)

                        if class_names and raw_str in class_names:
                            actual_class = class_names.index(raw_str)
                            actual_class_name = raw_str
                        elif class_names and str(raw_target) in class_names:
                            actual_class = class_names.index(str(raw_target))
                            actual_class_name = str(raw_target)
                        else:
                            actual_class = None
                            actual_class_name = raw_str
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
        """Executes low-latency inference on given input feature dictionary using cached model & metadata."""
        _, model, data_file = self._get_job_and_model(job_id, session)

        problem_type_str = self._resolve_problem_type_str(model.model_type)
        if problem_type_str == "image_classification":
            raise AppException(400, "Interactive tabular inference is not available for image classification models.")

        model_path = EXPORTS_BASE / job_id / "model.keras"
        upload_folder = get_settings().upload_folder
        file_path = Path(upload_folder) / data_file.disk_name

        entry = self.model_cache.get_or_load(
            job_id=job_id,
            model_path=model_path,
            file_path=file_path,
            target_field=model.target_field,
            model_type=model.model_type,
        )

        # Strict input validation: reject unknown features, missing features, and invalid numeric values
        unknown_keys = set(inputs.keys()) - set(entry.feature_names)
        if unknown_keys:
            raise AppException(400, f"Unknown feature(s) provided: {', '.join(sorted(unknown_keys))}")

        missing_keys = set(entry.feature_names) - set(inputs.keys())
        if missing_keys:
            raise AppException(400, f"Missing required feature(s): {', '.join(sorted(missing_keys))}")

        feature_vector: list[float] = []
        for col in entry.feature_names:
            val = inputs[col]
            try:
                f_val = float(val)
                if np.isnan(f_val) or np.isinf(f_val):
                    raise ValueError()
            except (ValueError, TypeError) as err:
                raise AppException(400, f"Invalid numeric value for feature '{col}': {val!r}") from err
            feature_vector.append(f_val)

        # Run model(x, training=False) without overhead of model.predict()
        import tensorflow as tf

        input_tensor = tf.convert_to_tensor([feature_vector], dtype=tf.float32)

        t0 = time.perf_counter()
        with entry.lock:
            raw_pred = entry.model(input_tensor, training=False).numpy()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if problem_type_str == "classification":
            pred_classes, pred_probs = _resolve_classification_predictions(raw_pred)
            pred_class = int(pred_classes[0])
            class_names = entry.class_names
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
                input_features={col: feature_vector[idx] for idx, col in enumerate(entry.feature_names)},
            )
        else:
            # Regression
            pred_val = float(raw_pred.flatten()[0])
            return InferenceResponse(
                success=True,
                problem_type="regression",
                predicted_value=round(pred_val, 4),
                latency_ms=latency_ms,
                input_features={col: feature_vector[idx] for idx, col in enumerate(entry.feature_names)},
            )
