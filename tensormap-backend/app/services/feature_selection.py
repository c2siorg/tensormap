"""Feature relevance scoring and safe CSV feature pruning."""

import contextlib
import os
import shutil
import uuid as uuid_pkg
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sqlalchemy import select
from sqlmodel import Session

from app.config import get_settings
from app.models import DataFile
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

_CLASSIFICATION_CARDINALITY_LIMIT = 20


def _resp(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    return {"success": success, "message": message, "data": data}, status_code


def _encode_feature(values: pd.Series) -> tuple[np.ndarray, bool]:
    """Fill missing values and return numeric values plus their discrete flag."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = numeric.median()
        fill_value = float(median) if pd.notna(median) and np.isfinite(median) else 0.0
        return numeric.fillna(fill_value).to_numpy(dtype=float), False

    categories = values.astype("string").fillna("__missing__").astype(str)
    codes, _ = pd.factorize(categories, sort=True)
    return codes.astype(float), True


def _infer_task(target: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    return "classification" if target.nunique() <= _CLASSIFICATION_CARDINALITY_LIMIT else "regression"


def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(scores, dtype=float)
    finite_mask = np.isfinite(scores)
    if finite_mask.any():
        finite_scores = scores[finite_mask]
        low = float(finite_scores.min())
        high = float(finite_scores.max())
        if not np.isclose(low, high):
            normalized[finite_mask] = (finite_scores - low) / (high - low)
    normalized[np.isposinf(scores)] = 1.0
    return normalized


def _get_csv(db: Session, file_id: uuid_pkg.UUID) -> tuple[DataFile | None, pd.DataFrame | None, tuple | None]:
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return None, None, _resp(404, False, "File doesn't exist")
    if file.file_type != "csv":
        return None, None, _resp(400, False, "Feature selection is only supported for CSV files")

    file_path = os.path.join(get_settings().upload_folder, file.disk_name)
    try:
        return file, pd.read_csv(file_path), None
    except FileNotFoundError:
        return None, None, _resp(404, False, "Dataset file not found")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.exception("Could not parse feature selection CSV %s", file_path)
        return None, None, _resp(422, False, f"Could not parse dataset CSV: {exc}")
    except OSError as exc:
        logger.exception("Could not read feature selection CSV %s", file_path)
        return None, None, _resp(500, False, f"Could not read dataset CSV: {exc}")


def analyze_features(db: Session, file_id: uuid_pkg.UUID, target: str, task: str = "auto") -> tuple:
    """Score each predictor by mutual information, ANOVA F-statistic, and variance."""
    _, frame, error = _get_csv(db, file_id)
    if error:
        return error

    if target not in frame.columns:
        return _resp(422, False, f"Target column '{target}' not found in dataset")
    if len(frame.columns) < 2:
        return _resp(422, False, "Dataset must contain at least one predictor and a target")

    target_values = frame[target]
    if task == "auto":
        task = _infer_task(target_values.dropna())
    if task not in ("classification", "regression"):
        return _resp(422, False, "Task must be 'auto', 'classification', or 'regression'")

    if task == "regression":
        target_values = pd.to_numeric(target_values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_rows = target_values.notna()
    frame = frame.loc[valid_rows].reset_index(drop=True)
    target_values = target_values.loc[valid_rows].reset_index(drop=True)
    if len(target_values) < 2 or target_values.nunique() < 2:
        return _resp(422, False, "Target must contain at least two non-null rows and two distinct values")

    feature_names = [column for column in frame.columns if column != target]
    encoded = [_encode_feature(frame[column]) for column in feature_names]
    features = np.column_stack([values for values, _ in encoded])
    discrete_features = np.array([is_discrete for _, is_discrete in encoded], dtype=bool)
    y = (
        pd.factorize(target_values.astype("string").astype(str), sort=True)[0]
        if task == "classification"
        else target_values.to_numpy(dtype=float)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if task == "classification":
            mutual_information = mutual_info_classif(features, y, discrete_features=discrete_features, random_state=0)
            anova_scores = f_classif(features, y)[0]
        else:
            mutual_information = mutual_info_regression(
                features, y, discrete_features=discrete_features, random_state=0
            )
            anova_scores = f_regression(features, y)[0]

    variance_scores = np.array(
        [
            1.0 - float(pd.Series(values).value_counts(normalize=True).pow(2).sum())
            if is_discrete
            else float(np.var(values))
            for values, is_discrete in encoded
        ]
    )
    normalized_mi = _normalise_scores(mutual_information)
    normalized_anova = _normalise_scores(anova_scores)
    normalized_variance = _normalise_scores(variance_scores)
    composite = (normalized_mi + normalized_anova + normalized_variance) / 3

    max_float = np.finfo(float).max
    mutual_information = np.nan_to_num(mutual_information, nan=0.0, posinf=max_float, neginf=0.0)
    anova_scores = np.nan_to_num(anova_scores, nan=0.0, posinf=max_float, neginf=0.0)
    variance_scores = np.nan_to_num(variance_scores, nan=0.0, posinf=max_float, neginf=0.0)
    scores = [
        {
            "feature": feature,
            "mutual_information": round(float(mi), 6),
            "anova_f": round(float(anova), 6),
            "variance": round(float(variance), 6),
            "composite_score": round(float(rank), 6),
        }
        for feature, mi, anova, variance, rank in zip(
            feature_names,
            mutual_information,
            anova_scores,
            variance_scores,
            composite,
            strict=True,
        )
    ]
    scores.sort(key=lambda result: result["composite_score"], reverse=True)

    return _resp(
        200,
        True,
        "Feature relevance scores computed successfully",
        {"target": target, "task": task, "sample_rows": len(frame), "scores": scores},
    )


def apply_feature_selection(db: Session, file_id: uuid_pkg.UUID, target: str, columns_to_drop: list[str]) -> tuple:
    """Back up a dataset and atomically remove selected predictor columns."""
    file, frame, error = _get_csv(db, file_id)
    if error:
        return error

    if target not in frame.columns:
        return _resp(422, False, f"Target column '{target}' not found in dataset")
    if not columns_to_drop:
        return _resp(422, False, "Select at least one predictor column to remove")
    if target in columns_to_drop:
        return _resp(422, False, "The target column cannot be removed")
    if len(columns_to_drop) != len(set(columns_to_drop)):
        return _resp(422, False, "Duplicate columns_to_drop entries are not allowed")

    missing_columns = [column for column in columns_to_drop if column not in frame.columns]
    if missing_columns:
        return _resp(422, False, f"Columns not found in dataset: {missing_columns}")
    if len(frame.columns) - len(columns_to_drop) < 2:
        return _resp(422, False, "At least one predictor must remain alongside the target")

    file_path = os.path.join(get_settings().upload_folder, file.disk_name)
    backup_path = file_path + ".bak"
    tmp_path = file_path + ".tmp"
    backup_tmp_path = backup_path + ".tmp"
    selected = frame.drop(columns=columns_to_drop)
    backup_created = False
    dataset_replaced = False

    try:
        selected.to_csv(tmp_path, index=False)
        shutil.copy2(file_path, backup_tmp_path)
        os.replace(backup_tmp_path, backup_path)
        backup_created = True
        os.replace(tmp_path, file_path)
        dataset_replaced = True
        file.columns = list(selected.columns)
        db.add(file)
        db.commit()
        return _resp(
            200,
            True,
            "Selected features removed successfully",
            {
                "columns": list(selected.columns),
                "removed_columns": columns_to_drop,
                "backup_file": file.disk_name + ".bak",
            },
        )
    except Exception as exc:
        logger.exception("Error applying feature selection for file_id=%s", file_id)
        for temporary_path in (tmp_path, backup_tmp_path):
            with contextlib.suppress(OSError):
                os.remove(temporary_path)
        try:
            db.rollback()
            if backup_created and dataset_replaced:
                shutil.copy2(backup_path, file_path)
        except OSError as restore_error:
            logger.exception("Could not restore dataset after feature selection failure")
            return _resp(500, False, f"Feature selection failed and dataset restoration failed: {restore_error}")
        return _resp(500, False, f"Error applying feature selection: {exc}")


__all__ = ["analyze_features", "apply_feature_selection"]
