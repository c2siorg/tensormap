"""Shared tabular dataset preparation and preprocessing helpers."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def prepare_tabular_dataset(
    dataset: pd.DataFrame | Path | str | None = None,
    target_field: str | None = None,
    training_split: float = 80.0,
    random_state: int = 42,
    *,
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    """Loads, cleans, and structures a tabular dataset consistently across services.

    Ensures that training, interpretability, and live inference use the exact same:
    - Null-value dropping (dropna)
    - Deterministic shuffle (random_state=42)
    - Valid numerical feature filtering
    - Train / test split indexing
    - Class label encoding & class-name extraction

    Args:
        dataset: Path to dataset CSV or an existing pandas DataFrame (or file_path)
        target_field: Optional target column name
        training_split: Percentage of data used for training (0 < split < 100)
        random_state: Seed for reproducible shuffle (default: 42)
        file_path: Alias for dataset when passing file path as keyword argument

    Returns:
        dict containing:
            - "raw_df": cleaned and shuffled DataFrame
            - "feature_names": list[str] of valid numeric feature column names
            - "X_df": DataFrame of non-target features
            - "y_series": Series of raw target values (or None)
            - "y_encoded": Series or ndarray of encoded target values (or None)
            - "class_names": list[str] of unique sorted or categorical class names (or None)
            - "split_idx": int split index between train and test
            - "X_train": training features slice
            - "X_test": testing features slice
            - "y_train": training target slice
            - "y_test": testing target slice
            - "feature_medians": dict[str, float] of column medians
            - "feature_mins": dict[str, float]
            - "feature_maxs": dict[str, float]
            - "feature_means": dict[str, float]
            - "feature_dtypes": dict[str, str] ("int" or "float")
            - "feature_steps": dict[str, float]
    """
    source = dataset if dataset is not None else file_path
    if source is None:
        raise ValueError("A dataset DataFrame or file_path must be provided")

    if isinstance(source, pd.DataFrame):
        df = source.copy().dropna()
    else:
        path_obj = Path(source)
        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found at {path_obj}")
        df = pd.read_csv(path_obj).dropna()

    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values")

    # Deterministic shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_series = None
    y_encoded = None
    class_names = None
    if target_field:
        if target_field not in df.columns:
            raise ValueError(f"Target field '{target_field}' not found in dataset after cleaning")
        y_series = df[target_field]
        non_target_df = df.drop(columns=[target_field])
        if not pd.api.types.is_numeric_dtype(y_series):
            categories = list(pd.Categorical(y_series).categories)
            class_names = [str(c) for c in categories]
            y_encoded = pd.Categorical(y_series).codes
        else:
            unique_vals = sorted(y_series.unique())
            class_names = [
                str(int(c))
                if isinstance(c, (int, np.integer)) or (isinstance(c, (float, np.floating)) and float(c).is_integer())
                else str(c)
                for c in unique_vals
            ]
            y_encoded = y_series.values
    else:
        non_target_df = df

    # Retain strictly columns that can be coerced to numeric and are non-empty
    feature_names: list[str] = []
    feature_medians: dict[str, float] = {}
    feature_mins: dict[str, float] = {}
    feature_maxs: dict[str, float] = {}
    feature_means: dict[str, float] = {}
    feature_dtypes: dict[str, str] = {}
    feature_steps: dict[str, float] = {}

    for col in non_target_df.columns:
        num_s = pd.to_numeric(non_target_df[col], errors="coerce").dropna()
        if num_s.empty:
            continue
        feature_names.append(col)
        c_min = float(num_s.min())
        c_max = float(num_s.max())
        c_mean = float(num_s.mean())
        c_median = float(num_s.median())
        is_int = bool(np.all(np.equal(np.mod(num_s.values, 1), 0)))

        feature_mins[col] = c_min
        feature_maxs[col] = c_max
        feature_means[col] = c_mean
        feature_medians[col] = c_median
        feature_dtypes[col] = "int" if is_int else "float"

        if is_int:
            step = 1.0
        else:
            span = c_max - c_min
            step = round(max(span / 100.0, 0.01), 3) if span > 0 else 0.1
        feature_steps[col] = step

    split_pct = float(training_split) if training_split else 80.0
    split_idx = int(len(df) * split_pct / 100.0)

    X_train = non_target_df.iloc[:split_idx]
    X_test = non_target_df.iloc[split_idx:]
    y_train = y_encoded[:split_idx] if y_encoded is not None else None
    y_test = y_encoded[split_idx:] if y_encoded is not None else None

    return {
        "raw_df": df,
        "feature_names": feature_names,
        "X_df": non_target_df,
        "y_series": y_series,
        "y_encoded": y_encoded,
        "class_names": class_names,
        "split_idx": split_idx,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_medians": feature_medians,
        "feature_mins": feature_mins,
        "feature_maxs": feature_maxs,
        "feature_means": feature_means,
        "feature_dtypes": feature_dtypes,
        "feature_steps": feature_steps,
    }
