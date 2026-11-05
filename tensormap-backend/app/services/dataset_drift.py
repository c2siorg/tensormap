"""Dataset Drift and Covariate Shift Profiler for Tabular Datasets.

Computes distribution divergence between baseline and target datasets (or train/test splits)
using Kolmogorov-Smirnov 2-sample tests, Population Stability Index (PSI),
Wasserstein Distance, and Chi-square contingency tests.
"""

import math
import uuid as uuid_pkg
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
from sqlmodel import Session, select

from app.config import get_settings
from app.models import DataFile
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _response(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    """Build the standard API response tuple."""
    return {"success": success, "message": message, "data": data}, status_code


def _compute_numeric_psi(
    vals_base: np.ndarray, vals_target: np.ndarray, n_bins: int | None = None
) -> tuple[float, list[dict]]:
    """Compute Population Stability Index (PSI) and histogram distribution bins for numeric data."""
    if len(vals_base) == 0 or len(vals_target) == 0:
        return 0.0, []

    if n_bins is None:
        n_bins = min(10, max(3, int(min(len(vals_base), len(vals_target)) / 15)))

    # Decile percentiles on baseline
    percentiles = np.linspace(0, 100, n_bins + 1)
    raw_edges = np.percentile(vals_base, percentiles)
    bin_edges = np.unique(raw_edges)

    # If all baseline values are identical, create 2 trivial bins
    if len(bin_edges) < 2:
        val = float(bin_edges[0])
        bin_edges = np.array([val - 1.0, val, val + 1.0])

    # Extend endpoints to capture target range as well
    min_val = min(float(np.min(vals_base)), float(np.min(vals_target)))
    max_val = max(float(np.max(vals_base)), float(np.max(vals_target)))
    bin_edges[0] = min(bin_edges[0], min_val)
    bin_edges[-1] = max(bin_edges[-1], max_val)

    # Bin counts
    base_counts, _ = np.histogram(vals_base, bins=bin_edges)
    target_counts, _ = np.histogram(vals_target, bins=bin_edges)

    n_base = float(len(vals_base))
    n_target = float(len(vals_target))
    k = float(len(base_counts))

    psi_total = 0.0
    distribution_bins = []

    for i in range(len(base_counts)):
        b_start = round(float(bin_edges[i]), 4)
        b_end = round(float(bin_edges[i + 1]), 4)
        label = f"[{b_start:.2f}, {b_end:.2f}]" if i == 0 else f"({b_start:.2f}, {b_end:.2f}]"

        b_pct_raw = base_counts[i] / n_base
        t_pct_raw = target_counts[i] / n_target

        # Laplace smoothing to avoid division by zero or log(0) in PSI
        b_pct_smooth = (base_counts[i] + 0.5) / (n_base + 0.5 * k)
        t_pct_smooth = (target_counts[i] + 0.5) / (n_target + 0.5 * k)

        bin_psi = (t_pct_smooth - b_pct_smooth) * math.log(t_pct_smooth / b_pct_smooth)
        psi_total += bin_psi

        distribution_bins.append(
            {
                "bin_label": label,
                "bin_start": b_start,
                "bin_end": b_end,
                "baseline_count": int(base_counts[i]),
                "target_count": int(target_counts[i]),
                "baseline_pct": round(float(b_pct_raw), 4),
                "target_pct": round(float(t_pct_raw), 4),
            }
        )

    return round(float(psi_total), 4), distribution_bins


def _compute_categorical_psi(base_series: pd.Series, target_series: pd.Series) -> tuple[float, list[dict], list[str]]:
    """Compute Categorical PSI and category frequency comparisons."""
    base_counts = base_series.value_counts()
    target_counts = target_series.value_counts()

    all_categories = sorted(list(set(base_counts.index).union(set(target_counts.index))))
    n_base = float(len(base_series))
    n_target = float(len(target_series))

    unseen_in_target = [str(cat) for cat in target_counts.index if cat not in base_counts]

    eps = 1e-4
    psi_total = 0.0
    cat_distribution = []

    for cat in all_categories:
        b_count = int(base_counts.get(cat, 0))
        t_count = int(target_counts.get(cat, 0))

        b_pct = b_count / n_base if n_base > 0 else 0.0
        t_pct = t_count / n_target if n_target > 0 else 0.0

        b_pct_smooth = max(b_pct, eps)
        t_pct_smooth = max(t_pct, eps)

        bin_psi = (t_pct_smooth - b_pct_smooth) * math.log(t_pct_smooth / b_pct_smooth)
        psi_total += bin_psi

        cat_distribution.append(
            {
                "category": str(cat),
                "baseline_count": b_count,
                "target_count": t_count,
                "baseline_pct": round(float(b_pct), 4),
                "target_pct": round(float(t_pct), 4),
            }
        )

    # Sort categories by descending total frequency
    cat_distribution.sort(key=lambda x: x["baseline_count"] + x["target_count"], reverse=True)
    return round(float(psi_total), 4), cat_distribution, unseen_in_target


def compute_dataset_drift(
    db: Session,
    file_id: uuid_pkg.UUID,
    baseline_file_id: uuid_pkg.UUID | None = None,
    training_split: float = 0.8,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[dict, int]:
    """Profile dataset drift and covariate shift between baseline and target datasets.

    If baseline_file_id is provided, compares two separate uploaded dataset files.
    Otherwise, splits the dataset by `training_split` ratio to compare train vs test split.
    """
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if file is None:
        return _response(404, False, "Dataset file was not found")

    if file.file_type != "csv":
        return _response(
            400,
            False,
            "Dataset drift profiling is only supported for tabular CSV datasets.",
        )

    settings = get_settings()
    upload_folder = Path(settings.upload_folder).resolve()

    # Load target CSV
    target_path = (upload_folder / file.disk_name).resolve()
    if target_path == upload_folder or upload_folder not in target_path.parents:
        return _response(400, False, "Could not access target CSV file.")

    try:
        frame_target = pd.read_csv(target_path)
    except (pd.errors.ParserError, OSError, UnicodeDecodeError, MemoryError, ValueError):
        logger.exception("Could not read uploaded CSV for file_id=%s", file_id)
        return _response(400, False, "Could not read target CSV file.")

    if len(frame_target.index) == 0:
        return _response(400, False, "Target dataset contains no rows.")

    mode = "dataset_comparison" if baseline_file_id is not None else "train_test_split"

    if baseline_file_id is not None:
        baseline_file = db.exec(select(DataFile).where(DataFile.id == baseline_file_id)).first()
        if baseline_file is None:
            return _response(404, False, "Baseline dataset file was not found")

        baseline_path = (upload_folder / baseline_file.disk_name).resolve()
        if baseline_path == upload_folder or upload_folder not in baseline_path.parents:
            return _response(400, False, "Could not access baseline CSV file.")

        try:
            frame_baseline = pd.read_csv(baseline_path)
        except (pd.errors.ParserError, OSError, UnicodeDecodeError, MemoryError, ValueError):
            logger.exception("Could not read baseline CSV for file_id=%s", baseline_file_id)
            return _response(400, False, "Could not read baseline CSV file.")

        if len(frame_baseline.index) == 0:
            return _response(400, False, "Baseline dataset contains no rows.")
        baseline_name = baseline_file.file_name
        target_name = file.file_name
    else:
        # Split single dataset into train (baseline) and test (target)
        if len(frame_target.index) < 10:
            return _response(
                400,
                False,
                "Dataset must contain at least 10 rows to perform train-test drift profiling.",
            )

        if not (0.1 <= training_split <= 0.9):
            return _response(
                400,
                False,
                "training_split must be between 0.1 and 0.9.",
            )

        # Deterministic stratified or random split
        train_indices = frame_target.sample(frac=training_split, random_state=seed).index
        frame_baseline = frame_target.loc[train_indices].reset_index(drop=True)
        frame_target = frame_target.drop(train_indices).reset_index(drop=True)
        baseline_name = f"Training Split ({int(training_split * 100)}%)"
        target_name = f"Testing Split ({int((1.0 - training_split) * 100)}%)"

    common_columns = [col for col in frame_baseline.columns if col in frame_target.columns]
    if not common_columns:
        return _response(400, False, "No common columns found between baseline and target datasets.")

    features_report = []
    drifted_count = 0

    for col in common_columns:
        col_base = frame_baseline[col]
        col_target = frame_target[col]

        is_numeric = pd.api.types.is_numeric_dtype(col_base) and pd.api.types.is_numeric_dtype(col_target)

        # Basic missingness statistics
        base_missing = int(col_base.isna().sum())
        target_missing = int(col_target.isna().sum())
        base_missing_pct = round(base_missing / len(col_base), 4) if len(col_base) > 0 else 0.0
        target_missing_pct = round(target_missing / len(col_target), 4) if len(col_target) > 0 else 0.0

        if is_numeric:
            base_clean = col_base.dropna().values.astype(float)
            target_clean = col_target.dropna().values.astype(float)

            if len(base_clean) < 2 or len(target_clean) < 2:
                features_report.append(
                    {
                        "feature_name": col,
                        "data_type": "numeric",
                        "drift_detected": False,
                        "drift_severity": "none",
                        "insufficient_data": True,
                        "ks_statistic": None,
                        "ks_pvalue": None,
                        "wasserstein_distance": None,
                        "normalized_wasserstein": None,
                        "psi": None,
                        "stats": {
                            "baseline": {
                                "count": len(base_clean),
                                "mean": round(float(np.mean(base_clean)), 4) if len(base_clean) > 0 else None,
                                "std": round(float(np.std(base_clean)), 4) if len(base_clean) > 0 else None,
                                "min": round(float(np.min(base_clean)), 4) if len(base_clean) > 0 else None,
                                "max": round(float(np.max(base_clean)), 4) if len(base_clean) > 0 else None,
                                "missing_pct": base_missing_pct,
                            },
                            "target": {
                                "count": len(target_clean),
                                "mean": round(float(np.mean(target_clean)), 4) if len(target_clean) > 0 else None,
                                "std": round(float(np.std(target_clean)), 4) if len(target_clean) > 0 else None,
                                "min": round(float(np.min(target_clean)), 4) if len(target_clean) > 0 else None,
                                "max": round(float(np.max(target_clean)), 4) if len(target_clean) > 0 else None,
                                "missing_pct": target_missing_pct,
                            },
                        },
                        "distribution": [],
                    }
                )
                continue

            # KS 2-sample test
            ks_res = ks_2samp(base_clean, target_clean)
            ks_stat = round(float(ks_res.statistic), 4)
            ks_pvalue = round(float(ks_res.pvalue), 4)

            # Wasserstein distance & normalized distance
            w_dist = float(wasserstein_distance(base_clean, target_clean))
            pooled_std = float(np.sqrt((np.var(base_clean) + np.var(target_clean)) / 2.0))
            norm_w_dist = round(float(w_dist / pooled_std), 4) if pooled_std > 1e-9 else 0.0

            # PSI & histogram distribution
            psi, distribution_bins = _compute_numeric_psi(base_clean, target_clean)

            # Determine drift severity based on statistical hypothesis testing and effect size
            is_significant = ks_pvalue < alpha
            if is_significant and (psi >= 0.25 or ks_stat >= 0.20):
                severity = "severe"
                drift_detected = True
            elif (is_significant and (psi >= 0.10 or ks_stat >= 0.10)) or (psi >= 0.25 and ks_stat >= 0.20):
                severity = "moderate"
                drift_detected = True
            else:
                severity = "none"
                drift_detected = False

            if drift_detected:
                drifted_count += 1

            features_report.append(
                {
                    "feature_name": col,
                    "data_type": "numeric",
                    "drift_detected": drift_detected,
                    "drift_severity": severity,
                    "insufficient_data": False,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "wasserstein_distance": round(w_dist, 4),
                    "normalized_wasserstein": norm_w_dist,
                    "psi": psi,
                    "stats": {
                        "baseline": {
                            "count": len(base_clean),
                            "mean": round(float(np.mean(base_clean)), 4),
                            "std": round(float(np.std(base_clean)), 4),
                            "min": round(float(np.min(base_clean)), 4),
                            "max": round(float(np.max(base_clean)), 4),
                            "missing_pct": base_missing_pct,
                        },
                        "target": {
                            "count": len(target_clean),
                            "mean": round(float(np.mean(target_clean)), 4),
                            "std": round(float(np.std(target_clean)), 4),
                            "min": round(float(np.min(target_clean)), 4),
                            "max": round(float(np.max(target_clean)), 4),
                            "missing_pct": target_missing_pct,
                        },
                    },
                    "distribution": distribution_bins,
                }
            )
        else:
            # Categorical feature
            base_cat = col_base.dropna().astype(str)
            target_cat = col_target.dropna().astype(str)

            if len(base_cat) < 2 or len(target_cat) < 2:
                features_report.append(
                    {
                        "feature_name": col,
                        "data_type": "categorical",
                        "drift_detected": False,
                        "drift_severity": "none",
                        "insufficient_data": True,
                        "chi2_statistic": None,
                        "chi2_pvalue": None,
                        "psi": None,
                        "unseen_categories": [],
                        "stats": {
                            "baseline": {
                                "count": len(base_cat),
                                "unique_categories": int(col_base.nunique()),
                                "missing_pct": base_missing_pct,
                            },
                            "target": {
                                "count": len(target_cat),
                                "unique_categories": int(col_target.nunique()),
                                "missing_pct": target_missing_pct,
                            },
                        },
                        "distribution": [],
                    }
                )
                continue

            psi, cat_distribution, unseen_categories = _compute_categorical_psi(base_cat, target_cat)

            # Chi-square test
            all_cats = [d["category"] for d in cat_distribution]
            contingency_matrix = np.array(
                [
                    [d["baseline_count"] for d in cat_distribution],
                    [d["target_count"] for d in cat_distribution],
                ]
            )

            chi2_stat: float | None = None
            chi2_pvalue: float | None = None
            if len(all_cats) >= 2 and np.sum(contingency_matrix) > 0:
                try:
                    res = chi2_contingency(contingency_matrix)
                    chi2_stat = round(float(res.statistic), 4)
                    chi2_pvalue = round(float(res.pvalue), 4)
                except Exception:
                    pass

            is_significant = chi2_pvalue is not None and chi2_pvalue < alpha
            has_unseen = len(unseen_categories) > 0

            if psi >= 0.25 or (is_significant and has_unseen):
                severity = "severe"
                drift_detected = True
            elif psi >= 0.10 or is_significant:
                severity = "moderate"
                drift_detected = True
            else:
                severity = "none"
                drift_detected = False

            if drift_detected:
                drifted_count += 1

            features_report.append(
                {
                    "feature_name": col,
                    "data_type": "categorical",
                    "drift_detected": drift_detected,
                    "drift_severity": severity,
                    "insufficient_data": False,
                    "chi2_statistic": chi2_stat,
                    "chi2_pvalue": chi2_pvalue,
                    "psi": psi,
                    "unseen_categories": unseen_categories,
                    "stats": {
                        "baseline": {
                            "count": len(base_cat),
                            "unique_categories": int(col_base.nunique()),
                            "missing_pct": base_missing_pct,
                        },
                        "target": {
                            "count": len(target_cat),
                            "unique_categories": int(col_target.nunique()),
                            "missing_pct": target_missing_pct,
                        },
                    },
                    "distribution": cat_distribution[:15],  # top 15 categories for visual display
                }
            )

    total_features = len(features_report)
    drift_share = round(drifted_count / total_features, 4) if total_features > 0 else 0.0

    has_severe = any(f.get("drift_severity") == "severe" for f in features_report)
    dataset_drift_detected = has_severe or (drift_share >= 0.30 and drifted_count >= 2)

    result_data = {
        "dataset_drift_detected": dataset_drift_detected,
        "drift_share": drift_share,
        "drifted_features_count": drifted_count,
        "total_features_count": total_features,
        "mode": mode,
        "alpha": alpha,
        "baseline": {
            "name": baseline_name,
            "rows": len(frame_baseline),
        },
        "target": {
            "name": target_name,
            "rows": len(frame_target),
        },
        "features": features_report,
    }

    return _response(200, True, "Dataset drift report generated successfully", result_data)
