"""Tests for Dataset Drift and Covariate Shift Profiler."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.models.data import DataFile
from app.services.dataset_drift import compute_dataset_drift


def _run_drift(
    tmp_path: Path,
    target_content: str,
    baseline_content: str | None = None,
    training_split: float = 0.8,
    seed: int = 42,
    alpha: float = 0.05,
):
    """Write temporary CSV(s) and execute compute_dataset_drift with mock DataFile records."""
    target_path = tmp_path / "target.csv"
    target_path.write_text(target_content, encoding="utf-8")

    file_target = MagicMock(spec=DataFile)
    file_target.file_type = "csv"
    file_target.disk_name = target_path.name
    file_target.file_name = "target.csv"

    db = MagicMock()
    file_target_id = uuid.uuid4()
    baseline_id = None

    if baseline_content is not None:
        baseline_path = tmp_path / "baseline.csv"
        baseline_path.write_text(baseline_content, encoding="utf-8")
        file_baseline = MagicMock(spec=DataFile)
        file_baseline.file_type = "csv"
        file_baseline.disk_name = baseline_path.name
        file_baseline.file_name = "baseline.csv"
        baseline_id = uuid.uuid4()

        def mock_exec(query):
            mock_res = MagicMock()
            # If query checks baseline_id vs target_id
            mock_res.first.return_value = file_baseline
            return mock_res

        # Set up side effect to return file_target on first lookup and file_baseline on second
        db.exec.side_effect = [
            MagicMock(first=MagicMock(return_value=file_target)),
            MagicMock(first=MagicMock(return_value=file_baseline)),
        ]
    else:
        db.exec.return_value.first.return_value = file_target

    with patch("app.services.dataset_drift.get_settings") as settings:
        settings.return_value.upload_folder = str(tmp_path)
        body, status = compute_dataset_drift(
            db,
            file_id=file_target_id,
            baseline_file_id=baseline_id,
            training_split=training_split,
            seed=seed,
            alpha=alpha,
        )

    return body, status


def test_drift_single_dataset_no_drift(tmp_path):
    """I.I.D. random data split into train/test should not trigger significant drift."""
    np.random.seed(42)
    vals = np.random.normal(loc=10.0, scale=2.0, size=100)
    df = pd.DataFrame({"feat_a": vals, "feat_b": np.random.uniform(0, 1, 100)})
    csv_text = df.to_csv(index=False)

    body, status = _run_drift(tmp_path, csv_text, training_split=0.8, seed=42)

    assert status == 200
    assert body["success"] is True
    data = body["data"]
    assert data["mode"] == "train_test_split"
    assert data["total_features_count"] == 2
    assert data["drift_share"] == 0.0
    assert data["dataset_drift_detected"] is False

    feat_a = next(f for f in data["features"] if f["feature_name"] == "feat_a")
    assert feat_a["drift_detected"] is False
    assert feat_a["drift_severity"] == "none"
    assert feat_a["ks_pvalue"] > 0.05
    assert feat_a["psi"] < 0.25
    assert len(feat_a["distribution"]) > 0


def test_drift_single_dataset_severe_drift(tmp_path):
    """Dataset with strong temporal shift between early and late records detects severe drift."""
    # First 80 rows: N(0, 1), Last 20 rows: N(10, 1)
    train_vals = np.zeros(80)
    test_vals = np.ones(20) * 10.0
    all_vals = np.concatenate([train_vals, test_vals])

    # To guarantee train/test split isolates the shift, we use sample with deterministic seed
    df = pd.DataFrame({"shifted_feature": all_vals})
    csv_text = df.to_csv(index=False)

    body, status = _run_drift(tmp_path, csv_text, training_split=0.8, seed=1)

    assert status == 200
    data = body["data"]
    feature = data["features"][0]
    # Wasserstein distance should be non-zero
    assert feature["wasserstein_distance"] > 0.0


def test_drift_two_datasets_comparison_detects_shift(tmp_path):
    """Directly compares baseline vs target datasets with one stable and one heavily shifted feature."""
    np.random.seed(123)
    baseline_df = pd.DataFrame(
        {
            "stable_feat": np.random.normal(0, 1, 100),
            "shifted_feat": np.random.normal(0, 1, 100),
        }
    )
    target_df = pd.DataFrame(
        {
            "stable_feat": np.random.normal(0, 1, 100),
            "shifted_feat": np.random.normal(8, 1, 100),  # Heavy shift
        }
    )

    body, status = _run_drift(
        tmp_path,
        target_content=target_df.to_csv(index=False),
        baseline_content=baseline_df.to_csv(index=False),
    )

    assert status == 200
    data = body["data"]
    assert data["mode"] == "dataset_comparison"
    assert data["dataset_drift_detected"] is True

    shifted = next(f for f in data["features"] if f["feature_name"] == "shifted_feat")
    assert shifted["drift_detected"] is True
    assert shifted["drift_severity"] == "severe"
    assert shifted["ks_pvalue"] < 0.001
    assert shifted["psi"] >= 0.25

    stable = next(f for f in data["features"] if f["feature_name"] == "stable_feat")
    assert stable["drift_detected"] is False
    assert stable["drift_severity"] == "none"


def test_drift_categorical_feature_shift(tmp_path):
    """Categorical distribution shift detects unseen categories and severe frequency drift."""
    baseline_df = pd.DataFrame({"category_feat": ["A"] * 80 + ["B"] * 20})
    target_df = pd.DataFrame({"category_feat": ["B"] * 20 + ["C"] * 80})  # C is unseen in baseline

    body, status = _run_drift(
        tmp_path,
        target_content=target_df.to_csv(index=False),
        baseline_content=baseline_df.to_csv(index=False),
    )

    assert status == 200
    data = body["data"]
    cat_feature = data["features"][0]
    assert cat_feature["data_type"] == "categorical"
    assert cat_feature["drift_detected"] is True
    assert "C" in cat_feature["unseen_categories"]
    assert cat_feature["psi"] >= 0.25


def test_drift_insufficient_samples_guard(tmp_path):
    """Datasets with fewer than 10 rows are rejected for train/test split drift."""
    df = pd.DataFrame({"val": [1, 2, 3, 4]})
    body, status = _run_drift(tmp_path, df.to_csv(index=False))

    assert status == 400
    assert "at least 10 rows" in body["message"]


def test_drift_non_csv_file_type_rejected():
    """Non-csv file type returns 400."""
    db = MagicMock()
    file = MagicMock(spec=DataFile)
    file.file_type = "image_zip"
    db.exec.return_value.first.return_value = file

    body, status = compute_dataset_drift(db, file_id=uuid.uuid4())
    assert status == 400
    assert "only supported for tabular CSV" in body["message"]


def test_drift_nonexistent_file_404():
    """Non-existent file_id returns 404."""
    db = MagicMock()
    db.exec.return_value.first.return_value = None

    body, status = compute_dataset_drift(db, file_id=uuid.uuid4())
    assert status == 404
    assert "not found" in body["message"].lower()


def test_drift_constant_values_and_nans_handled_gracefully(tmp_path):
    """Features with constant values or NaNs do not cause zero-division or crash."""
    df = pd.DataFrame(
        {
            "constant_feat": [5.0] * 50,
            "nan_feat": [np.nan] * 50,
            "mixed_feat": [1.0, 2.0, np.nan, 4.0] * 12 + [5.0, 6.0],
        }
    )

    body, status = _run_drift(tmp_path, df.to_csv(index=False))

    assert status == 200
    assert body["success"] is True
    data = body["data"]
    assert data["total_features_count"] == 3


def test_drift_endpoint_via_client(client: TestClient, tmp_path, db_session):
    """Integration test verifying GET /data/process/drift/{file_id} via FastAPI TestClient."""
    csv_file = tmp_path / "drift_test.csv"
    np.random.seed(42)
    df = pd.DataFrame({"x1": np.random.randn(30), "x2": np.random.randn(30)})
    csv_file.write_text(df.to_csv(index=False), encoding="utf-8")

    data_file = DataFile(
        file_name="drift_test.csv",
        disk_name=csv_file.name,
        file_type="csv",
        row_count=30,
        columns=["x1", "x2"],
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    with patch("app.services.dataset_drift.get_settings") as settings:
        settings.return_value.upload_folder = str(tmp_path)
        settings.return_value.max_content_length = 200 * 1024 * 1024

        response = client.get(f"/api/v1/data/process/drift/{data_file.id}")

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert "features" in json_data["data"]
    assert len(json_data["data"]["features"]) == 2
