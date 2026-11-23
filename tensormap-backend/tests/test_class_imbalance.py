"""Unit and integration tests for Class Imbalance Inspector and Cost-Sensitive Weight Calculator."""

import math
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.models import DataFile, DataProcess
from app.services.data_process import (
    compute_class_imbalance,
    get_class_imbalance_service,
)

# ---------------------------------------------------------------------------
# compute_class_imbalance unit tests
# ---------------------------------------------------------------------------


def test_compute_class_imbalance_balanced_binary():
    """Verify perfectly balanced binary dataset produces ratio 1.0, severity balanced, and weights 1.0."""
    series = pd.Series([0] * 50 + [1] * 50)
    result = compute_class_imbalance(series, target_name="label")

    assert result["target_field"] == "label"
    assert result["total_samples"] == 100
    assert result["null_count"] == 0
    assert result["num_classes"] == 2
    assert result["imbalance_ratio"] == 1.0
    assert result["severity"] == "balanced"
    assert result["class_weights"]["0"] == 1.0
    assert result["class_weights"]["1"] == 1.0
    assert math.isclose(result["gini_impurity"], 0.5, abs_tol=1e-3)
    assert math.isclose(result["normalized_entropy"], 1.0, abs_tol=1e-3)
    assert "well-balanced" in result["recommendation"]


def test_compute_class_imbalance_mild_imbalance():
    """Verify mild imbalance (e.g. 2:1 ratio) categorizes as mild_imbalance."""
    series = pd.Series([0] * 66 + [1] * 34)
    result = compute_class_imbalance(series, target_name="target")

    assert result["severity"] == "mild_imbalance"
    assert pytest.approx(result["imbalance_ratio"], 0.05) == 1.94
    assert result["majority_class"] == "0"
    assert result["minority_class"] == "1"
    assert "Mild class imbalance detected" in result["recommendation"]


def test_compute_class_imbalance_moderate_imbalance():
    """Verify moderate imbalance (e.g. 4:1 ratio) categorizes as moderate_imbalance."""
    series = pd.Series([0] * 80 + [1] * 20)
    result = compute_class_imbalance(series, target_name="churn")

    assert result["severity"] == "moderate_imbalance"
    assert result["imbalance_ratio"] == 4.0
    assert result["majority_class"] == "0"
    assert result["minority_class"] == "1"
    # w_0 = 100 / (2 * 80) = 0.625, w_1 = 100 / (2 * 20) = 2.5
    assert result["class_weights"]["0"] == 0.625
    assert result["class_weights"]["1"] == 2.5
    assert "Moderate class imbalance detected" in result["recommendation"]


def test_compute_class_imbalance_severe_imbalance():
    """Verify severe imbalance (e.g. 95:5 ratio) categorizes as severe_imbalance with cost-sensitive weights."""
    series = pd.Series(["legit"] * 950 + ["fraud"] * 50)
    result = compute_class_imbalance(series, target_name="is_fraud")

    assert result["severity"] == "severe_imbalance"
    assert result["imbalance_ratio"] == 19.0
    assert result["majority_class"] == "legit"
    assert result["minority_class"] == "fraud"
    # w_legit = 1000 / (2 * 950) = 0.5263, w_fraud = 1000 / (2 * 50) = 10.0
    assert pytest.approx(result["class_weights"]["legit"], 1e-3) == 0.5263
    assert result["class_weights"]["fraud"] == 10.0
    assert "Severe class imbalance detected" in result["recommendation"]
    assert result["gini_impurity"] < 0.1


def test_compute_class_imbalance_multiclass():
    """Verify 3-class distribution weights and entropy."""
    series = pd.Series(["A"] * 600 + ["B"] * 300 + ["C"] * 100)
    result = compute_class_imbalance(series, target_name="category")

    assert result["num_classes"] == 3
    assert result["total_samples"] == 1000
    assert result["imbalance_ratio"] == 6.0
    assert result["severity"] == "moderate_imbalance"
    # w_A = 1000 / (3 * 600) = 0.5556, w_B = 1000 / (3 * 300) = 1.1111, w_C = 1000 / (3 * 100) = 3.3333
    assert pytest.approx(result["class_weights"]["A"], 1e-3) == 0.5556
    assert pytest.approx(result["class_weights"]["B"], 1e-3) == 1.1111
    assert pytest.approx(result["class_weights"]["C"], 1e-3) == 3.3333
    assert len(result["classes"]) == 3


def test_compute_class_imbalance_with_nulls():
    """Verify null values are excluded from valid sample count and recorded in null_count."""
    series = pd.Series([0] * 50 + [1] * 50 + [None] * 10 + [np.nan] * 5)
    result = compute_class_imbalance(series, target_name="label")

    assert result["total_samples"] == 100
    assert result["null_count"] == 15
    assert result["severity"] == "balanced"


def test_compute_class_imbalance_rejects_empty_or_all_null():
    """Verify ValueError is raised when series contains no valid samples."""
    series = pd.Series([None, np.nan])
    with pytest.raises(ValueError, match="contains no valid"):
        compute_class_imbalance(series, target_name="target")


def test_compute_class_imbalance_rejects_single_class():
    """Verify ValueError is raised when target column has fewer than 2 classes."""
    series = pd.Series(["only_one"] * 20)
    with pytest.raises(ValueError, match="must have at least 2 distinct classes"):
        compute_class_imbalance(series, target_name="target")


def test_compute_class_imbalance_rejects_high_cardinality_continuous():
    """Verify ValueError is raised when target has > 100 distinct unique classes."""
    series = pd.Series(range(105))
    with pytest.raises(ValueError, match="discrete classification targets"):
        compute_class_imbalance(series, target_name="price")


# ---------------------------------------------------------------------------
# get_class_imbalance_service tests
# ---------------------------------------------------------------------------


def _mock_data_file(tmp_path, csv_content, file_name="imbalance_test", file_type="csv"):
    disk_name = f"{file_name}.csv" if file_type == "csv" else f"{file_name}.zip"
    (tmp_path / disk_name).write_text(csv_content)
    f = MagicMock(spec=DataFile)
    f.file_name = file_name
    f.file_type = file_type
    f.disk_name = disk_name
    f.columns = ["feat1", "label"]
    return f


def test_service_returns_404_when_file_not_found():
    db = MagicMock()
    db.exec.return_value.first.return_value = None

    body, status = get_class_imbalance_service(db, uuid.uuid4())
    assert status == 404
    assert body["success"] is False
    assert "not found" in body["message"]


def test_service_returns_400_for_non_csv_file(tmp_path):
    fake_file = _mock_data_file(tmp_path, "dummy", file_type="zip")
    db = MagicMock()
    db.exec.return_value.first.return_value = fake_file

    body, status = get_class_imbalance_service(db, uuid.uuid4())
    assert status == 400
    assert body["success"] is False
    assert "Only CSV files" in body["message"]


def test_service_returns_400_when_no_target_specified_or_configured(tmp_path):
    fake_file = _mock_data_file(tmp_path, "feat1,label\n1,0\n2,1\n")
    db = MagicMock()
    # First query for DataFile, second query for DataProcess returns None
    db.exec.return_value.first.side_effect = [fake_file, None]

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, status = get_class_imbalance_service(db, uuid.uuid4(), target_field=None)

    assert status == 400
    assert body["success"] is False
    assert "Target field was not specified" in body["message"]


def test_service_resolves_target_from_dataprocess_db(tmp_path):
    csv_data = "feat1,target_col\n1,0\n2,1\n3,0\n4,0\n"
    fake_file = _mock_data_file(tmp_path, csv_data)
    fake_dp = MagicMock(spec=DataProcess)
    fake_dp.target = "target_col"

    db = MagicMock()
    db.exec.return_value.first.side_effect = [fake_file, fake_dp]

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, status = get_class_imbalance_service(db, uuid.uuid4(), target_field=None)

    assert status == 200
    assert body["success"] is True
    assert body["data"]["target_field"] == "target_col"
    assert body["data"]["total_samples"] == 4


def test_service_returns_400_when_target_not_in_columns(tmp_path):
    csv_data = "feat1,label\n1,0\n2,1\n"
    fake_file = _mock_data_file(tmp_path, csv_data)
    db = MagicMock()
    db.exec.return_value.first.return_value = fake_file

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, status = get_class_imbalance_service(db, uuid.uuid4(), target_field="non_existent")

    assert status == 400
    assert body["success"] is False
    assert "not found in dataset columns" in body["message"]


def test_service_returns_500_when_file_missing_on_disk(tmp_path):
    fake_file = MagicMock(spec=DataFile)
    fake_file.file_name = "ghost"
    fake_file.file_type = "csv"
    fake_file.disk_name = "ghost.csv"
    db = MagicMock()
    db.exec.return_value.first.return_value = fake_file

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, status = get_class_imbalance_service(db, uuid.uuid4(), target_field="label")

    assert status == 500
    assert body["success"] is False
    assert "File not found" in body["message"]


# ---------------------------------------------------------------------------
# FastAPI Router integration tests
# ---------------------------------------------------------------------------


def test_api_route_class_imbalance_both_paths(client: TestClient, db_session, tmp_path):
    """Test both GET /api/v1/data/process/{file_id}/class-imbalance and class-imbalance/{file_id}."""
    file_id = uuid.uuid4()
    csv_content = "feature_a,label\n0.1,0\n0.2,0\n0.3,0\n0.4,1\n"
    disk_name = f"{file_id}.csv"
    (tmp_path / disk_name).write_text(csv_content)

    data_file = DataFile(
        id=file_id,
        file_name="imbalance_api_test.csv",
        disk_name=disk_name,
        file_type="csv",
        columns=["feature_a", "label"],
    )
    db_session.add(data_file)
    db_session.commit()

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)

        # Path 1: /api/v1/data/process/{file_id}/class-imbalance?target_field=label
        resp1 = client.get(f"/api/v1/data/process/{file_id}/class-imbalance?target_field=label")
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["success"] is True
        assert data1["data"]["target_field"] == "label"
        assert data1["data"]["total_samples"] == 4
        assert data1["data"]["imbalance_ratio"] == 3.0
        assert data1["data"]["severity"] == "moderate_imbalance"
        assert "class_weights" in data1["data"]
        assert data1["data"]["class_weights"]["0"] == 0.6667
        assert data1["data"]["class_weights"]["1"] == 2.0

        # Path 2: /api/v1/data/process/class-imbalance/{file_id}?target_field=label
        resp2 = client.get(f"/api/v1/data/process/class-imbalance/{file_id}?target_field=label")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["success"] is True
        assert data2["data"]["target_field"] == "label"
        assert data2["data"]["imbalance_ratio"] == 3.0

        # Path 3: /api/v1/data/process/imbalance/{file_id}?target_field=label
        resp3 = client.get(f"/api/v1/data/process/imbalance/{file_id}?target_field=label")
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["success"] is True
        assert data3["data"]["target_field"] == "label"
        assert data3["data"]["imbalance_ratio"] == 3.0


def test_compute_class_imbalance_rejects_continuous_float_target():
    """Verify ValueError is raised when series contains continuous float regression values."""
    series = pd.Series([0.15, 0.28, 0.42, 0.99, 1.34, 2.56, 3.88])
    with pytest.raises(ValueError, match="continuous regression variable"):
        compute_class_imbalance(series, target_name="continuous_target")


def test_compute_class_imbalance_accepts_integral_float_target_with_nans():
    """Verify float series containing integral numbers and NaNs is treated as discrete classification."""
    series = pd.Series([0.0] * 50 + [1.0] * 50 + [np.nan] * 5)
    result = compute_class_imbalance(series, target_name="float_labels")

    assert result["total_samples"] == 100
    assert result["null_count"] == 5
    assert result["num_classes"] == 2
    assert result["severity"] == "balanced"
    assert "0" in result["class_weights"]
    assert "1" in result["class_weights"]
