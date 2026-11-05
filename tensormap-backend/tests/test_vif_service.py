"""Unit and integration tests for Variance Inflation Factor (VIF) service and router endpoint."""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.services.data_process import get_vif_metrics

FILE_ID = uuid.uuid4()
_FILE_PATH_PATCH = "app.services.data_process._get_file_path"
_FAKE_PATH = "/tmp/fake_dataset.csv"


def _make_db(file_obj):
    """Return a mock Session whose .exec().first() returns *file_obj*."""
    db = MagicMock()
    db.exec.return_value.first.return_value = file_obj
    return db


def _fake_file(name="test", ftype="csv"):
    f = MagicMock()
    f.file_name = name
    f.file_type = ftype
    f.disk_name = f"{name}.{ftype}"
    return f


class TestVIFService:
    """Unit tests for get_vif_metrics service."""

    def test_file_not_found_in_db(self):
        db = _make_db(None)
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 400
        assert body["success"] is False
        assert "doesn't exist" in body["message"]

    def test_rejects_non_csv_file(self):
        db = _make_db(_fake_file("images", "zip"))
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 400
        assert body["success"] is False
        assert "CSV" in body["message"]

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv", side_effect=FileNotFoundError("Disk read error"))
    def test_missing_file_on_disk(self, _mock_read, _mock_path):
        db = _make_db(_fake_file())
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 500
        assert body["success"] is False
        assert "File not found" in body["message"]

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_no_numeric_columns(self, mock_read, _mock_path):
        mock_read.return_value = pd.DataFrame(
            {
                "col_a": ["cat", "dog", "bird"],
                "col_b": ["red", "blue", "green"],
            }
        )
        db = _make_db(_fake_file())
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 200
        assert body["success"] is True
        assert body["data"]["vif_scores"] == []
        assert body["data"]["summary"]["total_features_evaluated"] == 0

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_less_than_two_numeric_features(self, mock_read, _mock_path):
        mock_read.return_value = pd.DataFrame(
            {
                "num_1": [1.0, 2.0, 3.0],
                "text": ["a", "b", "c"],
            }
        )
        db = _make_db(_fake_file())
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 200
        assert body["success"] is True
        assert body["data"]["vif_scores"] == []
        assert "At least 2 non-constant numeric features" in body["message"]

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_constant_columns_excluded(self, mock_read, _mock_path):
        mock_read.return_value = pd.DataFrame(
            {
                "const_col": [5.0, 5.0, 5.0, 5.0, 5.0],
                "var_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "var_2": [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        )
        db = _make_db(_fake_file())
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 200
        assert "const_col" in body["data"]["constant_columns"]
        features = [item["feature"] for item in body["data"]["vif_scores"]]
        assert "const_col" not in features
        assert "var_1" in features
        assert "var_2" in features

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_independent_features_have_low_vif(self, mock_read, _mock_path):
        np.random.seed(42)
        n = 100
        mock_read.return_value = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.normal(5, 2, n),
                "x3": np.random.normal(-3, 1.5, n),
            }
        )
        db = _make_db(_fake_file())
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 200
        assert body["success"] is True

        scores = body["data"]["vif_scores"]
        assert len(scores) == 3
        for item in scores:
            assert item["vif_numeric"] < 5.0
            assert item["severity"] == "LOW"
            assert "Healthy feature" in item["recommendation"]

        assert body["data"]["summary"]["overall_multicollinearity"] == "LOW"
        assert body["data"]["summary"]["high_vif_features"] == 0

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_perfect_collinearity(self, mock_read, _mock_path):
        x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        x2 = 2.0 * x1  # exact linear dependency
        x3 = np.array([5.0, 3.0, 9.0, 1.0, 4.0, 6.0, 8.0, 2.0])

        mock_read.return_value = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        db = _make_db(_fake_file())
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 200

        scores = {item["feature"]: item for item in body["data"]["vif_scores"]}
        assert scores["x1"]["severity"] == "PERFECT_COLLINEARITY"
        assert scores["x1"]["vif"] == "Infinity"
        assert scores["x2"]["severity"] == "PERFECT_COLLINEARITY"
        assert scores["x2"]["vif"] == "Infinity"
        assert scores["x1"]["tolerance"] == 0.0
        assert scores["x1"]["r_squared"] == 1.0

        assert body["data"]["summary"]["overall_multicollinearity"] == "HIGH"
        assert body["data"]["summary"]["high_vif_features"] >= 2

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_high_collinearity_linear_combination(self, mock_read, _mock_path):
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        # Strong linear combination with very tiny noise
        x3 = 0.8 * x1 + 0.6 * x2 + np.random.normal(0, 0.05, n)

        mock_read.return_value = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        db = _make_db(_fake_file())
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 200

        scores = {item["feature"]: item for item in body["data"]["vif_scores"]}
        assert scores["x3"]["vif_numeric"] >= 10.0
        assert scores["x3"]["severity"] == "HIGH"
        assert "High collinearity detected" in scores["x3"]["recommendation"]
        assert body["data"]["summary"]["overall_multicollinearity"] == "HIGH"

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_nan_values_handled_gracefully(self, mock_read, _mock_path):
        mock_read.return_value = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
                "b": [2.0, np.nan, 3.0, 4.0, 5.0, 6.0],
                "c": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            }
        )
        db = _make_db(_fake_file())
        body, status = get_vif_metrics(db, file_id=FILE_ID)
        assert status == 200
        assert len(body["data"]["vif_scores"]) == 3


class TestVIFEndpoint:
    """Integration test for GET /data/process/vif/{file_id} endpoint."""

    @patch("app.routers.data_process.get_vif_metrics")
    def test_get_vif_endpoint(self, mock_get_vif, client: TestClient):
        mock_get_vif.return_value = (
            {
                "success": True,
                "message": "Variance Inflation Factor (VIF) computed successfully",
                "data": {
                    "vif_scores": [
                        {
                            "feature": "col_1",
                            "vif": 2.5,
                            "vif_numeric": 2.5,
                            "tolerance": 0.4,
                            "r_squared": 0.6,
                            "severity": "LOW",
                            "recommendation": "Healthy feature with low multicollinearity.",
                        }
                    ],
                    "constant_columns": [],
                    "summary": {
                        "overall_multicollinearity": "LOW",
                        "condition_index": 3.2,
                        "total_features_evaluated": 1,
                        "high_vif_features": 0,
                    },
                },
            },
            200,
        )

        file_id = uuid.uuid4()
        response = client.get(f"/api/v1/data/process/vif/{file_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"]["summary"]["overall_multicollinearity"] == "LOW"
        assert body["data"]["vif_scores"][0]["feature"] == "col_1"
