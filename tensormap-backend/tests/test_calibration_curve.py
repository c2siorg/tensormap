"""Tests for probability calibration curves, reliability diagrams, and Brier score analysis."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from app.services.interpretability import InterpretabilityService
from tests.fixtures.create_test_model import (
    create_test_training_job,
)

client = TestClient(app)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def export_dir(tmp_path):
    """Provide a temporary export directory."""
    return tmp_path / "exports"


@pytest.fixture
def trained_job(db_session: Session, export_dir: Path):
    """Create a completed training job with a trained 3-class model."""
    job_id, X_test, y_test, feature_names, model = create_test_training_job(
        session=db_session,
        export_dir=export_dir,
        model_name="test_clf_calib_model",
        num_features=4,
        num_classes=3,
        epochs=5,
    )
    return {
        "job_id": job_id,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "model": model,
        "export_dir": export_dir,
    }


# ------------------------------------------------------------------
# Unit & Integration Tests
# ------------------------------------------------------------------


def test_calibration_curve_multiclass_computation(trained_job, db_session):
    """Multi-class model produces per-class reliability points, ECE, MCE, and Brier scores."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [f"class_{i}" for i in range(3)]

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
    ):
        result = service._compute_calibration_curve_sync(job_id, db_session, n_bins=10)

    assert result["n_bins"] == 10
    assert result["is_binary"] is False
    assert len(result["classes"]) == 3
    assert result["n_samples"] == len(y_test)

    assert result["macro_brier_score"] is not None
    assert 0.0 <= result["macro_brier_score"] <= 1.0
    assert result["macro_ece"] is not None
    assert 0.0 <= result["macro_ece"] <= 1.0
    assert result["macro_mce"] is not None
    assert 0.0 <= result["macro_mce"] <= 1.0

    for item in result["classes"]:
        assert item["insufficient_data"] is False
        assert 0.0 <= item["brier_score"] <= 1.0
        assert 0.0 <= item["ece"] <= 1.0
        assert 0.0 <= item["mce"] <= 1.0
        assert len(item["points"]) == 10
        assert len(item["histogram"]) == 10

        # Verify sample conservation across histogram bins
        total_hist_samples = sum(h["count"] for h in item["histogram"])
        assert total_hist_samples == result["n_samples"]

        # Verify point coordinates
        for pt in item["points"]:
            assert 0.0 <= pt["mean_predicted"] <= 1.0
            if pt["fraction_positives"] is not None:
                assert 0.0 <= pt["fraction_positives"] <= 1.0
                assert pt["error"] == pytest.approx(abs(pt["fraction_positives"] - pt["mean_predicted"]), abs=1e-3)


def test_calibration_curve_binary_computation(db_session, export_dir):
    """Binary model produces valid calibration curve with is_binary=True."""
    job_id, X_test, y_test, feature_names, _ = create_test_training_job(
        session=db_session,
        export_dir=export_dir,
        model_name="test_binary_calib_model",
        num_features=4,
        num_classes=2,
        epochs=5,
    )
    class_names = ["neg", "pos"]
    service = InterpretabilityService()

    # Mock model that makes sharp, highly calibrated predictions
    mock_model = MagicMock()
    probs = np.zeros((len(y_test), 2))
    for i, label in enumerate(y_test):
        probs[i, int(label)] = 0.95
        probs[i, 1 - int(label)] = 0.05
    mock_model.predict.return_value = probs

    with (
        patch.object(
            service,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch("app.services.interpretability.EXPORTS_BASE", export_dir),
        patch("tensorflow.keras.models.load_model", return_value=mock_model),
    ):
        result = service._compute_calibration_curve_sync(job_id, db_session, n_bins=10)

    assert result["is_binary"] is True
    assert len(result["classes"]) == 2
    # Sharply calibrated model has low Brier score and low ECE
    assert result["macro_brier_score"] < 0.10
    assert result["macro_ece"] < 0.15


def test_calibration_curve_single_class_fallback(db_session):
    """When a class has 0 positives in test data, handles gracefully with insufficient_data=True."""
    service = InterpretabilityService()
    job_id = "job-single-class-calib"

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.2], [0.9, 0.1]])

    # Both samples are class 0 (class 1 has zero positive samples)
    with (
        patch.object(
            service,
            "_load_test_data",
            return_value=(np.zeros((2, 4)), np.array([0, 0]), ["class0", "class1"]),
        ),
        patch("tensorflow.keras.models.load_model", return_value=mock_model),
    ):
        result = service._compute_calibration_curve_sync(job_id, db_session)

    assert result["classes"][0]["insufficient_data"] is True
    assert result["classes"][0]["brier_score"] is None
    assert result["classes"][0]["ece"] is None
    assert result["classes"][1]["insufficient_data"] is True


def test_calibration_curve_caching(trained_job, db_session):
    """Subsequent calls to compute_calibration_curve_async hit cache without recomputing."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [f"class_{i}" for i in range(3)]
    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
    ):
        result1 = asyncio.run(service.compute_calibration_curve_async(job_id, db_session))

        with patch.object(service, "_compute_calibration_curve_sync") as mock_sync:
            result2 = asyncio.run(service.compute_calibration_curve_async(job_id, db_session))
            mock_sync.assert_not_called()

    assert result1["macro_brier_score"] == result2["macro_brier_score"]
    assert result1["macro_ece"] == result2["macro_ece"]


def test_calibration_curve_endpoint_returns_200(trained_job, db_session):
    """GET /api/v1/model/analysis/{job_id}/calibration-curve returns 200 with calibration data."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [f"class_{i}" for i in range(3)]
    from app.routers.analysis import _service

    with (
        patch("app.routers.analysis.get_analysis_type", return_value="classification"),
        patch.object(_service, "cache") as mock_cache,
        patch.object(_service, "_load_test_data", return_value=(X_test, y_test, class_names)),
        patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
    ):
        mock_cache.get_cached.return_value = None
        mock_cache.set_cached.return_value = None

        resp = client.get(f"/api/v1/model/analysis/{job_id}/calibration-curve?n_bins=8")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is False
        assert "classes" in data
        assert "macro_brier_score" in data
        assert data["n_bins"] == 8

        # Cache hit on second call
        mock_cache.get_cached.return_value = data
        resp2 = client.get(f"/api/v1/model/analysis/{job_id}/calibration-curve")
        assert resp2.status_code == 200
        assert resp2.json()["cached"] is True


def test_calibration_curve_endpoint_rejects_regression_model():
    """GET /api/v1/model/analysis/{job_id}/calibration-curve rejects regression models with 400."""
    with (
        patch("app.routers.analysis._verify_job_completed"),
        patch("app.routers.analysis.get_analysis_type", return_value="regression"),
    ):
        resp = client.get("/api/v1/model/analysis/reg-job-id/calibration-curve")
        assert resp.status_code == 400
        assert "classification" in resp.json()["detail"].lower()


def test_calibration_curve_endpoint_rejects_pending_job():
    """GET /api/v1/model/analysis/{job_id}/calibration-curve rejects non-completed jobs."""
    from fastapi import HTTPException

    with patch(
        "app.routers.analysis._verify_job_completed",
        side_effect=HTTPException(status_code=400, detail="Training job is not yet completed"),
    ):
        resp = client.get("/api/v1/model/analysis/pending-job-id/calibration-curve")
        assert resp.status_code == 400
        assert "not yet completed" in resp.json()["detail"]


def test_calibration_curve_endpoint_404_for_unknown_job():
    """GET /api/v1/model/analysis/{job_id}/calibration-curve returns 404 for unknown job."""
    from fastapi import HTTPException

    with patch(
        "app.routers.analysis._verify_job_completed",
        side_effect=HTTPException(status_code=404, detail="Training job not found"),
    ):
        resp = client.get("/api/v1/model/analysis/00000000-0000-0000-0000-000000000000/calibration-curve")
        assert resp.status_code == 404
