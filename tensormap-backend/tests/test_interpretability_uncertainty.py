"""Tests for prediction uncertainty and entropy analysis studio."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest
from sqlmodel import Session

from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.interpretability import InterpretabilityService
from app.shared.enums import ProblemType
from tests.fixtures.create_test_model import create_test_training_job


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
        model_name="test_clf_model",
        num_features=4,
        num_classes=3,
        epochs=5,
    )
    return {
        "job_id": job_id,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "class_names": [str(i) for i in range(3)],
        "model": model,
        "export_dir": export_dir,
    }


@pytest.fixture
def regression_job(db_session: Session):
    """Create a completed training job configured as regression."""
    model = ModelBasic(
        model_name="test_regression_model",
        model_type=ProblemType.REGRESSION,
        graph_ir={},
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    return job_id


@pytest.fixture
def pending_job(db_session: Session):
    """Create a pending (not completed) training job."""
    model = ModelBasic(model_name="test_pending_model", graph_ir={})
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.PENDING,
        hyperparams={},
    )
    db_session.add(job)
    db_session.commit()

    return job_id


class TestPredictionUncertainty:
    def test_uncertainty_math_properties(self):
        """Verify mathematical bounds of entropy and margin uncertainty."""
        # 2-class uniform: p = [0.5, 0.5] -> normalized entropy must be 1.0, margin uncertainty 1.0
        p_uniform = np.array([[0.5, 0.5]])
        eps = 1e-12
        entropy_raw = -np.sum(p_uniform * np.log2(p_uniform + eps), axis=1)
        normalized = entropy_raw / np.log2(2)
        assert normalized[0] == pytest.approx(1.0, rel=1e-3)

        sorted_probs = np.sort(p_uniform, axis=1)
        margin = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])
        assert margin[0] == pytest.approx(1.0)

        # Completely certain: p = [1.0, 0.0] -> entropy 0.0, margin uncertainty 0.0
        p_certain = np.array([[1.0, 0.0]])
        entropy_certain = -np.sum(p_certain * np.log2(p_certain + eps), axis=1)
        normalized_certain = entropy_certain / np.log2(2)
        assert normalized_certain[0] == pytest.approx(0.0, abs=1e-5)

        sorted_certain = np.sort(p_certain, axis=1)
        margin_certain = 1.0 - (sorted_certain[:, -1] - sorted_certain[:, -2])
        assert margin_certain[0] == pytest.approx(0.0, abs=1e-5)

    def test_compute_prediction_uncertainty_sync(self, db_session: Session, trained_job):
        job_id = trained_job["job_id"]
        service = InterpretabilityService()

        with (
            patch.object(
                service,
                "_load_test_data",
                return_value=(trained_job["X_test"], trained_job["y_test"], trained_job["class_names"]),
            ),
            patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
        ):
            result = service._compute_prediction_uncertainty_sync(job_id, db_session, top_n_ambiguous=5)

        assert result["analysis_type"] == "classification"
        assert result["n_samples"] > 0
        assert result["num_classes"] == 3
        assert len(result["class_names"]) == 3

        # Check summary metrics
        summary = result["summary"]
        assert 0.0 <= summary["mean_entropy"] <= 1.0
        assert 0.0 <= summary["median_entropy"] <= 1.0
        assert 0.0 <= summary["mean_margin_uncertainty"] <= 1.0
        assert 0.0 <= summary["mean_least_confidence"] <= 1.0

        # Check tiers sum to total samples
        tiers = result["tiers"]
        total_tier_samples = tiers["low"]["count"] + tiers["moderate"]["count"] + tiers["high"]["count"]
        assert total_tier_samples == result["n_samples"]

        # Check histogram
        assert len(result["histogram"]) == 10
        total_hist_count = sum(bin_info["count"] for bin_info in result["histogram"])
        assert total_hist_count == result["n_samples"]

        # Check top ambiguous samples
        top_samples = result["top_ambiguous_samples"]
        assert len(top_samples) <= 5
        # Verify sorted by entropy descending
        entropies = [s["entropy"] for s in top_samples]
        assert entropies == sorted(entropies, reverse=True)

    @pytest.mark.asyncio
    async def test_compute_prediction_uncertainty_async_and_caching(self, db_session: Session, trained_job):
        job_id = trained_job["job_id"]
        service = InterpretabilityService()

        with (
            patch.object(
                service,
                "_load_test_data",
                return_value=(trained_job["X_test"], trained_job["y_test"], trained_job["class_names"]),
            ),
            patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
        ):
            # First call: computes and caches
            res1 = await service.compute_prediction_uncertainty_async(job_id, db_session, top_n_ambiguous=10)
            cached_val = service.cache.get_cached(job_id, "uncertainty_10", db_session)
            assert cached_val is not None
            assert cached_val["summary"] == res1["summary"]

            # Second call: returns cached without recomputation
            with patch.object(service, "_compute_prediction_uncertainty_sync") as mock_sync:
                res2 = await service.compute_prediction_uncertainty_async(job_id, db_session, top_n_ambiguous=10)
                mock_sync.assert_not_called()
                assert res2 == cached_val

    def test_uncertainty_endpoint_200(self, client, trained_job):
        job_id = trained_job["job_id"]

        with (
            patch(
                "app.routers.analysis._service._load_test_data",
                return_value=(trained_job["X_test"], trained_job["y_test"], trained_job["class_names"]),
            ),
            patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
        ):
            response = client.get(f"/api/v1/model/analysis/{job_id}/uncertainty?top_n=8")

        assert response.status_code == 200
        body = response.json()
        assert body["cached"] is False
        assert "summary" in body
        assert "tiers" in body
        assert "histogram" in body
        assert len(body["top_ambiguous_samples"]) <= 8

        # Calling again should return cached: True
        with patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]):
            res_cached = client.get(f"/api/v1/model/analysis/{job_id}/uncertainty?top_n=8")
        assert res_cached.status_code == 200
        assert res_cached.json()["cached"] is True

    def test_uncertainty_endpoint_rejects_regression_model(self, client, regression_job):
        response = client.get(f"/api/v1/model/analysis/{regression_job}/uncertainty")
        assert response.status_code == 400
        assert "only available for classification" in response.json()["detail"]

    def test_uncertainty_endpoint_rejects_pending_job(self, client, pending_job):
        response = client.get(f"/api/v1/model/analysis/{pending_job}/uncertainty")
        assert response.status_code == 400
        assert "only available for completed jobs" in response.json()["detail"]

    def test_uncertainty_endpoint_nonexistent_job_404(self, client, db_session: Session):
        response = client.get("/api/v1/model/analysis/00000000-0000-0000-0000-000000000000/uncertainty")
        assert response.status_code == 404
