"""Tests for Phase 4 analysis scaffolding (Week 8) — updated for Week 10.

These tests verify that the analysis routes exist and return expected
status codes. Updated to reflect Week 10 implementation:
- confusion-matrix and feature-importance now return 200/202/500
- predictions now returns 200/500 (Week 10 implementation complete)
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus

client = TestClient(app)


@pytest.fixture
def completed_job(db_session: Session):
    """Create a completed training job for testing."""
    # Create model
    model = ModelBasic(model_name=f"test_model_{uuid4().hex[:8]}", graph_ir={})
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    # Create completed job
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


def test_analysis_routes_exist(completed_job):
    """All 3 analysis routes should exist and not return 404."""
    job_id = completed_job

    # Confusion matrix — now attempts real computation but will fail with 500
    # due to missing model file (no trained model in scaffolding test).
    # The important thing is the route exists (not 404).
    response = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")
    assert response.status_code != 404, "Route should exist"
    # Will be 500 because model file doesn't exist

    # Feature importance — will return 202 (starts background computation)
    response = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
    assert response.status_code != 404, "Route should exist"

    # Predictions — now implemented (Week 10), will return 500 due to missing model file
    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions")
    assert response.status_code != 404, "Route should exist"
    # Will be 500 because model file doesn't exist


def test_analysis_requires_completed_job(db_session: Session):
    """Analysis routes should return 400 for non-completed jobs."""
    # Create model
    model = ModelBasic(model_name=f"test_model_{uuid4().hex[:8]}", graph_ir={})
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    # Create running job
    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.RUNNING,
        hyperparams={},
        started_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    # All routes should return 400 (job not completed)
    response = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")
    assert response.status_code == 400
    assert "completed" in response.json()["detail"].lower()

    response = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
    assert response.status_code == 400

    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions")
    assert response.status_code == 400


def test_analysis_returns_404_for_nonexistent_job(db_session: Session):
    """Analysis routes should return 404 for non-existent jobs."""
    nonexistent_job_id = str(uuid4())

    # All routes should return 404
    response = client.get(f"/api/v1/model/analysis/{nonexistent_job_id}/confusion-matrix")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

    response = client.get(f"/api/v1/model/analysis/{nonexistent_job_id}/feature-importance")
    assert response.status_code == 404

    response = client.get(f"/api/v1/model/analysis/{nonexistent_job_id}/predictions")
    assert response.status_code == 404


def test_predictions_accepts_pagination_params(completed_job):
    """Predictions route should accept offset and limit query parameters."""
    job_id = completed_job

    # Test with default params - will return 500 because model file doesn't exist
    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions")
    assert response.status_code == 500  # Implementation exists but no model file

    # Test with custom params - will also return 500
    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions?offset=10&limit=50")
    assert response.status_code == 500

    # Test with invalid params (limit too high)
    response = client.get(f"/api/v1/model/analysis/{job_id}/predictions?limit=200")
    # Route declares limit: int = Query(25, ge=1, le=100), so FastAPI rejects with 422
    assert response.status_code == 422


def test_interpretability_service_predictions_implemented():
    """InterpretabilityService.get_predictions_async should exist and be callable."""
    from app.services.interpretability import InterpretabilityService

    service = InterpretabilityService()

    # Method should exist (Week 10 implementation)
    assert hasattr(service, "get_predictions_async")
    assert callable(service.get_predictions_async)


def test_analysis_cache_operations(db_session: Session):
    """AnalysisCache should be able to get and set cached results."""
    from app.services.interpretability import AnalysisCache

    # Create a completed job
    model = ModelBasic(model_name=f"test_model_{uuid4().hex[:8]}", graph_ir={})
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    cache = AnalysisCache()

    # Initially, no cache
    result = cache.get_cached(job_id, "confusion_matrix", db_session)
    assert result is None

    # Set cache
    test_data = {"matrix": [[10, 2], [1, 15]], "accuracy": 0.89}
    cache.set_cached(job_id, "confusion_matrix", test_data, db_session)

    # Retrieve cache
    result = cache.get_cached(job_id, "confusion_matrix", db_session)
    assert result is not None
    assert result["matrix"] == [[10, 2], [1, 15]]
    assert result["accuracy"] == 0.89

    # Different analysis type should not be cached
    result = cache.get_cached(job_id, "feature_importance", db_session)
    assert result is None
