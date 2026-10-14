"""Tests for Week 9 interpretability features.

Tests cover:
- Confusion matrix computation and caching (tests 1-8)
- Feature importance with 202/200 polling and guardrails (tests 9-14)

Uses test fixtures from tests/fixtures/create_test_model.py that create
synthetic datasets and trained models without requiring the full data pipeline.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.interpretability import AnalysisCache, InterpretabilityService
from app.shared.enums import ProblemType
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


# ------------------------------------------------------------------
# Test 1: test_confusion_matrix_shape
# ------------------------------------------------------------------


@pytest.mark.slow
def test_confusion_matrix_shape(trained_job, db_session):
    """3-class classification → 3×3 confusion matrix."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [str(i) for i in range(3)]

    service = InterpretabilityService()

    # Mock _load_test_data to return our fixture data
    with (
        patch.object(
            service,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        result = service._compute_confusion_matrix_sync(job_id, db_session)

    cm = result["confusion_matrix"]
    assert len(cm) == 3, f"Expected 3 rows, got {len(cm)}"
    for row in cm:
        assert len(row) == 3, f"Expected 3 columns, got {len(row)}"
    assert result["analysis_type"] == "classification"


# ------------------------------------------------------------------
# Test 2: test_confusion_matrix_cached_on_second_call
# ------------------------------------------------------------------


@pytest.mark.slow
def test_confusion_matrix_cached_on_second_call(trained_job, db_session):
    """Call twice, verify DB read on 2nd call (cache hit)."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [str(i) for i in range(3)]

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        # First call — should compute
        result1 = asyncio.get_event_loop().run_until_complete(
            service.compute_confusion_matrix_async(job_id, db_session)
        )

        # Second call — should hit cache (no _compute call)
        with patch.object(service, "_compute_confusion_matrix_sync") as mock_compute:
            result2 = asyncio.get_event_loop().run_until_complete(
                service.compute_confusion_matrix_async(job_id, db_session)
            )
            mock_compute.assert_not_called()

    assert result1["confusion_matrix"] == result2["confusion_matrix"]


# ------------------------------------------------------------------
# Test 3: test_classification_report_has_precision_recall
# ------------------------------------------------------------------


@pytest.mark.slow
def test_classification_report_has_precision_recall(trained_job, db_session):
    """Classification report has precision, recall, f1 for each class."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [str(i) for i in range(3)]

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        result = service._compute_confusion_matrix_sync(job_id, db_session)

    report = result["classification_report"]

    # Each class should have precision, recall, f1-score
    for class_name in class_names:
        assert class_name in report, f"Class {class_name} not in report"
        assert "precision" in report[class_name]
        assert "recall" in report[class_name]
        assert "f1-score" in report[class_name]

    # Averages should be present
    assert "accuracy" in report
    assert "macro avg" in report
    assert "weighted avg" in report


# ------------------------------------------------------------------
# Test 4: test_regression_returns_residuals
# ------------------------------------------------------------------


@pytest.mark.slow
def test_regression_returns_residuals(trained_job, db_session):
    """Regression model → result has residuals array."""
    job_id = trained_job["job_id"]
    # Use the same model but pretend it's regression for testing
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"].astype(float)

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data",
            return_value=(X_test, y_test, []),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        result = service._compute_regression_analysis_sync(job_id, db_session)

    assert "residuals" in result
    assert "y_pred" in result
    assert "y_true" in result
    assert "mae" in result
    assert "mse" in result
    assert result["analysis_type"] == "regression"
    assert len(result["residuals"]) == len(y_test)


# ------------------------------------------------------------------
# Test 5: test_confusion_matrix_not_available_for_regression
# ------------------------------------------------------------------


def test_confusion_matrix_not_available_for_regression(regression_job, db_session):
    """Endpoint returns 400 for regression job."""
    response = client.get(f"/api/v1/model/analysis/{regression_job}/confusion-matrix")
    assert response.status_code == 400
    assert "regression" in response.json()["detail"].lower()


# ------------------------------------------------------------------
# Test 6: test_get_confusion_matrix_endpoint_200
# ------------------------------------------------------------------


@pytest.mark.slow
def test_get_confusion_matrix_endpoint_200(trained_job, db_session):
    """GET /analysis/{job_id}/confusion-matrix → 200."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [str(i) for i in range(3)]

    with (
        patch.object(
            InterpretabilityService,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        response = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")

    assert response.status_code == 200
    data = response.json()
    assert "confusion_matrix" in data
    assert "classification_report" in data
    assert "overall_accuracy" in data
    assert data["cached"] is False


# ------------------------------------------------------------------
# Test 7: test_get_confusion_matrix_endpoint_cached
# ------------------------------------------------------------------


@pytest.mark.slow
def test_get_confusion_matrix_endpoint_cached(trained_job, db_session):
    """Second request returns same data (from cache)."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [str(i) for i in range(3)]

    with (
        patch.object(
            InterpretabilityService,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        # First request — computes
        response1 = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")
        assert response1.status_code == 200
        assert response1.json()["cached"] is False

        # Second request — from cache
        response2 = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")
        assert response2.status_code == 200
        assert response2.json()["cached"] is True

    # Same confusion matrix data
    assert response1.json()["confusion_matrix"] == response2.json()["confusion_matrix"]


# ------------------------------------------------------------------
# Test 8: test_analysis_requires_completed_job
# ------------------------------------------------------------------


def test_analysis_requires_completed_job(pending_job, db_session):
    """Analysis on pending job → 400 'training not complete'."""
    response = client.get(f"/api/v1/model/analysis/{pending_job}/confusion-matrix")
    assert response.status_code == 400
    assert "completed" in response.json()["detail"].lower()


# ------------------------------------------------------------------
# Test 9: test_feature_importance_returns_202_first
# ------------------------------------------------------------------


def test_feature_importance_returns_202_first(trained_job, db_session):
    """First call → 202 with {status: 'computing'}."""
    job_id = trained_job["job_id"]

    # Mock the background task so it doesn't actually run
    with patch("app.services.interpretability.asyncio.create_task"):
        response = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")

    assert response.status_code == 202
    assert response.json()["status"] == "computing"


# ------------------------------------------------------------------
# Test 10: test_feature_importance_returns_200_after_compute
# ------------------------------------------------------------------


@pytest.mark.slow
def test_feature_importance_returns_200_after_compute(trained_job, db_session):
    """After background task completes → 200 with data."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    feature_names = trained_job["feature_names"]

    # Pre-populate cache as if background task completed
    cache = AnalysisCache()
    cached_result = {
        "features": feature_names,
        "importances_mean": [0.1, 0.2, 0.3, 0.05],
        "importances_std": [0.01, 0.02, 0.03, 0.005],
        "analysis_type": "feature_importance",
        "n_samples_used": len(X_test),
        "n_repeats": 10,
    }
    cache.set_cached(job_id, "feature_importance", cached_result, db_session)

    response = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
    assert response.status_code == 200
    data = response.json()
    assert data["analysis_type"] == "feature_importance"
    assert "features" in data
    assert "importances_mean" in data


# ------------------------------------------------------------------
# Test 11: test_feature_importance_capped_at_1000_samples
# ------------------------------------------------------------------


@pytest.mark.slow
def test_feature_importance_capped_at_1000_samples(trained_job, db_session):
    """Dataset with 2000 rows → result uses 1000."""
    job_id = trained_job["job_id"]

    # Create a large dataset (2000 samples, 4 features)
    np.random.seed(42)
    X_large = np.random.randn(2000, 4).astype(np.float32)
    y_large = np.random.randint(0, 3, 2000)
    feature_names = ["f0", "f1", "f2", "f3"]

    # Create a mock Keras model that accepts any input shape
    mock_model = MagicMock()
    mock_model.predict = MagicMock(side_effect=lambda X, **kwargs: np.random.rand(len(X), 3).astype(np.float32))

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data_with_features",
            return_value=(X_large, y_large, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
        patch(
            "tensorflow.keras.models.load_model",
            return_value=mock_model,
        ),
    ):
        result = service._compute_feature_importance_sync(job_id, db_session)

    assert result["n_samples_used"] == 1000


# ------------------------------------------------------------------
# Test 12: test_feature_importance_capped_at_20_features
# ------------------------------------------------------------------


@pytest.mark.slow
def test_feature_importance_capped_at_20_features(db_session, export_dir):
    """30-feature dataset → result has 20 features."""
    # Create a model (only need DB records, actual model will be mocked)
    job_id, X_test, y_test, feature_names, model = create_test_training_job(
        session=db_session,
        export_dir=export_dir,
        model_name="test_30feat_model",
        num_features=30,
        num_classes=3,
        epochs=3,
    )

    # Generate feature names for 30 features
    feature_names_30 = [f"feature_{i}" for i in range(30)]

    # Mock Keras model that handles any input shape
    mock_model = MagicMock()
    mock_model.predict = MagicMock(side_effect=lambda X, **kwargs: np.random.rand(len(X), 3).astype(np.float32))

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data_with_features",
            return_value=(X_test, y_test, feature_names_30),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            export_dir,
        ),
        patch(
            "tensorflow.keras.models.load_model",
            return_value=mock_model,
        ),
    ):
        result = service._compute_feature_importance_sync(job_id, db_session)

    assert len(result["features"]) == 20
    assert len(result["importances_mean"]) == 20


# ------------------------------------------------------------------
# Test 13: test_feature_importance_cached_on_second_call
# ------------------------------------------------------------------


def test_feature_importance_cached_on_second_call(trained_job, db_session):
    """Second 200 response uses cache."""
    job_id = trained_job["job_id"]
    feature_names = trained_job["feature_names"]

    # Pre-populate cache
    cache = AnalysisCache()
    cached_result = {
        "features": feature_names,
        "importances_mean": [0.1, 0.2, 0.3, 0.05],
        "importances_std": [0.01, 0.02, 0.03, 0.005],
        "analysis_type": "feature_importance",
        "n_samples_used": 30,
        "n_repeats": 10,
    }
    cache.set_cached(job_id, "feature_importance", cached_result, db_session)

    # Both calls should return 200 from cache
    response1 = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
    response2 = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json()["features"] == response2.json()["features"]


# ------------------------------------------------------------------
# Test 14: test_feature_importance_names_match_dataset
# ------------------------------------------------------------------


@pytest.mark.slow
def test_feature_importance_names_match_dataset(trained_job, db_session):
    """Feature names in result match dataset columns."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    feature_names = trained_job["feature_names"]

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data_with_features",
            return_value=(X_test, y_test, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        result = service._compute_feature_importance_sync(job_id, db_session)

    # All returned feature names should be from the original dataset
    for name in result["features"]:
        assert name in feature_names, f"Feature {name} not in original dataset columns"

    # Since we have 4 features (< 20 cap), all should be present
    assert len(result["features"]) == len(feature_names)


# ------------------------------------------------------------------
# Week 10 Tests: Predictions Endpoint
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Test 15: test_predictions_endpoint_pagination
# ------------------------------------------------------------------


@pytest.mark.slow
def test_predictions_endpoint_pagination(trained_job, db_session):
    """GET /predictions?offset=0&limit=5 → 5 predictions."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [str(i) for i in range(3)]
    feature_names = trained_job["feature_names"]

    with (
        patch.object(
            InterpretabilityService,
            "_load_test_data_complete",
            return_value=(X_test, y_test, class_names, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        response = client.get(
            f"/api/v1/model/analysis/{job_id}/predictions",
            params={"offset": 0, "limit": 5},
        )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) <= 5  # May be fewer if test set < 5
    assert data["offset"] == 0
    assert data["limit"] == 5


# ------------------------------------------------------------------
# Test 16: test_predictions_endpoint_filter_correct
# ------------------------------------------------------------------


@pytest.mark.slow
def test_predictions_endpoint_filter_correct(trained_job, db_session):
    """GET /predictions?filter=correct → only is_correct=true rows."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [str(i) for i in range(3)]
    feature_names = trained_job["feature_names"]

    with (
        patch.object(
            InterpretabilityService,
            "_load_test_data_complete",
            return_value=(X_test, y_test, class_names, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        response = client.get(
            f"/api/v1/model/analysis/{job_id}/predictions",
            params={"filter": "correct", "limit": 100},
        )

    assert response.status_code == 200
    data = response.json()
    # All returned predictions should be correct
    for pred in data["predictions"]:
        assert pred["is_correct"] is True


# ------------------------------------------------------------------
# Test 17: test_predictions_confidence_sorted_desc
# ------------------------------------------------------------------


@pytest.mark.slow
def test_predictions_confidence_sorted_desc(trained_job, db_session):
    """Predictions are sorted by confidence DESC by default."""
    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    class_names = [str(i) for i in range(3)]
    feature_names = trained_job["feature_names"]

    with (
        patch.object(
            InterpretabilityService,
            "_load_test_data_complete",
            return_value=(X_test, y_test, class_names, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        response = client.get(
            f"/api/v1/model/analysis/{job_id}/predictions",
            params={"limit": 100},
        )

    assert response.status_code == 200
    data = response.json()
    predictions = data["predictions"]

    if len(predictions) > 1:
        # Verify descending order
        confidences = [p["confidence"] for p in predictions]
        assert confidences == sorted(confidences, reverse=True), "Predictions not sorted by confidence DESC"


# ------------------------------------------------------------------
# Test 18: test_predictions_endpoint_invalid_filter
# ------------------------------------------------------------------


def test_predictions_endpoint_invalid_filter(db_session, export_dir):
    """GET /predictions?filter=invalid → 400 Bad Request."""
    # Create a new trained job with unique model name
    job_id, X_test, y_test, feature_names, model = create_test_training_job(
        session=db_session,
        export_dir=export_dir,
        model_name=f"test_invalid_filter_{uuid4().hex[:8]}",
        num_features=4,
        num_classes=3,
        epochs=5,
    )

    response = client.get(
        f"/api/v1/model/analysis/{job_id}/predictions",
        params={"filter": "invalid"},
    )

    assert response.status_code == 400
    data = response.json()
    assert "Invalid filter parameter" in data["detail"]


# ------------------------------------------------------------------
# Test 19: test_predictions_endpoint_regression_job
# ------------------------------------------------------------------


def test_predictions_endpoint_regression_job(regression_job, db_session):
    """GET /predictions on a regression job → 400 Bad Request."""
    job_id = regression_job

    response = client.get(
        f"/api/v1/model/analysis/{job_id}/predictions",
    )

    assert response.status_code == 400
    data = response.json()
    assert "not available for regression" in data["detail"].lower()


# ------------------------------------------------------------------
# Test 20: test_residuals_endpoint_classification_job
# ------------------------------------------------------------------


@pytest.mark.slow
def test_residuals_endpoint_classification_job(db_session, export_dir):
    """GET /residuals on a classification job → 400 Bad Request."""
    # Create a new trained job with unique model name
    job_id, X_test, y_test, feature_names, model = create_test_training_job(
        session=db_session,
        export_dir=export_dir,
        model_name=f"test_residuals_clf_{uuid4().hex[:8]}",
        num_features=4,
        num_classes=3,
        epochs=5,
    )

    response = client.get(
        f"/api/v1/model/analysis/{job_id}/residuals",
    )

    assert response.status_code == 400
    data = response.json()
    assert "only available for regression" in data["detail"].lower()


# ------------------------------------------------------------------
# Test 21: Unit tests for _resolve_classification_predictions
# ------------------------------------------------------------------


def test_resolve_classification_predictions_binary_and_multiclass():
    """Verify prediction and probability resolution for binary and multiclass models."""
    from app.services.interpretability import (
        _resolve_classification_predictions,
        _resolve_ground_truth_classes,
    )

    # Binary single-column 2D: shape (4, 1)
    raw_2d = np.array([[0.9], [0.1], [0.5], [0.49]], dtype=np.float32)
    classes, probs = _resolve_classification_predictions(raw_2d)
    assert np.array_equal(classes, np.array([1, 0, 1, 0]))
    assert probs.shape == (4, 2)
    assert np.isclose(probs[0, 0], 0.1)
    assert np.isclose(probs[0, 1], 0.9)
    assert np.isclose(probs[1, 0], 0.9)
    assert np.isclose(probs[1, 1], 0.1)

    # Binary 1D: shape (4,)
    raw_1d = np.array([0.8, 0.2, 0.7, 0.3], dtype=np.float32)
    classes_1d, probs_1d = _resolve_classification_predictions(raw_1d)
    assert np.array_equal(classes_1d, np.array([1, 0, 1, 0]))
    assert probs_1d.shape == (4, 2)

    # Multi-class: shape (3, 3)
    raw_multi = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32)
    classes_m, probs_m = _resolve_classification_predictions(raw_multi)
    assert np.array_equal(classes_m, np.array([1, 0, 2]))
    assert np.array_equal(probs_m, raw_multi)

    # Ground truth resolution
    y_1d = np.array([0, 1, 0])
    assert np.array_equal(_resolve_ground_truth_classes(y_1d), np.array([0, 1, 0]))
    y_col = np.array([[0], [1], [0]])
    assert np.array_equal(_resolve_ground_truth_classes(y_col), np.array([0, 1, 0]))
    y_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.array_equal(_resolve_ground_truth_classes(y_onehot), np.array([0, 1, 2]))


# ------------------------------------------------------------------
# Test 22: Binary classification model confusion matrix & predictions
# ------------------------------------------------------------------


@pytest.mark.slow
def test_binary_classification_confusion_matrix_and_predictions(db_session, export_dir):
    """Binary classifier with 1 sigmoid output unit produces valid predictions and 2x2 confusion matrix."""
    import tensorflow as tf

    # Create a small binary model (1 sigmoid output)
    num_features = 4
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(num_features,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Synthetic binary data
    np.random.seed(42)
    X_train = np.random.randn(80, num_features).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.float32)
    X_test = np.random.randn(20, num_features).astype(np.float32)
    y_test = (X_test[:, 0] > 0).astype(np.float32)
    feature_names = [f"f_{i}" for i in range(num_features)]
    class_names = ["class_0", "class_1"]

    model.fit(X_train, y_train, epochs=3, verbose=0)

    # Save model
    job_id = str(uuid4())
    job_dir = export_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(job_dir / "model.keras"))

    # DB records
    db_model = ModelBasic(
        model_name=f"test_binary_{uuid4().hex[:8]}",
        model_type=ProblemType.CLASSIFICATION,
        graph_ir={},
    )
    db_session.add(db_model)
    db_session.commit()
    db_session.refresh(db_model)

    job = TrainingJob(
        id=job_id,
        model_id=db_model.id,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data",
            return_value=(X_test, y_test, class_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            export_dir,
        ),
    ):
        cm_result = service._compute_confusion_matrix_sync(job_id, db_session)

    # Confusion matrix must be 2x2
    cm = cm_result["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2
    assert cm_result["n_samples"] == len(y_test)

    # Predictions sync
    with (
        patch.object(
            service,
            "_load_test_data_complete",
            return_value=(X_test, y_test, class_names, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            export_dir,
        ),
    ):
        pred_result = service._compute_all_predictions_sync(job_id, db_session)

    predictions = pred_result["predictions"]
    assert len(predictions) == len(X_test)
    for p in predictions:
        assert len(p["probabilities"]) == 2, "Binary model probabilities must have length 2 [p0, p1]"
        assert np.isclose(sum(p["probabilities"]), 1.0, atol=1e-5), "Probabilities must sum to 1"
        assert p["predicted_class"] in (0, 1)
        assert p["confidence"] >= 0.5, "Confidence in binary prediction should be >= 0.5"


# ------------------------------------------------------------------
# Test 23: Image classification guards on feature importance & predictions
# ------------------------------------------------------------------


def test_image_classification_analysis_endpoints_guard(db_session):
    """GET /feature-importance and /predictions return 400 for image classification jobs."""
    db_model = ModelBasic(
        model_name=f"test_img_clf_{uuid4().hex[:8]}",
        model_type=ProblemType.IMAGE_CLASSIFICATION,
        graph_ir={},
    )
    db_session.add(db_model)
    db_session.commit()
    db_session.refresh(db_model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=db_model.id,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    # Feature importance should reject image classification with 400
    fi_resp = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
    assert fi_resp.status_code == 400
    assert "image classification" in fi_resp.json()["detail"].lower()

    # Predictions explorer should reject image classification with 400
    pred_resp = client.get(f"/api/v1/model/analysis/{job_id}/predictions")
    assert pred_resp.status_code == 400
    assert "image classification" in pred_resp.json()["detail"].lower()


# ------------------------------------------------------------------
# Test 24: test_feature_importance_with_real_keras_model_more_than_20_features
# ------------------------------------------------------------------


def test_feature_importance_with_real_keras_model_more_than_20_features(db_session, export_dir):
    """Real Keras model with 25 features evaluates without shape mismatch and returns 20 features."""
    job_id, X_test, y_test, feature_names, model = create_test_training_job(
        session=db_session,
        export_dir=export_dir,
        model_name=f"test_real_25feat_{uuid4().hex[:8]}",
        num_features=25,
        num_classes=3,
        epochs=2,
    )

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data_with_features",
            return_value=(X_test, y_test, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            export_dir,
        ),
    ):
        result = service._compute_feature_importance_sync(job_id, db_session)

    assert len(result["features"]) == 20
    assert len(result["importances_mean"]) == 20
    assert len(result["importances_std"]) == 20
    assert result["analysis_type"] == "feature_importance"


# ------------------------------------------------------------------
# Test 25: test_feature_importance_regression_model
# ------------------------------------------------------------------


@pytest.mark.slow
def test_feature_importance_regression_model(db_session, export_dir):
    """Regression model computes feature importance using r2 scoring without classification errors."""
    import tensorflow as tf

    num_features = 6
    X_train = np.random.randn(100, num_features).astype(np.float32)
    y_train = (X_train[:, 0] * 2.5 + X_train[:, 1] * 1.2 + np.random.randn(100) * 0.1).astype(np.float32)
    X_test = np.random.randn(30, num_features).astype(np.float32)
    y_test = (X_test[:, 0] * 2.5 + X_test[:, 1] * 1.2 + np.random.randn(30) * 0.1).astype(np.float32)
    feature_names = [f"reg_feat_{i}" for i in range(num_features)]

    reg_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(num_features,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    reg_model.compile(optimizer="adam", loss="mse")
    reg_model.fit(X_train, y_train, epochs=2, verbose=0)

    model_record = ModelBasic(
        model_name=f"test_reg_fi_{uuid4().hex[:8]}",
        model_type=ProblemType.REGRESSION,
        graph_ir={},
    )
    db_session.add(model_record)
    db_session.commit()
    db_session.refresh(model_record)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model_record.id,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    job_export_dir = export_dir / job_id
    job_export_dir.mkdir(parents=True, exist_ok=True)
    reg_model.save(str(job_export_dir / "model.keras"))

    service = InterpretabilityService()

    with (
        patch.object(
            service,
            "_load_test_data_with_features",
            return_value=(X_test, y_test, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            export_dir,
        ),
    ):
        result = service._compute_feature_importance_sync(job_id, db_session)

    assert len(result["features"]) == num_features
    assert len(result["importances_mean"]) == num_features
    assert result["analysis_type"] == "feature_importance"


# ------------------------------------------------------------------
# Test 26: test_feature_importance_failed_status_returns_500
# ------------------------------------------------------------------


def test_feature_importance_failed_status_returns_500(trained_job, db_session):
    """When background task failed, endpoint returns generic 500 and clears failed sentinel."""
    job_id = trained_job["job_id"]
    cache = AnalysisCache()
    cache.set_cached(
        job_id,
        "feature_importance_status",
        {"status": "failed", "error": "Disk read failure"},
        db_session,
    )

    response = client.get(f"/api/v1/model/analysis/{job_id}/feature-importance")
    assert response.status_code == 500
    assert response.json()["detail"] == "Feature importance computation failed"

    # Verify sentinel was cleared to allow subsequent requests to retry
    sentinel = cache.get_cached(job_id, "feature_importance_status", db_session)
    assert sentinel is None


# ------------------------------------------------------------------
# Test 27: test_feature_importance_isolated_session_background_computation
# ------------------------------------------------------------------


def test_feature_importance_isolated_session_background_computation(trained_job, db_session):
    """Background computation runs with an isolated session factory without request session leak."""
    from contextlib import contextmanager

    job_id = trained_job["job_id"]
    X_test = trained_job["X_test"]
    y_test = trained_job["y_test"]
    feature_names = trained_job["feature_names"]

    service = InterpretabilityService()

    @contextmanager
    def isolated_session_factory():
        yield db_session

    with (
        patch.object(
            service,
            "_load_test_data_with_features",
            return_value=(X_test, y_test, feature_names),
        ),
        patch(
            "app.services.interpretability.EXPORTS_BASE",
            trained_job["export_dir"],
        ),
    ):
        asyncio.run(service._compute_and_cache_feature_importance(job_id, session_factory=isolated_session_factory))

    # Verify cached result exists
    cached = service.cache.get_cached(job_id, "feature_importance", db_session)
    assert cached is not None
    assert cached["analysis_type"] == "feature_importance"
    assert len(cached["features"]) == len(feature_names)

    # Verify status is completed
    status = service.cache.get_cached(job_id, "feature_importance_status", db_session)
    assert status == {"status": "completed"}
