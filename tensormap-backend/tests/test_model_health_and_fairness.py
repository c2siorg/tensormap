"""Comprehensive unit and integration tests for Model Health Diagnostics and Algorithmic Fairness Audit."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.interpretability import InterpretabilityService
from app.shared.enums import ProblemType
from tests.fixtures.create_test_model import create_test_training_job

client = TestClient(app)


@pytest.fixture
def export_dir(tmp_path: Path):
    """Temporary export directory."""
    return tmp_path / "exports"


@pytest.fixture
def trained_job(db_session: Session, export_dir: Path):
    """Create a completed training job with a trained 2-class model."""
    job_id, X_test, y_test, feature_names, model = create_test_training_job(
        session=db_session,
        export_dir=export_dir,
        model_name="test_health_model",
        num_features=4,
        num_classes=2,
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
    """Create a completed regression job."""
    model = ModelBasic(
        model_name="test_reg_model",
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
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()
    return job_id


# -----------------------------------------------------------------------------
# Layer-wise Weight & Activation Diagnostics Tests
# -----------------------------------------------------------------------------


def test_layer_diagnostics_structure_and_metrics(trained_job, db_session):
    """Layer diagnostics extracts weights, norms, sparsity, and histogram bins."""
    job_id = trained_job["job_id"]
    service = InterpretabilityService()

    with patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]):
        result = service._compute_layer_diagnostics_sync(job_id, db_session)

    assert result["job_id"] == job_id
    assert result["total_layers"] >= 1
    assert result["total_parameters"] > 0
    assert result["overall_status"] in ("HEALTHY", "WARNING", "CRITICAL")

    first_layer = result["layers"][0]
    assert "layer_name" in first_layer
    assert "param_count" in first_layer
    assert "l2_norm" in first_layer
    assert "dead_weights_ratio" in first_layer
    assert len(first_layer["histogram"]) == 10
    assert first_layer["status"] == "HEALTHY"


def test_layer_diagnostics_vanishing_weights_detection(trained_job, db_session):
    """Detects vanishing weights when layer weight norm is near zero."""
    import tensorflow as tf

    job_id = trained_job["job_id"]
    model_path = trained_job["export_dir"] / job_id / "model.keras"
    model = tf.keras.models.load_model(model_path)

    # Force first trainable layer weights to near zero
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            near_zero = [np.full_like(w, 1e-6) for w in weights]
            layer.set_weights(near_zero)
            break
    model.save(model_path)

    service = InterpretabilityService()
    with patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]):
        result = service._compute_layer_diagnostics_sync(job_id, db_session)

    assert result["overall_status"] in ("WARNING", "CRITICAL")
    vanishing_layers = [layer for layer in result["layers"] if layer["status"] == "VANISHING_WEIGHTS"]
    assert len(vanishing_layers) >= 1


def test_layer_diagnostics_exploding_weights_detection(trained_job, db_session):
    """Detects exploding weights when weight magnitudes exceed threshold."""
    import tensorflow as tf

    job_id = trained_job["job_id"]
    model_path = trained_job["export_dir"] / job_id / "model.keras"
    model = tf.keras.models.load_model(model_path)

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            exploding = [np.full_like(w, 500.0) for w in weights]
            layer.set_weights(exploding)
            break
    model.save(model_path)

    service = InterpretabilityService()
    with patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]):
        result = service._compute_layer_diagnostics_sync(job_id, db_session)

    assert result["overall_status"] in ("WARNING", "CRITICAL")
    exploding_layers = [layer for layer in result["layers"] if layer["status"] == "EXPLODING_WEIGHTS"]
    assert len(exploding_layers) >= 1


def test_layer_diagnostics_nan_detection(trained_job, db_session):
    """Detects NaN/Inf weights and marks layer as critical NAN_OR_INF."""
    import tensorflow as tf

    job_id = trained_job["job_id"]
    model_path = trained_job["export_dir"] / job_id / "model.keras"
    model = tf.keras.models.load_model(model_path)

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            nan_weights = [np.full_like(w, np.nan) for w in weights]
            layer.set_weights(nan_weights)
            break
    model.save(model_path)

    service = InterpretabilityService()
    with patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]):
        result = service._compute_layer_diagnostics_sync(job_id, db_session)

    assert result["overall_status"] == "CRITICAL"
    nan_layers = [layer for layer in result["layers"] if layer["status"] == "NAN_OR_INF"]
    assert len(nan_layers) >= 1


def test_layer_diagnostics_high_sparsity_detection(trained_job, db_session):
    """Detects dead weights / extreme sparsity (>80% dead weights)."""
    import tensorflow as tf

    job_id = trained_job["job_id"]
    model_path = trained_job["export_dir"] / job_id / "model.keras"
    model = tf.keras.models.load_model(model_path)

    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            sparse_w = np.zeros_like(weights[0])
            # Keep only 5% of weights non-zero
            sparse_w.flat[0] = 1.0
            layer.set_weights([sparse_w] + weights[1:])
            break
    model.save(model_path)

    service = InterpretabilityService()
    with patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]):
        result = service._compute_layer_diagnostics_sync(job_id, db_session)

    sparse_layers = [layer for layer in result["layers"] if layer["status"] == "HIGH_SPARSITY"]
    assert len(sparse_layers) >= 1
    assert sparse_layers[0]["dead_weights_ratio"] > 0.8


def test_layer_diagnostics_caching(trained_job, db_session):
    """Repeated calls to compute_layer_diagnostics_async serve from cache."""
    job_id = trained_job["job_id"]
    service = InterpretabilityService()

    with patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]):
        res1 = asyncio.run(service.compute_layer_diagnostics_async(job_id, db_session))
        with patch.object(service, "_compute_layer_diagnostics_sync") as mock_sync:
            res2 = asyncio.run(service.compute_layer_diagnostics_async(job_id, db_session))
            mock_sync.assert_not_called()

    assert res1["total_layers"] == res2["total_layers"]


# -----------------------------------------------------------------------------
# Algorithmic Fairness & Demographic Parity Tests
# -----------------------------------------------------------------------------


def test_fairness_audit_balanced_subgroups(trained_job, db_session):
    """Fairness audit computes disparate impact ratio and 80% rule compliance."""
    job_id = trained_job["job_id"]
    service = InterpretabilityService()

    n_samples = len(trained_job["y_test"])
    raw_df = pd.DataFrame(
        {
            "feature_0": trained_job["X_test"][:, 0],
            "feature_1": trained_job["X_test"][:, 1],
            "gender": ["Group_A" if i % 2 == 0 else "Group_B" for i in range(n_samples)],
            "target": trained_job["y_test"],
        }
    )

    with (
        patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
        patch.object(
            service,
            "_load_test_data_with_raw_features",
            return_value=(trained_job["X_test"], trained_job["y_test"], ["0", "1"], raw_df),
        ),
    ):
        result = service._compute_fairness_audit_sync(job_id, "gender", None, db_session)

    assert result["sensitive_feature"] == "gender"
    assert "Group_A" in result["group_metrics"]
    assert "Group_B" in result["group_metrics"]
    assert result["privileged_group"] in ("Group_A", "Group_B")

    gm_a = result["group_metrics"]["Group_A"]
    assert "selection_rate" in gm_a
    assert "accuracy" in gm_a
    assert "true_positive_rate" in gm_a
    assert "false_positive_rate" in gm_a

    assert "disparities" in result
    assert result["overall_summary"]["groups_evaluated"] == 2


def test_fairness_audit_biased_subgroups_fails_80_percent_rule(trained_job, db_session):
    """Detects disparate impact violation when selection rate ratio is below 0.8."""
    job_id = trained_job["job_id"]
    service = InterpretabilityService()

    n_samples = len(trained_job["y_test"])
    raw_df = pd.DataFrame(
        {
            "gender": ["Privileged" if i < (n_samples // 2) else "Unprivileged" for i in range(n_samples)],
            "target": trained_job["y_test"],
        }
    )

    # Mock predictions where Privileged has 100% positive predictions and Unprivileged has 10%
    y_pred_biased = np.array([1] * (n_samples // 2) + [0] * (n_samples - n_samples // 2))

    with (
        patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
        patch.object(
            service,
            "_load_test_data_with_raw_features",
            return_value=(trained_job["X_test"], trained_job["y_test"], ["0", "1"], raw_df),
        ),
        patch("app.services.interpretability._resolve_classification_predictions", return_value=(y_pred_biased, None)),
    ):
        result = service._compute_fairness_audit_sync(job_id, "gender", "Privileged", db_session)

    assert result["privileged_group"] == "Privileged"
    disp = result["disparities"]["Unprivileged"]
    assert disp["disparate_impact_ratio"] < 0.8
    assert disp["passes_80_percent_rule"] is False
    assert result["overall_summary"]["compliant_with_80_percent_rule"] is False


def test_fairness_audit_single_group_rejected(trained_job, db_session):
    """Rejects sensitive features with only 1 unique subgroup."""
    job_id = trained_job["job_id"]
    service = InterpretabilityService()

    n_samples = len(trained_job["y_test"])
    raw_df = pd.DataFrame(
        {
            "all_same": ["Uniform"] * n_samples,
            "target": trained_job["y_test"],
        }
    )

    with (
        patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
        patch.object(
            service,
            "_load_test_data_with_raw_features",
            return_value=(trained_job["X_test"], trained_job["y_test"], ["0", "1"], raw_df),
        ),
        pytest.raises(ValueError, match="At least 2 groups are required"),
    ):
        service._compute_fairness_audit_sync(job_id, "all_same", None, db_session)


def test_fairness_audit_missing_feature_rejected(trained_job, db_session):
    """Rejects sensitive features not present in dataset."""
    job_id = trained_job["job_id"]
    service = InterpretabilityService()

    raw_df = pd.DataFrame({"target": trained_job["y_test"]})

    with (
        patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
        patch.object(
            service,
            "_load_test_data_with_raw_features",
            return_value=(trained_job["X_test"], trained_job["y_test"], ["0", "1"], raw_df),
        ),
        pytest.raises(ValueError, match="Sensitive feature 'missing_col' not found"),
    ):
        service._compute_fairness_audit_sync(job_id, "missing_col", None, db_session)


# -----------------------------------------------------------------------------
# Defensive Feature Handling Tests (Option 6 Safeguards)
# -----------------------------------------------------------------------------


def test_defensive_load_test_data_non_numeric_features(db_session, export_dir, tmp_path):
    """Encodes non-numeric string features into integer categories without float conversion crash."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    csv_path = upload_dir / "mixed_data.csv"

    # CSV with string features and target
    df = pd.DataFrame(
        {
            "cat_feature": ["A", "B", "A", "C", "B"] * 20,
            "num_feature": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
            "label": [0, 1, 0, 1, 0] * 20,
        }
    )
    df.to_csv(csv_path, index=False)

    data_file = DataFile(
        file_name="mixed_data.csv",
        disk_name="mixed_data.csv",
        file_type="csv",
        row_count=100,
        columns=["cat_feature", "num_feature", "label"],
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    model = ModelBasic(
        model_name="test_mixed_model",
        file_id=data_file.id,
        target_field="label",
        training_split=80.0,
        model_type=ProblemType.CLASSIFICATION,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job = TrainingJob(
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    service = InterpretabilityService()
    settings_mock = type("Settings", (), {"upload_folder": str(upload_dir)})()

    with patch("app.config.get_settings", return_value=settings_mock):
        X_test, y_test, class_names = service._load_test_data(job.id, db_session)
        assert X_test.dtype == np.float32
        assert X_test.shape[1] == 2
        assert len(y_test) == 20

        X_c, y_c, classes, raw_df = service._load_test_data_with_raw_features(job.id, db_session)
        assert "cat_feature" in raw_df.columns
        assert len(raw_df) == 20


def test_defensive_load_test_data_empty_split_guard(db_session, tmp_path):
    """Guards against splits that leave zero evaluation samples."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    csv_path = upload_dir / "tiny.csv"

    df = pd.DataFrame({"x": [1.0, 2.0], "label": [0, 1]})
    df.to_csv(csv_path, index=False)

    data_file = DataFile(
        file_name="tiny.csv",
        disk_name="tiny.csv",
        file_type="csv",
        row_count=2,
        columns=["x", "label"],
    )
    db_session.add(data_file)
    db_session.commit()

    model = ModelBasic(
        model_name="test_tiny_model",
        file_id=data_file.id,
        target_field="label",
        training_split=100.0,
        model_type=ProblemType.CLASSIFICATION,
    )
    db_session.add(model)
    db_session.commit()

    job = TrainingJob(
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()

    service = InterpretabilityService()
    settings_mock = type("Settings", (), {"upload_folder": str(upload_dir)})()

    with (
        patch("app.config.get_settings", return_value=settings_mock),
        pytest.raises(ValueError, match="leaves no test samples"),
    ):
        service._load_test_data(job.id, db_session)


# -----------------------------------------------------------------------------
# REST API Endpoint Tests
# -----------------------------------------------------------------------------


def test_api_layer_diagnostics_endpoint(trained_job, db_session):
    """GET /model/analysis/{job_id}/layer-diagnostics returns 200 with diagnostics payload."""
    job_id = trained_job["job_id"]

    with patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]):
        response = client.get(f"/api/v1/model/analysis/{job_id}/layer-diagnostics")

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert "layers" in data
    assert "overall_status" in data


def test_api_fairness_audit_endpoint(trained_job, db_session):
    """GET /model/analysis/{job_id}/fairness returns 200 for classification."""
    job_id = trained_job["job_id"]
    n_samples = len(trained_job["y_test"])
    raw_df = pd.DataFrame(
        {
            "gender": ["M" if i % 2 == 0 else "F" for i in range(n_samples)],
            "target": trained_job["y_test"],
        }
    )

    with (
        patch("app.services.interpretability.EXPORTS_BASE", trained_job["export_dir"]),
        patch(
            "app.services.interpretability.InterpretabilityService._load_test_data_with_raw_features",
            return_value=(trained_job["X_test"], trained_job["y_test"], ["0", "1"], raw_df),
        ),
    ):
        response = client.get(f"/api/v1/model/analysis/{job_id}/fairness?sensitive_feature=gender")

    assert response.status_code == 200
    data = response.json()
    assert data["sensitive_feature"] == "gender"
    assert "disparities" in data


def test_api_fairness_audit_rejects_regression(regression_job, db_session):
    """Fairness audit endpoint returns 400 for regression models."""
    response = client.get(f"/api/v1/model/analysis/{regression_job}/fairness?sensitive_feature=gender")
    assert response.status_code == 400
    assert "tabular classification" in response.json()["detail"]


def test_api_nonexistent_job_returns_404(db_session):
    """Nonexistent job IDs return 404."""
    fake_id = str(uuid4())
    res_diag = client.get(f"/api/v1/model/analysis/{fake_id}/layer-diagnostics")
    assert res_diag.status_code == 404

    res_fair = client.get(f"/api/v1/model/analysis/{fake_id}/fairness?sensitive_feature=gender")
    assert res_fair.status_code == 404
