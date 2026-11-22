import csv
import io
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.main import app
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.shared.enums import ProblemType

client = TestClient(app)


@pytest.fixture
def classification_job(db_session: Session):
    model = ModelBasic(
        model_name="test_clf_export_model",
        model_type=ProblemType.CLASSIFICATION,
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
        analysis_cache={
            "confusion_matrix": {
                "confusion_matrix": [[10, 2], [1, 15]],
                "class_names": ["cat", "dog"],
                "overall_accuracy": 0.892,
            },
            "roc_curve": {
                "macro_auc": 0.94,
                "curves": [{"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0], "auc": 0.94}],
            },
        },
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    return job_id


@pytest.fixture
def regression_job(db_session: Session):
    model = ModelBasic(
        model_name="test_reg_export_model",
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
        analysis_cache={
            "regression_analysis": {
                "mae": 1.25,
                "mse": 2.50,
                "rmse": 1.58,
                "r2": 0.88,
            }
        },
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    return job_id


def test_export_rejects_nonexistent_job(db_session: Session):
    resp = client.get(f"/api/v1/model/analysis/{uuid4()}/export")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Training job not found"


def test_export_rejects_pending_job(db_session: Session):
    model = ModelBasic(model_name="pending_model", model_type=ProblemType.CLASSIFICATION)
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.RUNNING,
    )
    db_session.add(job)
    db_session.commit()

    resp = client.get(f"/api/v1/model/analysis/{job_id}/export")
    assert resp.status_code == 400
    assert "Analysis only available for completed jobs" in resp.json()["detail"]


def test_export_rejects_job_with_no_cached_analysis(db_session: Session):
    model = ModelBasic(model_name="empty_cache_model", model_type=ProblemType.CLASSIFICATION)
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        analysis_cache={},
    )
    db_session.add(job)
    db_session.commit()

    resp = client.get(f"/api/v1/model/analysis/{job_id}/export")
    assert resp.status_code == 409
    assert "Analysis results are not available for export yet" in resp.json()["detail"]


def test_export_classification_json_success(classification_job):
    resp = client.get(f"/api/v1/model/analysis/{classification_job}/export?format=json")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    assert f'filename="analysis-{classification_job}.json"' in resp.headers["content-disposition"]

    payload = resp.json()
    assert payload["job_id"] == classification_job
    assert payload["analysis_type"] == "classification"
    assert "confusion_matrix" in payload["results"]
    assert "roc_curve" in payload["results"]
    assert "feature_importance" in payload["unavailable_results"]


def test_export_classification_csv_success(classification_job):
    resp = client.get(f"/api/v1/model/analysis/{classification_job}/export?format=csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert f'filename="analysis-{classification_job}.csv"' in resp.headers["content-disposition"]

    reader = list(csv.reader(io.StringIO(resp.text)))
    header = reader[0]
    assert header == ["analysis", "field", "type", "value"]

    # Verify rows exist for confusion_matrix and roc_curve
    analyses = {row[0] for row in reader[1:]}
    assert "confusion_matrix" in analyses
    assert "roc_curve" in analyses


def test_export_regression_json_success(regression_job):
    resp = client.get(f"/api/v1/model/analysis/{regression_job}/export?format=json")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["job_id"] == regression_job
    assert payload["analysis_type"] == "regression"
    assert "regression_analysis" in payload["results"]
    assert payload["results"]["regression_analysis"]["mae"] == 1.25


def test_json_safe_converts_nan_and_inf():
    from app.routers.analysis import _json_safe

    input_data = {
        "metrics": {"mae": float("nan"), "mse": float("inf"), "rmse": float("-inf")},
        "score": 0.95,
        "nested": [float("nan"), 1.0],
    }
    result = _json_safe(input_data)
    assert result["metrics"]["mae"] is None
    assert result["metrics"]["mse"] is None
    assert result["metrics"]["rmse"] is None
    assert result["score"] == 0.95
    assert result["nested"] == [None, 1.0]
