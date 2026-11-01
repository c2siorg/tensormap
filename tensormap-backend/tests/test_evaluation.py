"""Tests for holdout dataset evaluation, error diagnostics, and batch prediction API.

Tests cover:
- POST /model/analysis/{job_id}/evaluate for classification models
- POST /model/analysis/{job_id}/evaluate for regression models
- Top error sample extraction and ranking
- Schema validation (missing features, missing target column, empty dataset)
- Image classification guard returning 400
- Lifecycle guards (incomplete jobs 400, non-existent jobs 404)
- POST /model/analysis/{job_id}/batch-predict preview endpoint
- POST /model/analysis/{job_id}/batch-predict/export CSV download endpoint
"""

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.shared.enums import ProblemType


@pytest.fixture
def eval_classification_setup(db_session: Session, tmp_path: Path):
    """Sets up a synthetic classification model, training dataset, holdout test dataset, and model.keras."""
    np.random.seed(42)
    n_train = 60
    n_test = 30

    train_df = pd.DataFrame(
        {
            "feat_a": np.random.uniform(1.0, 5.0, n_train),
            "feat_b": np.random.uniform(10.0, 20.0, n_train),
            "species": np.random.choice(["setosa", "versicolor", "virginica"], n_train),
        }
    )

    test_df = pd.DataFrame(
        {
            "feat_a": np.random.uniform(1.0, 5.0, n_test),
            "feat_b": np.random.uniform(10.0, 20.0, n_test),
            "species": np.random.choice(["setosa", "versicolor", "virginica"], n_test),
        }
    )

    upload_folder = tmp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)

    train_csv = f"train_{uuid4().hex[:6]}.csv"
    test_csv = f"test_{uuid4().hex[:6]}.csv"
    train_df.to_csv(upload_folder / train_csv, index=False)
    test_df.to_csv(upload_folder / test_csv, index=False)

    train_file = DataFile(
        file_name="train.csv",
        disk_name=train_csv,
        file_type="csv",
        total_rows=n_train,
        total_columns=3,
    )
    test_file = DataFile(
        file_name="test.csv",
        disk_name=test_csv,
        file_type="csv",
        total_rows=n_test,
        total_columns=3,
    )
    db_session.add(train_file)
    db_session.add(test_file)
    db_session.commit()
    db_session.refresh(train_file)
    db_session.refresh(test_file)

    model = ModelBasic(
        model_name=f"clf_model_{uuid4().hex[:6]}",
        file_id=train_file.id,
        target_field="species",
        model_type=ProblemType.CLASSIFICATION,
        training_split=80.0,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
    )
    db_session.add(job)
    db_session.commit()

    import tensorflow as tf

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True, exist_ok=True)

    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(4, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    keras_model.save(export_dir / "model.keras")

    return {
        "job_id": job_id,
        "train_file": train_file,
        "test_file": test_file,
        "model": model,
        "export_dir": export_dir,
        "upload_folder": upload_folder,
        "train_df": train_df,
        "test_df": test_df,
    }


@pytest.fixture
def eval_regression_setup(db_session: Session, tmp_path: Path):
    """Sets up a synthetic regression model, training dataset, holdout test dataset, and model.keras."""
    np.random.seed(42)
    n_train = 50
    n_test = 25

    train_df = pd.DataFrame(
        {
            "size": np.random.uniform(50.0, 200.0, n_train),
            "rooms": np.random.uniform(1.0, 5.0, n_train),
            "price": np.random.uniform(100.0, 500.0, n_train),
        }
    )

    test_df = pd.DataFrame(
        {
            "size": np.random.uniform(50.0, 200.0, n_test),
            "rooms": np.random.uniform(1.0, 5.0, n_test),
            "price": np.random.uniform(100.0, 500.0, n_test),
        }
    )

    upload_folder = tmp_path / "uploads_reg"
    upload_folder.mkdir(parents=True, exist_ok=True)

    train_csv = f"train_reg_{uuid4().hex[:6]}.csv"
    test_csv = f"test_reg_{uuid4().hex[:6]}.csv"
    train_df.to_csv(upload_folder / train_csv, index=False)
    test_df.to_csv(upload_folder / test_csv, index=False)

    train_file = DataFile(
        file_name="train_reg.csv",
        disk_name=train_csv,
        file_type="csv",
        total_rows=n_train,
        total_columns=3,
    )
    test_file = DataFile(
        file_name="test_reg.csv",
        disk_name=test_csv,
        file_type="csv",
        total_rows=n_test,
        total_columns=3,
    )
    db_session.add(train_file)
    db_session.add(test_file)
    db_session.commit()
    db_session.refresh(train_file)
    db_session.refresh(test_file)

    model = ModelBasic(
        model_name=f"reg_model_{uuid4().hex[:6]}",
        file_id=train_file.id,
        target_field="price",
        model_type=ProblemType.REGRESSION,
        training_split=80.0,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
    )
    db_session.add(job)
    db_session.commit()

    import tensorflow as tf

    export_dir = tmp_path / "exports_reg" / job_id
    export_dir.mkdir(parents=True, exist_ok=True)

    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(4, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    keras_model.save(export_dir / "model.keras")

    return {
        "job_id": job_id,
        "train_file": train_file,
        "test_file": test_file,
        "model": model,
        "export_dir": export_dir,
        "upload_folder": upload_folder,
        "train_df": train_df,
        "test_df": test_df,
    }


def test_evaluate_classification_holdout_dataset(eval_classification_setup, client: TestClient):
    """Verifies that holdout evaluation on a classification dataset computes full scorecards."""
    job_id = eval_classification_setup["job_id"]
    test_file = eval_classification_setup["test_file"]
    upload_folder = str(eval_classification_setup["upload_folder"])
    exports_base = eval_classification_setup["export_dir"].parent

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = upload_folder

        resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(test_file.id)},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["problem_type"] == "classification"
        assert data["total_samples"] == 30
        assert "evaluated_at" in data

        metrics = data["metrics"]
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "confusion_matrix" in metrics
        assert len(metrics["class_names"]) == 3

        top_errors = data["top_errors"]
        assert isinstance(top_errors, list)
        for err in top_errors:
            assert "index" in err
            assert "features" in err
            assert "actual_label" in err
            assert "predicted_label" in err
            assert err["actual_label"] != err["predicted_label"]
            assert err["confidence"] is not None


def test_evaluate_regression_holdout_dataset(eval_regression_setup, client: TestClient):
    """Verifies that holdout evaluation on a regression dataset computes MSE, RMSE, MAE, R2."""
    job_id = eval_regression_setup["job_id"]
    test_file = eval_regression_setup["test_file"]
    upload_folder = str(eval_regression_setup["upload_folder"])
    exports_base = eval_regression_setup["export_dir"].parent

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = upload_folder

        resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(test_file.id)},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["problem_type"] == "regression"
        assert data["total_samples"] == 25

        metrics = data["metrics"]
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "max_error" in metrics
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]), rel=1e-3)
        assert metrics["max_error"] >= metrics["mae"]

        top_errors = data["top_errors"]
        assert len(top_errors) > 0
        # Check that top errors are sorted descending by error magnitude
        residuals = [err["loss_or_residual"] for err in top_errors]
        assert residuals == sorted(residuals, reverse=True)


def test_evaluate_missing_feature_column_returns_400(eval_classification_setup, client: TestClient, tmp_path: Path):
    """Verifies that evaluation fails with 400 when test CSV misses required features."""
    job_id = eval_classification_setup["job_id"]
    upload_folder = eval_classification_setup["upload_folder"]
    exports_base = eval_classification_setup["export_dir"].parent

    # Create CSV without feat_b
    bad_df = pd.DataFrame({"feat_a": [1.0, 2.0], "species": ["setosa", "versicolor"]})
    bad_csv = f"bad_{uuid4().hex[:6]}.csv"
    bad_df.to_csv(upload_folder / bad_csv, index=False)

    from sqlmodel import Session

    from app.database import engine

    with Session(engine) as s:
        bad_file = DataFile(file_name="bad.csv", disk_name=bad_csv, file_type="csv")
        s.add(bad_file)
        s.commit()
        s.refresh(bad_file)
        bad_file_id = bad_file.id

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)

        resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(bad_file_id)},
        )

        assert resp.status_code == 400
        error_msg = (resp.json().get("message") or resp.json().get("detail", "")).lower()
        assert "missing required feature columns" in error_msg
        assert "feat_b" in error_msg


def test_evaluate_missing_target_column_returns_400(eval_classification_setup, client: TestClient):
    """Verifies that evaluation fails with 400 when test CSV misses the target column."""
    job_id = eval_classification_setup["job_id"]
    upload_folder = eval_classification_setup["upload_folder"]
    exports_base = eval_classification_setup["export_dir"].parent

    no_target_df = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [10.0, 20.0]})
    no_target_csv = f"notarget_{uuid4().hex[:6]}.csv"
    no_target_df.to_csv(upload_folder / no_target_csv, index=False)

    from sqlmodel import Session

    from app.database import engine

    with Session(engine) as s:
        no_target_file = DataFile(file_name="notarget.csv", disk_name=no_target_csv, file_type="csv")
        s.add(no_target_file)
        s.commit()
        s.refresh(no_target_file)
        no_target_file_id = no_target_file.id

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)

        resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(no_target_file_id)},
        )

        assert resp.status_code == 400
        error_msg = (resp.json().get("message") or resp.json().get("detail", "")).lower()
        assert "target column 'species' not found" in error_msg


def test_evaluate_empty_dataset_returns_400(eval_classification_setup, client: TestClient):
    """Verifies that evaluation fails with 400 when dataset file is empty."""
    job_id = eval_classification_setup["job_id"]
    upload_folder = eval_classification_setup["upload_folder"]
    exports_base = eval_classification_setup["export_dir"].parent

    empty_csv = f"empty_{uuid4().hex[:6]}.csv"
    (upload_folder / empty_csv).write_text("")

    from sqlmodel import Session

    from app.database import engine

    with Session(engine) as s:
        empty_file = DataFile(file_name="empty.csv", disk_name=empty_csv, file_type="csv")
        s.add(empty_file)
        s.commit()
        s.refresh(empty_file)
        empty_file_id = empty_file.id

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)

        resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(empty_file_id)},
        )

        assert resp.status_code == 400
        error_msg = (resp.json().get("message") or resp.json().get("detail", "")).lower()
        assert "empty" in error_msg


def test_evaluate_image_classification_guard_returns_400(db_session: Session, client: TestClient, tmp_path: Path):
    """Verifies that image classification jobs return HTTP 400 guard."""
    file_id = uuid4()
    data_file = DataFile(id=file_id, file_name="img.csv", disk_name="img.csv", file_type="csv")
    db_session.add(data_file)
    db_session.commit()

    model = ModelBasic(
        model_name=f"img_model_{uuid4().hex[:6]}",
        file_id=file_id,
        target_field="label",
        model_type=ProblemType.IMAGE_CLASSIFICATION,
    )
    db_session.add(model)
    db_session.commit()

    job_id = str(uuid4())
    job = TrainingJob(id=job_id, model_id=model.id, status=TrainingStatus.COMPLETED)
    db_session.add(job)
    db_session.commit()

    resp = client.post(
        f"/api/v1/model/analysis/{job_id}/evaluate",
        json={"file_id": str(file_id)},
    )
    assert resp.status_code == 400
    error_msg = (resp.json().get("message") or resp.json().get("detail", "")).lower()
    assert "supported for tabular models" in error_msg


def test_evaluate_incomplete_and_nonexistent_jobs(db_session: Session, client: TestClient):
    """Verifies that incomplete jobs return 400 and non-existent jobs return 404."""
    file_id = uuid4()
    data_file = DataFile(id=file_id, file_name="df.csv", disk_name="df.csv", file_type="csv")
    db_session.add(data_file)
    db_session.commit()

    model = ModelBasic(
        model_name=f"pending_model_{uuid4().hex[:6]}",
        file_id=file_id,
        target_field="label",
        model_type=ProblemType.CLASSIFICATION,
    )
    db_session.add(model)
    db_session.commit()

    pending_job = TrainingJob(id="job-pending-test", model_id=model.id, status=TrainingStatus.RUNNING)
    db_session.add(pending_job)
    db_session.commit()

    # Incomplete job -> 400
    resp_pending = client.post(
        "/api/v1/model/analysis/job-pending-test/evaluate",
        json={"file_id": str(file_id)},
    )
    assert resp_pending.status_code == 400
    error_msg = (resp_pending.json().get("message") or resp_pending.json().get("detail", "")).lower()
    assert "only available for completed jobs" in error_msg

    # Non-existent job -> 404
    resp_missing = client.post(
        f"/api/v1/model/analysis/{uuid4()}/evaluate",
        json={"file_id": str(file_id)},
    )
    assert resp_missing.status_code == 404


def test_batch_predict_and_export_csv(eval_classification_setup, client: TestClient):
    """Verifies batch prediction preview and streaming CSV export."""
    job_id = eval_classification_setup["job_id"]
    test_file = eval_classification_setup["test_file"]
    upload_folder = str(eval_classification_setup["upload_folder"])
    exports_base = eval_classification_setup["export_dir"].parent

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = upload_folder

        # 1. Batch predict preview
        resp_pred = client.post(
            f"/api/v1/model/analysis/{job_id}/batch-predict",
            json={"file_id": str(test_file.id)},
        )
        assert resp_pred.status_code == 200
        pred_data = resp_pred.json()
        assert pred_data["total_rows"] == 30
        assert "predicted_class" in pred_data["columns"]
        assert "confidence" in pred_data["columns"]
        assert len(pred_data["preview_rows"]) <= 20
        assert "predicted_class" in pred_data["preview_rows"][0]

        # 2. Batch predict CSV export
        resp_export = client.post(
            f"/api/v1/model/analysis/{job_id}/batch-predict/export",
            json={"file_id": str(test_file.id)},
        )
        assert resp_export.status_code == 200
        assert "text/csv" in resp_export.headers["content-type"]
        assert f"predictions_{job_id[:8]}.csv" in resp_export.headers["content-disposition"]
        csv_text = resp_export.text
        lines = csv_text.strip().split("\n")
        assert len(lines) == 31  # 1 header + 30 rows
        header = lines[0]
        assert "predicted_class" in header
        assert "confidence" in header
