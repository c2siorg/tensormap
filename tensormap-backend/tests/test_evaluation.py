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


def test_evaluate_and_batch_predict_drops_nan_rows_and_preserves_csv_row_index(
    eval_classification_setup, client: TestClient, db_session: Session
):
    """Tests that NaNs are dropped consistently in evaluate and batch-predict,

    and original CSV row index is preserved.
    """
    job_id = eval_classification_setup["job_id"]
    upload_folder = eval_classification_setup["upload_folder"]
    exports_base = eval_classification_setup["export_dir"].parent

    # Create dataset with 10 rows, rows 2 and 5 contain NaN in feat_a
    n_rows = 10
    feat_a = [1.5, 2.5, np.nan, 3.5, 4.5, np.nan, 2.0, 3.0, 1.0, 4.0]
    feat_b = [12.0] * n_rows
    species = ["setosa"] * n_rows

    nan_df = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "species": species})
    disk_name = f"test_nan_{uuid4().hex[:8]}.csv"
    nan_file_path = upload_folder / disk_name
    nan_df.to_csv(nan_file_path, index=False)

    data_file = DataFile(
        file_name="nan_test.csv",
        file_type="csv",
        disk_name=disk_name,
        columns=["feat_a", "feat_b", "species"],
        row_count=n_rows,
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)

        # 1. Evaluation drops the 2 NaN rows (total 8 valid rows)
        resp_eval = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(data_file.id), "target_field": "species"},
        )
        assert resp_eval.status_code == 200
        eval_data = resp_eval.json()
        assert eval_data["total_samples"] == 8

        # 2. Batch predict also drops the 2 NaN rows (total 8 valid rows) instead of fillna(0)
        resp_pred = client.post(
            f"/api/v1/model/analysis/{job_id}/batch-predict",
            json={"file_id": str(data_file.id)},
        )
        assert resp_pred.status_code == 200
        pred_data = resp_pred.json()
        assert pred_data["total_rows"] == 8


def test_evaluate_cross_project_dataset_forbidden_403(
    eval_classification_setup, client: TestClient, db_session: Session
):
    """Tests that evaluating a dataset from a different project returns 403."""
    job_id = eval_classification_setup["job_id"]
    test_file = eval_classification_setup["test_file"]
    job = db_session.get(TrainingJob, job_id)
    model = db_session.get(ModelBasic, job.model_id)

    # Seed valid Project records to satisfy foreign key constraints
    from app.models.project import Project

    proj_a = Project(name=f"Project_A_{uuid4().hex[:6]}")
    proj_b = Project(name=f"Project_B_{uuid4().hex[:6]}")
    db_session.add(proj_a)
    db_session.add(proj_b)
    db_session.commit()
    db_session.refresh(proj_a)
    db_session.refresh(proj_b)

    model.project_id = proj_a.id
    test_file.project_id = proj_b.id
    db_session.add(model)
    db_session.add(test_file)
    db_session.commit()

    upload_folder = str(eval_classification_setup["upload_folder"])
    exports_base = eval_classification_setup["export_dir"].parent

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = upload_folder

        resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(test_file.id), "target_field": "species"},
        )
        assert resp.status_code == 403
        data = resp.json()
        assert "does not belong to the same project" in (data.get("message") or data.get("detail", ""))


def test_evaluate_holdout_missing_class_matrix_shape(
    eval_classification_setup, client: TestClient, db_session: Session
):
    """Tests that a holdout set missing a class still generates a full-dimension confusion matrix."""
    job_id = eval_classification_setup["job_id"]
    upload_folder = eval_classification_setup["upload_folder"]
    exports_base = eval_classification_setup["export_dir"].parent

    # Create holdout data with only species 'setosa' (missing virginica and versicolor)
    missing_class_df = pd.DataFrame(
        {
            "feat_a": [1.5, 2.5, 3.5],
            "feat_b": [11.0, 12.0, 13.0],
            "species": ["setosa", "setosa", "setosa"],
        }
    )
    disk_name = f"missing_class_{uuid4().hex[:8]}.csv"
    missing_class_df.to_csv(upload_folder / disk_name, index=False)
    test_file = DataFile(
        file_name="missing_class.csv",
        file_type="csv",
        disk_name=disk_name,
        columns=list(missing_class_df.columns),
        row_count=3,
    )
    db_session.add(test_file)
    db_session.commit()
    db_session.refresh(test_file)

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)
        resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(test_file.id), "target_field": "species"},
        )
        assert resp.status_code == 200
        metrics = resp.json()["metrics"]
        cm = metrics["confusion_matrix"]
        class_names = metrics["class_names"]
        # Matrix dimension must match number of classes even if some classes have 0 occurrences
        assert len(cm) == len(class_names)
        assert len(cm[0]) == len(class_names)


def test_evaluate_regression_non_numeric_target_returns_400(
    eval_regression_setup, client: TestClient, db_session: Session
):
    """Tests that a regression target containing non-numeric values returns a 400 error."""
    job_id = eval_regression_setup["job_id"]
    upload_folder = eval_regression_setup["upload_folder"]
    exports_base = eval_regression_setup["export_dir"].parent

    bad_reg_df = pd.DataFrame(
        {
            "size": [100.0, 150.0],
            "rooms": [2.0, 3.0],
            "price": ["not_a_number", "invalid_price"],
        }
    )
    disk_name = f"bad_reg_{uuid4().hex[:8]}.csv"
    bad_reg_df.to_csv(upload_folder / disk_name, index=False)
    bad_file = DataFile(
        file_name="bad_reg.csv",
        file_type="csv",
        disk_name=disk_name,
        columns=list(bad_reg_df.columns),
        row_count=2,
    )
    db_session.add(bad_file)
    db_session.commit()
    db_session.refresh(bad_file)

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)
        resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(bad_file.id), "target_field": "price"},
        )
        assert resp.status_code == 400
        assert "non-numeric values for regression model" in (
            resp.json().get("message") or resp.json().get("detail", "")
        )


def test_export_csv_sanitizes_formula_injection(eval_classification_setup, client: TestClient, db_session: Session):
    """Tests that CSV export sanitizes formulas starting with '=', '+', '-', or '@'."""
    job_id = eval_classification_setup["job_id"]
    upload_folder = eval_classification_setup["upload_folder"]
    exports_base = eval_classification_setup["export_dir"].parent

    formula_df = pd.DataFrame(
        {
            "feat_a": [2.0],
            "feat_b": [15.0],
            "species": ["=cmd|' /C calc'!A0"],
        }
    )
    disk_name = f"formula_{uuid4().hex[:8]}.csv"
    formula_df.to_csv(upload_folder / disk_name, index=False)
    formula_file = DataFile(
        file_name="formula.csv",
        file_type="csv",
        disk_name=disk_name,
        columns=list(formula_df.columns),
        row_count=1,
    )
    db_session.add(formula_file)
    db_session.commit()
    db_session.refresh(formula_file)

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", exports_base),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)
        resp = client.post(
            f"/api/v1/model/analysis/{job_id}/batch-predict/export",
            json={"file_id": str(formula_file.id)},
        )
        assert resp.status_code == 200
        csv_text = resp.text
        # Must be prefixed with single quote
        assert "'=cmd|" in csv_text


def test_resolve_classification_predictions_probabilities_and_logits():
    """Verifies probability preservation and logit normalization for binary and multiclass outputs."""
    from scipy.special import expit, softmax

    from app.services.evaluation_service import _resolve_classification_predictions

    # 1. Binary probabilities (emits_logits=False) remain unchanged
    bin_probs_raw = np.array([[0.1], [0.49], [0.5], [0.85]], dtype=np.float32)
    classes, probs = _resolve_classification_predictions(bin_probs_raw, emits_logits=False)
    assert np.array_equal(classes, np.array([0, 0, 1, 1]))
    assert np.allclose(probs[:, 1], bin_probs_raw.flatten(), atol=1e-6)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    # 2. Binary logits, including values strictly inside [0, 1] (e.g. 0.2 -> expit(0.2) > 0.5 -> class 1)
    bin_logits_raw = np.array([[-2.5], [0.0], [0.2], [0.8], [2.4]], dtype=np.float32)
    classes_log, probs_log = _resolve_classification_predictions(bin_logits_raw, emits_logits=True)
    expected_p1 = expit(bin_logits_raw.flatten())
    assert np.allclose(probs_log[:, 1], expected_p1, atol=1e-6)
    assert np.allclose(probs_log[:, 0], 1.0 - expected_p1, atol=1e-6)
    assert np.allclose(probs_log.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(classes_log, np.array([0, 1, 1, 1, 1]))

    # 3. Multiclass probabilities (emits_logits=False) remain unchanged
    multi_probs_raw = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
    classes_mp, probs_mp = _resolve_classification_predictions(multi_probs_raw, emits_logits=False)
    assert np.array_equal(classes_mp, np.array([0, 1]))
    assert np.allclose(probs_mp, multi_probs_raw, atol=1e-6)
    assert np.allclose(probs_mp.sum(axis=1), 1.0, atol=1e-6)

    # 4. Multiclass logits with mixed signs (emits_logits=True)
    multi_logits_raw = np.array([[2.5, -1.0, 0.5], [-0.5, 3.2, 1.1]], dtype=np.float32)
    classes_ml, probs_ml = _resolve_classification_predictions(multi_logits_raw, emits_logits=True)
    assert np.array_equal(classes_ml, np.array([0, 1]))
    assert np.allclose(probs_ml, softmax(multi_logits_raw, axis=1), atol=1e-6)
    assert np.allclose(probs_ml.sum(axis=1), 1.0, atol=1e-6)

    # 5. Multiclass logits containing only positive values (including values in [0, 1] summing to 1.0)
    multi_pos_logits = np.array([[0.4, 0.6, 0.0], [2.0, 1.0, 0.5]], dtype=np.float32)
    classes_mpl, probs_mpl = _resolve_classification_predictions(multi_pos_logits, emits_logits=True)
    assert np.array_equal(classes_mpl, np.array([1, 0]))
    assert np.allclose(probs_mpl, softmax(multi_pos_logits, axis=1), atol=1e-6)
    assert np.allclose(probs_mpl.sum(axis=1), 1.0, atol=1e-6)


def test_evaluate_and_batch_predict_normalize_logits_from_keras_model(
    db_session: Session, client: TestClient, tmp_path: Path
):
    """End-to-end test verifying evaluate and batch-predict normalize raw logits based on model activation."""
    import tensorflow as tf

    upload_folder = tmp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "feat_a": [0.0, 0.1, 1.0, 1.1],
            "label": [0, 0, 1, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "feat_a": [0.2, 0.8],
            "label": [0, 1],
        }
    )
    train_csv = f"train_logits_{uuid4().hex[:6]}.csv"
    test_csv = f"test_logits_{uuid4().hex[:6]}.csv"
    train_df.to_csv(upload_folder / train_csv, index=False)
    test_df.to_csv(upload_folder / test_csv, index=False)

    train_file = DataFile(
        file_name="train_logits.csv",
        disk_name=train_csv,
        file_type="csv",
        total_rows=4,
        total_columns=2,
    )
    test_file = DataFile(
        file_name="test_logits.csv",
        disk_name=test_csv,
        file_type="csv",
        total_rows=2,
        total_columns=2,
    )
    db_session.add(train_file)
    db_session.add(test_file)
    db_session.commit()
    db_session.refresh(train_file)
    db_session.refresh(test_file)

    model = ModelBasic(
        model_name=f"logits_bin_{uuid4().hex[:6]}",
        file_id=train_file.id,
        target_field="label",
        model_type=ProblemType.CLASSIFICATION,
        training_split=80.0,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(id=job_id, model_id=model.id, status=TrainingStatus.COMPLETED)
    db_session.add(job)
    db_session.commit()

    # Build a linear-output (logits) binary model where output == input feat_a (0.2 and 0.8)
    # Both logits lie in (0, 1): sigmoid(0.2) ~= 0.5498 -> class 1 (would be class 0 if clipped!)
    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True, exist_ok=True)
    inputs = tf.keras.Input(shape=(1,))
    outputs = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Ones(),
        bias_initializer=tf.keras.initializers.Zeros(),
    )(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.save(export_dir / "model.keras")

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", tmp_path / "exports"),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)

        pred_resp = client.post(
            f"/api/v1/model/analysis/{job_id}/batch-predict",
            json={"file_id": str(test_file.id)},
        )
        assert pred_resp.status_code == 200
        rows = pred_resp.json()["preview_rows"]
        # expit(0.2) ~= 0.5498 -> predicted class "1", expit(0.8) ~= 0.6900 -> predicted class "1"
        assert rows[0]["predicted_class"] == "1"
        assert rows[0]["confidence"] == pytest.approx(0.5498, abs=1e-3)
        assert rows[0]["prob_0"] + rows[0]["prob_1"] == pytest.approx(1.0, abs=1e-3)
        assert rows[1]["predicted_class"] == "1"
        assert rows[1]["confidence"] == pytest.approx(0.6900, abs=1e-3)

        eval_resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(test_file.id)},
        )
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        # Only sample 1 (label=1, pred=1) is right; sample 0 (label=0, pred=1) is misclassified
        assert eval_data["metrics"]["accuracy"] == pytest.approx(0.5, abs=1e-4)
        assert eval_data["metrics"]["log_loss"] is not None


def test_evaluate_and_batch_predict_normalize_multiclass_logits_from_keras_model(
    db_session: Session, client: TestClient, tmp_path: Path
):
    """End-to-end test verifying multiclass positive-only logits are softmax-normalized."""
    import tensorflow as tf
    from scipy.special import softmax
    from sklearn.metrics import log_loss

    upload_folder = tmp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "f0": [1.0, 0.0, 0.0],
            "f1": [0.0, 1.0, 0.0],
            "f2": [0.0, 0.0, 1.0],
            "cls": [0, 1, 2],
        }
    )
    # Positive-only logits:
    # row 0 -> [2.0, 0.5, 0.1] (class 0)
    # row 1 -> [0.2, 1.5, 0.3] (class 1)
    # row 2 -> [0.1, 0.4, 2.5] (class 2)
    test_df = pd.DataFrame(
        {
            "f0": [2.0, 0.2, 0.1],
            "f1": [0.5, 1.5, 0.4],
            "f2": [0.1, 0.3, 2.5],
            "cls": [0, 1, 2],
        }
    )
    train_csv = f"train_mlogits_{uuid4().hex[:6]}.csv"
    test_csv = f"test_mlogits_{uuid4().hex[:6]}.csv"
    train_df.to_csv(upload_folder / train_csv, index=False)
    test_df.to_csv(upload_folder / test_csv, index=False)

    train_file = DataFile(
        file_name="train_mlogits.csv",
        disk_name=train_csv,
        file_type="csv",
        total_rows=3,
        total_columns=4,
    )
    test_file = DataFile(
        file_name="test_mlogits.csv",
        disk_name=test_csv,
        file_type="csv",
        total_rows=3,
        total_columns=4,
    )
    db_session.add(train_file)
    db_session.add(test_file)
    db_session.commit()
    db_session.refresh(train_file)
    db_session.refresh(test_file)

    model = ModelBasic(
        model_name=f"mlogits_{uuid4().hex[:6]}",
        file_id=train_file.id,
        target_field="cls",
        model_type=ProblemType.CLASSIFICATION,
        training_split=80.0,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(id=job_id, model_id=model.id, status=TrainingStatus.COMPLETED)
    db_session.add(job)
    db_session.commit()

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True, exist_ok=True)
    inputs = tf.keras.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(
        3,
        activation="linear",
        kernel_initializer=tf.keras.initializers.Identity(),
        bias_initializer=tf.keras.initializers.Zeros(),
    )(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.save(export_dir / "model.keras")

    raw_logits = test_df[["f0", "f1", "f2"]].to_numpy(dtype=np.float32)
    expected_probs = softmax(raw_logits, axis=1)
    expected_log_loss = float(round(log_loss([0, 1, 2], expected_probs), 4))

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", tmp_path / "exports"),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)

        pred_resp = client.post(
            f"/api/v1/model/analysis/{job_id}/batch-predict",
            json={"file_id": str(test_file.id)},
        )
        assert pred_resp.status_code == 200
        rows = pred_resp.json()["preview_rows"]
        for i, expected_cls in enumerate(["0", "1", "2"]):
            assert rows[i]["predicted_class"] == expected_cls
            assert rows[i]["prob_0"] == pytest.approx(float(expected_probs[i, 0]), abs=1e-3)
            assert rows[i]["prob_1"] == pytest.approx(float(expected_probs[i, 1]), abs=1e-3)
            assert rows[i]["prob_2"] == pytest.approx(float(expected_probs[i, 2]), abs=1e-3)
            assert rows[i]["prob_0"] + rows[i]["prob_1"] + rows[i]["prob_2"] == pytest.approx(1.0, abs=1e-3)

        eval_resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(test_file.id)},
        )
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert eval_data["metrics"]["accuracy"] == pytest.approx(1.0, abs=1e-4)
        assert eval_data["metrics"]["log_loss"] == pytest.approx(expected_log_loss, abs=1e-3)


def test_evaluate_and_batch_predict_preserve_binary_sigmoid_probabilities(
    db_session: Session, client: TestClient, tmp_path: Path
):
    """End-to-end test verifying binary sigmoid models preserve probabilities without double-sigmoid."""
    import tensorflow as tf
    from scipy.special import expit

    upload_folder = tmp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "feat_a": [-2.0, -1.0, 1.0, 2.0],
            "label": [0, 0, 1, 1],
        }
    )
    # Sigmoid(feat_a): sigmoid(-1.0) ~= 0.2689 -> class 0; sigmoid(1.0) ~= 0.7311 -> class 1
    # If sigmoid were applied twice: expit(0.2689) ~= 0.5668 -> class 1 (wrong!)
    test_df = pd.DataFrame(
        {
            "feat_a": [-1.0, 1.0],
            "label": [0, 1],
        }
    )
    train_csv = f"train_sig_{uuid4().hex[:6]}.csv"
    test_csv = f"test_sig_{uuid4().hex[:6]}.csv"
    train_df.to_csv(upload_folder / train_csv, index=False)
    test_df.to_csv(upload_folder / test_csv, index=False)

    train_file = DataFile(
        file_name="train_sig.csv",
        disk_name=train_csv,
        file_type="csv",
        total_rows=4,
        total_columns=2,
    )
    test_file = DataFile(
        file_name="test_sig.csv",
        disk_name=test_csv,
        file_type="csv",
        total_rows=2,
        total_columns=2,
    )
    db_session.add(train_file)
    db_session.add(test_file)
    db_session.commit()
    db_session.refresh(train_file)
    db_session.refresh(test_file)

    model = ModelBasic(
        model_name=f"sig_bin_{uuid4().hex[:6]}",
        file_id=train_file.id,
        target_field="label",
        model_type=ProblemType.CLASSIFICATION,
        training_split=80.0,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(id=job_id, model_id=model.id, status=TrainingStatus.COMPLETED)
    db_session.add(job)
    db_session.commit()

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True, exist_ok=True)
    inputs = tf.keras.Input(shape=(1,))
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=tf.keras.initializers.Ones(),
        bias_initializer=tf.keras.initializers.Zeros(),
    )(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.save(export_dir / "model.keras")

    expected_p0 = float(expit(-1.0))
    expected_p1 = float(expit(1.0))

    with (
        patch("app.services.evaluation_service.get_settings") as mock_settings,
        patch("app.services.evaluation_service.EXPORTS_BASE", tmp_path / "exports"),
    ):
        mock_settings.return_value.upload_folder = str(upload_folder)

        pred_resp = client.post(
            f"/api/v1/model/analysis/{job_id}/batch-predict",
            json={"file_id": str(test_file.id)},
        )
        assert pred_resp.status_code == 200
        rows = pred_resp.json()["preview_rows"]
        assert rows[0]["predicted_class"] == "0"
        assert rows[0]["prob_1"] == pytest.approx(expected_p0, abs=1e-3)
        assert rows[1]["predicted_class"] == "1"
        assert rows[1]["prob_1"] == pytest.approx(expected_p1, abs=1e-3)

        eval_resp = client.post(
            f"/api/v1/model/analysis/{job_id}/evaluate",
            json={"file_id": str(test_file.id)},
        )
        assert eval_resp.status_code == 200
        assert eval_resp.json()["metrics"]["accuracy"] == pytest.approx(1.0, abs=1e-4)
