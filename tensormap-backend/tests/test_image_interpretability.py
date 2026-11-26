"""Tests for post-training interpretability on image classification models."""

import os
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from app.database import get_db
from app.main import app
from app.models.data import DataFile, ImageProperties
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.interpretability import InterpretabilityService
from app.shared.enums import ProblemType

client = TestClient(app)


@pytest.fixture(name="db_session")
def fixture_db_session():
    """In-memory SQLite session fixture."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(name="image_job_setup")
def fixture_image_job_setup(db_session, tmp_path):
    """Set up an image classification model and completed training job with dataset."""
    file_id = uuid4()
    data_file = DataFile(
        id=file_id,
        file_name="flowers.zip",
        file_type="zip",
        disk_name="flowers_archive.zip",
    )
    db_session.add(data_file)

    img_props = ImageProperties(
        id=file_id,
        image_size=64,
        batch_size=16,
        color_mode="rgb",
        label_mode="categorical",
    )
    db_session.add(img_props)

    model = ModelBasic(
        model_name="test_image_cnn",
        file_id=file_id,
        model_type=int(ProblemType.IMAGE_CLASSIFICATION),
        training_split=80.0,
        batch_size=16,
        optimizer="adam",
        metric="accuracy",
        epochs=5,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        hyperparams={"epochs": 5, "batch_size": 16},
        analysis_cache={},
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    # Create dummy directory structure
    extracted_dir = tmp_path / "flowers_archive"
    os.makedirs(extracted_dir / "daisy", exist_ok=True)
    os.makedirs(extracted_dir / "rose", exist_ok=True)

    return {
        "job_id": job_id,
        "model": model,
        "data_file": data_file,
        "image_props": img_props,
        "extracted_dir": str(extracted_dir),
    }


def test_load_image_test_dataset_missing_job(db_session):
    """Raises ValueError when training job ID does not exist."""
    service = InterpretabilityService()
    with pytest.raises(ValueError, match="Training job not found"):
        service._load_image_test_dataset("non-existent-job-id", db_session)


def test_load_image_test_dataset_non_zip_file(db_session):
    """Raises ValueError when dataset file is not a zip archive."""
    file_id = uuid4()
    csv_file = DataFile(
        id=file_id,
        file_name="tabular.csv",
        file_type="csv",
        disk_name="tabular.csv",
    )
    db_session.add(csv_file)

    model = ModelBasic(
        model_name="img_model_with_csv",
        file_id=file_id,
        model_type=int(ProblemType.IMAGE_CLASSIFICATION),
    )
    db_session.add(model)
    db_session.commit()

    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
    )
    db_session.add(job)
    db_session.commit()

    service = InterpretabilityService()
    with pytest.raises(ValueError, match="Image dataset must be a ZIP archive"):
        service._load_image_test_dataset(job_id, db_session)


def test_load_image_test_dataset_missing_directory(image_job_setup, db_session):
    """Raises FileNotFoundError when extracted image directory does not exist."""
    job_id = image_job_setup["job_id"]
    service = InterpretabilityService()
    with (
        patch("app.services.model_run.get_settings") as mock_settings,
        pytest.raises(FileNotFoundError, match="Extracted image directory not found"),
    ):
        mock_settings.return_value.upload_folder = "/non/existent/path"
        service._load_image_test_dataset(job_id, db_session)


def test_load_image_test_dataset_success(image_job_setup, db_session, tmp_path):
    """Successfully resolves parameters and loads image dataset from directory."""
    job_id = image_job_setup["job_id"]
    service = InterpretabilityService()

    mock_dataset = MagicMock()
    mock_dataset.class_names = ["daisy", "rose"]

    with (
        patch("app.services.model_run.get_settings") as mock_settings,
        patch("tensorflow.keras.utils.image_dataset_from_directory", return_value=mock_dataset) as mock_loader,
    ):
        mock_settings.return_value.upload_folder = str(tmp_path)
        ds, class_names = service._load_image_test_dataset(job_id, db_session)

        assert class_names == ["daisy", "rose"]
        assert ds is mock_dataset
        mock_loader.assert_called_once()
        _, kwargs = mock_loader.call_args
        assert kwargs["validation_split"] == pytest.approx(0.2, abs=1e-3)
        assert kwargs["subset"] == "validation"
        assert kwargs["seed"] == 123
        assert kwargs["image_size"] == (64, 64)
        assert kwargs["color_mode"] == "rgb"
        assert kwargs["label_mode"] == "categorical"


def test_image_confusion_matrix_computation(image_job_setup, db_session):
    """Computes confusion matrix and classification report for image models."""
    job_id = image_job_setup["job_id"]
    service = InterpretabilityService()

    # Mock evaluation: 4 samples across 2 batches and 2 classes (daisy=0, rose=1)
    y_true_batches = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]
    mock_batch1 = (MagicMock(), MagicMock(numpy=lambda: y_true_batches[0]))
    mock_batch2 = (MagicMock(), MagicMock(numpy=lambda: y_true_batches[1]))
    mock_ds = [mock_batch1, mock_batch2]

    # Model predictions: 3 correct, 1 misclassified (delivered per batch)
    mock_preds_b1 = np.array([[0.9, 0.1], [0.2, 0.8]])  # daisy->0 (ok), rose->1 (ok)
    mock_preds_b2 = np.array([[0.3, 0.7], [0.1, 0.9]])  # daisy->1 (err), rose->1 (ok)

    mock_model = MagicMock()
    mock_model.predict.side_effect = [mock_preds_b1, mock_preds_b2]

    with (
        patch.object(service, "_load_image_test_dataset", return_value=(mock_ds, ["daisy", "rose"])),
        patch("tensorflow.keras.models.load_model", return_value=mock_model),
        patch("pathlib.Path.exists", return_value=True),
    ):
        res = service._compute_confusion_matrix_sync(job_id, db_session)

        assert res["analysis_type"] == "image_classification"
        assert res["class_names"] == ["daisy", "rose"]
        assert res["n_samples"] == 4
        assert res["overall_accuracy"] == 0.75
        assert res["confusion_matrix"] == [[1, 1], [0, 2]]
        assert "daisy" in res["classification_report"]
        assert "rose" in res["classification_report"]


def test_image_roc_curve_computation(image_job_setup, db_session):
    """Computes ROC and PR curves with AUC scores for image models."""
    job_id = image_job_setup["job_id"]
    service = InterpretabilityService()

    y_true_batches = [
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
    ]
    mock_ds = [(MagicMock(), MagicMock(numpy=lambda: y_true_batches[0]))]
    mock_preds = np.array(
        [
            [0.85, 0.15],
            [0.10, 0.90],
            [0.70, 0.30],
            [0.20, 0.80],
        ]
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = mock_preds

    with (
        patch.object(service, "_load_image_test_dataset", return_value=(mock_ds, ["daisy", "rose"])),
        patch("tensorflow.keras.models.load_model", return_value=mock_model),
        patch("pathlib.Path.exists", return_value=True),
    ):
        res = service._compute_roc_curve_sync(job_id, db_session)

        assert res["analysis_type"] == "image_classification"
        assert res["class_names"] == ["daisy", "rose"]
        assert res["n_samples"] == 4
        assert res["macro_roc_auc"] == 1.0
        assert res["macro_pr_auc"] == 1.0
        assert len(res["classes"]) == 2


def test_image_calibration_curve_computation(image_job_setup, db_session):
    """Computes calibration curves and reliability metrics for image models."""
    job_id = image_job_setup["job_id"]
    service = InterpretabilityService()

    y_true_batches = [
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
    ]
    mock_ds = [(MagicMock(), MagicMock(numpy=lambda: y_true_batches[0]))]
    mock_preds = np.array(
        [
            [0.85, 0.15],
            [0.10, 0.90],
            [0.70, 0.30],
            [0.20, 0.80],
        ]
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = mock_preds

    with (
        patch.object(service, "_load_image_test_dataset", return_value=(mock_ds, ["daisy", "rose"])),
        patch("tensorflow.keras.models.load_model", return_value=mock_model),
        patch("pathlib.Path.exists", return_value=True),
    ):
        res = service._compute_calibration_curve_sync(job_id, db_session, n_bins=5)

        assert res["analysis_type"] == "image_classification"
        assert res["class_names"] == ["daisy", "rose"]
        assert res["n_samples"] == 4
        assert res["n_bins"] == 5
        assert len(res["classes"]) == 2
        assert "macro_brier_score" in res


def test_image_analysis_endpoints_return_200(image_job_setup, db_session):
    """FastAPI endpoints for confusion matrix, ROC, and calibration return 200 for image models."""
    job_id = image_job_setup["job_id"]

    mock_cm_payload = {
        "confusion_matrix": [[5, 1], [0, 6]],
        "classification_report": {"accuracy": 0.9167},
        "class_names": ["cat", "dog"],
        "overall_accuracy": 0.9167,
        "n_samples": 12,
        "analysis_type": "image_classification",
    }
    mock_roc_payload = {
        "classes": [],
        "macro_roc_auc": 0.95,
        "macro_pr_auc": 0.92,
        "is_binary": True,
        "class_names": ["cat", "dog"],
        "n_samples": 12,
        "analysis_type": "image_classification",
    }
    mock_cal_payload = {
        "classes": [],
        "macro_brier_score": 0.08,
        "class_names": ["cat", "dog"],
        "n_samples": 12,
        "n_bins": 10,
        "analysis_type": "image_classification",
    }

    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with (
            patch("app.routers.analysis._service.compute_confusion_matrix_async", return_value=mock_cm_payload),
            patch("app.routers.analysis._service.compute_roc_curve_async", return_value=mock_roc_payload),
            patch("app.routers.analysis._service.compute_calibration_curve_async", return_value=mock_cal_payload),
        ):
            cm_resp = client.get(f"/api/v1/model/analysis/{job_id}/confusion-matrix")
            assert cm_resp.status_code == 200
            assert cm_resp.json()["analysis_type"] == "image_classification"

            roc_resp = client.get(f"/api/v1/model/analysis/{job_id}/roc-curve")
            assert roc_resp.status_code == 200
            assert roc_resp.json()["analysis_type"] == "image_classification"

            cal_resp = client.get(f"/api/v1/model/analysis/{job_id}/calibration-curve")
            assert cal_resp.status_code == 200
            assert cal_resp.json()["analysis_type"] == "image_classification"
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_get_analysis_type_raises_on_missing_job_or_model(db_session):
    """InterpretabilityService.get_analysis_type raises LookupError instead of defaulting silently."""
    service = InterpretabilityService()
    with pytest.raises(LookupError, match="Training job not found"):
        service.get_analysis_type("non-existent-job", db_session)

    orphan_job_id = str(uuid4())
    db_session.add(TrainingJob(id=orphan_job_id, model_id=99999, status=TrainingStatus.COMPLETED))
    db_session.commit()

    with pytest.raises(LookupError, match="Model not found"):
        service.get_analysis_type(orphan_job_id, db_session)


def test_real_image_directory_single_pass_alignment(tmp_path, monkeypatch):
    """Real multi-batch tf.data directory evaluation keeps y_true and y_pred strictly aligned."""
    import tensorflow as tf

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        file_id = uuid4()
        session.add(DataFile(id=file_id, file_name="bw.zip", file_type="zip", disk_name="bw_archive.zip"))
        session.add(
            ImageProperties(id=file_id, image_size=28, batch_size=2, color_mode="rgb", label_mode="categorical")
        )
        model_record = ModelBasic(
            model_name="bw_cnn",
            file_id=file_id,
            model_type=int(ProblemType.IMAGE_CLASSIFICATION),
            training_split=40.0,
            batch_size=2,
            optimizer="adam",
            metric="accuracy",
            epochs=1,
        )
        session.add(model_record)
        session.commit()
        session.refresh(model_record)

        job_id = str(uuid4())
        session.add(
            TrainingJob(
                id=job_id,
                model_id=model_record.id,
                status=TrainingStatus.COMPLETED,
                hyperparams={"epochs": 1, "batch_size": 2},
                analysis_cache={},
            )
        )
        session.commit()

        # Write real PNG files: class 'black' (0s) and class 'white' (255s)
        upload_dir = tmp_path / "uploads"
        extracted = upload_dir / "bw_archive"
        (extracted / "black").mkdir(parents=True)
        (extracted / "white").mkdir(parents=True)
        black_png = tf.io.encode_png(tf.constant(np.zeros((28, 28, 3), dtype=np.uint8), dtype=tf.uint8)).numpy()
        white_png = tf.io.encode_png(tf.constant(np.full((28, 28, 3), 255, dtype=np.uint8), dtype=tf.uint8)).numpy()
        for i in range(6):
            (extracted / "black" / f"b_{i}.png").write_bytes(black_png)
            (extracted / "white" / f"w_{i}.png").write_bytes(white_png)

        # Save real deterministic Keras model mapping dark pixels -> class 0, bright pixels -> class 1
        exports_dir = tmp_path / "exports"
        model_dir = exports_dir / job_id
        model_dir.mkdir(parents=True)
        keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(28, 28, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )
        dense_layer = keras_model.layers[-1]
        kernel = np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]], dtype=np.float32)
        bias = np.array([5.0, -5.0], dtype=np.float32)
        dense_layer.set_weights([kernel, bias])
        keras_model.save(model_dir / "model.keras")

        monkeypatch.setattr("app.services.interpretability.EXPORTS_BASE", exports_dir)
        service = InterpretabilityService()

        with patch("app.services.model_run.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(upload_dir)
            cm_res = service._compute_confusion_matrix_sync(job_id, session)
            roc_res = service._compute_roc_curve_sync(job_id, session)

        assert cm_res["analysis_type"] == "image_classification"
        assert cm_res["class_names"] == ["black", "white"]
        assert cm_res["n_samples"] == 7  # 60% of 12 images across 4 shuffled batches
        assert cm_res["overall_accuracy"] == pytest.approx(1.0)
        assert cm_res["confusion_matrix"][0][1] == 0
        assert cm_res["confusion_matrix"][1][0] == 0
        assert roc_res["macro_roc_auc"] == pytest.approx(1.0)
