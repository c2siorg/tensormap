"""Tests for Training Run Reproducibility Metadata (Issue #520)."""

import os
import uuid as uuid_pkg

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.reproducibility import (
    capture_reproducibility_snapshot,
    compute_file_fingerprint,
    get_runtime_environment,
)
from app.services.training_service import create_training_job
from app.shared.enums import ProblemType

BASE = "/api/v1/model"


def test_compute_file_fingerprint_deterministic(tmp_path):
    """Identical file contents must produce identical deterministic SHA-256 fingerprints."""
    file1 = tmp_path / "file1.csv"
    file2 = tmp_path / "file2.csv"
    file3 = tmp_path / "file3.csv"

    content = b"feature1,feature2,target\n1.0,2.0,0\n3.0,4.0,1\n"
    file1.write_bytes(content)
    file2.write_bytes(content)
    file3.write_bytes(b"different,content\n1,2\n")

    fp1 = compute_file_fingerprint(str(file1))
    fp2 = compute_file_fingerprint(str(file2))
    fp3 = compute_file_fingerprint(str(file3))

    assert fp1 is not None
    assert fp1.startswith("sha256:")
    assert fp1 == fp2
    assert fp1 != fp3


def test_compute_file_fingerprint_directory(tmp_path):
    """Directory tree hashing must be deterministic regardless of file discovery order."""
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"

    dir1.mkdir()
    dir2.mkdir()

    (dir1 / "b.txt").write_text("file b")
    (dir1 / "a.txt").write_text("file a")

    (dir2 / "a.txt").write_text("file a")
    (dir2 / "b.txt").write_text("file b")

    fp1 = compute_file_fingerprint(str(dir1))
    fp2 = compute_file_fingerprint(str(dir2))

    assert fp1 is not None
    assert fp1 == fp2


def test_compute_file_fingerprint_missing_file():
    """Non-existent file should gracefully return None without raising exceptions."""
    assert compute_file_fingerprint("/non/existent/path/here.csv") is None
    assert compute_file_fingerprint(None) is None


def test_get_runtime_environment_no_secrets():
    """Runtime environment returns expected versions and contains no env vars or tokens."""
    env_info = get_runtime_environment()
    assert "python_version" in env_info
    assert "platform" in env_info
    assert "libraries" in env_info
    assert "tensorflow" in env_info["libraries"]
    assert "numpy" in env_info["libraries"]
    assert "pandas" in env_info["libraries"]

    # Verify no arbitrary environment keys leaked
    assert "SECRET_KEY" not in env_info
    assert "DATABASE_URL" not in env_info
    assert "PATH" not in env_info


def test_capture_reproducibility_snapshot(db_session: Session, tmp_path, monkeypatch):
    """Snapshot accurately records dataset fingerprint, configuration, seed, and environment."""
    from app.config import get_settings

    upload_folder = str(tmp_path / "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    monkeypatch.setattr(get_settings(), "upload_folder", upload_folder)

    # Create dummy data file
    disk_name = "iris_test.csv"
    data_path = os.path.join(upload_folder, disk_name)
    with open(data_path, "wb") as f:
        f.write(b"col1,col2,target\n1,2,0\n3,4,1\n")

    data_file = DataFile(
        file_name="iris.csv",
        file_type="csv",
        disk_name=disk_name,
        row_count=2,
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    model = ModelBasic(
        model_name="reproducibility_test_model",
        file_id=data_file.id,
        model_type=ProblemType.CLASSIFICATION,
        target_field="target",
        optimizer="adam",
        metric="accuracy",
        loss="sparse_categorical_crossentropy",
        epochs=15,
        batch_size=16,
        training_split=80.0,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    snapshot = capture_reproducibility_snapshot(
        session=db_session,
        model=model,
        hyperparams={"lr": 0.005, "epochs": 20, "seed": 999},
    )

    assert snapshot["dataset"]["file_name"] == "iris.csv"
    assert snapshot["dataset"]["fingerprint"].startswith("sha256:")
    assert snapshot["dataset"]["row_count"] == 2
    assert snapshot["training_config"]["problem_type"] == "classification"
    assert snapshot["training_config"]["target_field"] == "target"
    assert snapshot["training_config"]["epochs"] == 20  # overridden by hyperparams
    assert snapshot["training_config"]["learning_rate"] == 0.005
    assert snapshot["random_seed"] == 999  # custom seed


def test_capture_reproducibility_snapshot_default_seeds(db_session: Session):
    """Default seeds are 42 for tabular and 123 for image classification."""
    model_tabular = ModelBasic(
        model_name="tabular_model",
        model_type=ProblemType.REGRESSION,
        target_field="price",
        optimizer="adam",
        metric="mse",
        loss="mean_squared_error",
    )
    model_image = ModelBasic(
        model_name="image_model",
        model_type=ProblemType.IMAGE_CLASSIFICATION,
        optimizer="adam",
        metric="accuracy",
        loss="sparse_categorical_crossentropy",
    )
    db_session.add(model_tabular)
    db_session.add(model_image)
    db_session.commit()

    snap_tabular = capture_reproducibility_snapshot(db_session, model_tabular)
    snap_image = capture_reproducibility_snapshot(db_session, model_image)

    assert snap_tabular["random_seed"] == 42
    assert snap_image["random_seed"] == 123


def test_create_training_job_persists_reproducibility(db_session: Session):
    """create_training_job automatically populates and persists reproducibility metadata."""
    model = ModelBasic(
        model_name="job_repro_model",
        model_type=ProblemType.CLASSIFICATION,
        optimizer="adam",
        metric="accuracy",
        loss="sparse_categorical_crossentropy",
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job = create_training_job(
        model_id=model.id,
        project_id=None,
        hyperparams={"epochs": 5},
        session=db_session,
    )

    assert job.reproducibility is not None
    assert job.reproducibility["training_config"]["model_name"] == "job_repro_model"
    assert job.reproducibility["training_config"]["epochs"] == 5


def test_get_training_job_reproducibility_api(client: TestClient, db_session: Session):
    """GET /api/v1/model/training-job/{job_id}/reproducibility returns snapshot."""
    model = ModelBasic(
        model_name="api_repro_model",
        model_type=ProblemType.CLASSIFICATION,
        optimizer="adam",
        metric="accuracy",
        loss="sparse_categorical_crossentropy",
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job = create_training_job(
        model_id=model.id,
        project_id=None,
        hyperparams={"epochs": 10},
        session=db_session,
    )

    resp = client.get(f"{BASE}/training-job/{job.id}/reproducibility")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["job_id"] == job.id
    assert data["has_snapshot"] is True
    assert data["reproducibility"]["training_config"]["model_name"] == "api_repro_model"

    # Also verify that GET /api/v1/model/training-job/{job_id} includes reproducibility
    detail_resp = client.get(f"{BASE}/training-job/{job.id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()["data"]
    assert detail_data["reproducibility"] is not None


def test_get_training_job_reproducibility_legacy_fallback(client: TestClient, db_session: Session):
    """Legacy jobs without reproducibility snapshot return graceful fallback without errors."""
    model = ModelBasic(
        model_name="legacy_repro_model",
        model_type=ProblemType.CLASSIFICATION,
        optimizer="adam",
        metric="accuracy",
        loss="sparse_categorical_crossentropy",
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    job_id = str(uuid_pkg.uuid4())
    legacy_job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        reproducibility=None,
    )
    db_session.add(legacy_job)
    db_session.commit()

    resp = client.get(f"{BASE}/training-job/{job_id}/reproducibility")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["job_id"] == job_id
    assert data["has_snapshot"] is False
    assert data["reproducibility"] is None


def test_get_training_job_reproducibility_not_found(client: TestClient):
    """Non-existent job ID returns 404."""
    resp = client.get(f"{BASE}/training-job/non-existent-uuid/reproducibility")
    assert resp.status_code == 404
