"""Tests for the training-job CRUD API and service helpers.

Background training is patched out in HTTP tests so we deterministically test
the request/persistence behaviour without spinning up a real Keras run.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, select

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.project import Project
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric
from app.services.training_service import orphan_recovery

BASE = "/api/v1/model"


def _seed_model(session: Session, name: str = "m1") -> ModelBasic:
    """Create a fully-configured model (with project + dataset) ready to train."""
    project = Project(name="proj")
    session.add(project)
    session.commit()
    session.refresh(project)

    data_file = DataFile(file_name="d.csv", file_type="csv", disk_name="d.csv", project_id=project.id)
    session.add(data_file)
    session.commit()
    session.refresh(data_file)

    model = ModelBasic(
        model_name=name,
        file_id=data_file.id,
        project_id=project.id,
        model_type=1,
        target_field="label",
        training_split=80,
        optimizer="adam",
        metric="accuracy",
        epochs=1,
        batch_size=32,
        loss="sparse_categorical_crossentropy",
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


def _seed_job(
    session: Session, model_id: int, status: TrainingStatus, started_at: datetime | None = None
) -> TrainingJob:
    job = TrainingJob(model_id=model_id, status=status, started_at=started_at)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


@pytest.fixture(autouse=True)
def _no_background_training(mocker):
    """Stop the run endpoint from launching real training in every HTTP test."""
    return mocker.patch("app.routers.training.launch_training_job")


def test_start_training_returns_202(client, db_session):
    _seed_model(db_session, "m1")
    resp = client.post(f"{BASE}/run", json={"model_name": "m1"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["data"]["job_id"]
    assert body["data"]["status"] == "pending"


def test_job_created_with_pending_status(client, db_session):
    _seed_model(db_session, "m1")
    resp = client.post(f"{BASE}/run", json={"model_name": "m1"})
    job_id = resp.json()["data"]["job_id"]

    job = db_session.get(TrainingJob, job_id)
    assert job is not None
    assert job.status == TrainingStatus.PENDING
    assert job.hyperparams["optimizer"] == "adam"
    assert job.hyperparams["epochs"] == 1


def test_start_training_launches_background(client, db_session, _no_background_training):
    _seed_model(db_session, "m1")
    resp = client.post(f"{BASE}/run", json={"model_name": "m1"})
    job_id = resp.json()["data"]["job_id"]
    _no_background_training.assert_called_once()
    # model_name and job_id are forwarded to the launcher.
    args = _no_background_training.call_args.args
    assert args[0] == "m1"
    assert args[1] == job_id


def test_start_training_unknown_model_404(client, db_session):
    resp = client.post(f"{BASE}/run", json={"model_name": "does-not-exist"})
    assert resp.status_code == 404


def test_start_training_unconfigured_model_400(client, db_session):
    # Model exists but has no file_id/epochs configured yet.
    model = ModelBasic(model_name="bare", optimizer="adam")
    db_session.add(model)
    db_session.commit()
    resp = client.post(f"{BASE}/run", json={"model_name": "bare"})
    assert resp.status_code == 400


def test_get_job_status(client, db_session):
    model = _seed_model(db_session, "m1")
    job = _seed_job(db_session, model.id, TrainingStatus.RUNNING)
    resp = client.get(f"{BASE}/training-job/{job.id}")
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "running"


def test_cancel_job(client, db_session):
    model = _seed_model(db_session, "m1")
    job = _seed_job(db_session, model.id, TrainingStatus.RUNNING)
    resp = client.delete(f"{BASE}/training-job/{job.id}")
    assert resp.status_code == 204
    db_session.refresh(job)
    assert job.status == TrainingStatus.CANCELLED


def test_get_job_404(client, db_session):
    # No auth/owner model exists, so unauthorized-access is expressed as 404.
    resp = client.get(f"{BASE}/training-job/missing-id")
    assert resp.status_code == 404
    # Errors follow the same {success, message, data} envelope as the rest of the API.
    assert resp.json()["success"] is False


def test_orphan_recovery_marks_running_as_failed(db_session):
    model = _seed_model(db_session, "m1")
    _seed_job(db_session, model.id, TrainingStatus.RUNNING)
    _seed_job(db_session, model.id, TrainingStatus.PENDING)

    recovered = orphan_recovery(db_session)
    assert recovered == 2

    jobs = db_session.exec(select(TrainingJob)).all()
    assert all(j.status == TrainingStatus.FAILED for j in jobs)
    assert all("orphan" in (j.error_message or "").lower() for j in jobs)


def test_metrics_endpoint_empty(client, db_session):
    model = _seed_model(db_session, "m1")
    job = _seed_job(db_session, model.id, TrainingStatus.RUNNING)
    resp = client.get(f"{BASE}/training-job/{job.id}/metrics")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_metrics_endpoint_after_insert(client, db_session):
    model = _seed_model(db_session, "m1")
    job = _seed_job(db_session, model.id, TrainingStatus.RUNNING)
    for epoch in (1, 2, 3):
        db_session.add(TrainingMetric(job_id=job.id, epoch=epoch, metric_name="loss", metric_value=1.0 / epoch))
        db_session.add(TrainingMetric(job_id=job.id, epoch=epoch, metric_name="accuracy", metric_value=epoch / 10))
    db_session.commit()

    resp = client.get(f"{BASE}/training-job/{job.id}/metrics")
    data = resp.json()["data"]
    assert len(data) == 3
    assert data[0] == {"epoch": 1, "loss": 1.0, "accuracy": 0.1}
    assert [row["epoch"] for row in data] == [1, 2, 3]


def test_concurrency_limit(client, db_session, monkeypatch):
    model = _seed_model(db_session, "m1")
    # Saturate with 3 active jobs (the default MAX_CONCURRENT_TRAINING_JOBS).
    for _ in range(3):
        _seed_job(db_session, model.id, TrainingStatus.RUNNING)
    resp = client.post(f"{BASE}/run", json={"model_name": "m1"})
    assert resp.status_code == 429
    assert resp.json()["success"] is False


def test_list_jobs_for_model(client, db_session):
    model = _seed_model(db_session, "m1")
    now = datetime.now(UTC)
    _seed_job(db_session, model.id, TrainingStatus.COMPLETED, started_at=now - timedelta(minutes=5))
    _seed_job(db_session, model.id, TrainingStatus.COMPLETED, started_at=now - timedelta(minutes=1))
    _seed_job(db_session, model.id, TrainingStatus.COMPLETED, started_at=now - timedelta(minutes=3))

    resp = client.get(f"{BASE}/training-jobs", params={"model_name": "m1"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 3
    starts = [row["started_at"] for row in data]
    assert starts == sorted(starts, reverse=True)  # most recent first
