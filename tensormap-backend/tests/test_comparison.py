"""Tests for GET /model/compare endpoint (Phase 6: Training run comparison)."""

import pytest

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.project import Project
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric


BASE = "/api/v1/model"


def _seed_model(db_session, model_name="test-model"):
    """Helper to create a model with project and datafile."""
    project = Project(name="test-project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    data_file = DataFile(
        file_name="test.csv",
        file_type="csv",
        disk_name="test.csv",
        project_id=project.id,
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.refresh(data_file)

    model = ModelBasic(
        model_name=model_name,
        file_id=data_file.id,
        project_id=project.id,
        model_type=1,
        target_field="label",
        training_split=80,
        optimizer="adam",
        metric="accuracy",
        epochs=10,
        batch_size=32,
        loss="categorical_crossentropy",
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


def _seed_job_with_metrics(db_session, job_id, model_id, hyperparams, metrics_data):
    """Helper to create a training job with metrics."""
    job = TrainingJob(
        id=job_id,
        model_id=model_id,
        status=TrainingStatus.COMPLETED,
        hyperparams=hyperparams,
    )
    db_session.add(job)
    db_session.commit()

    for metric_dict in metrics_data:
        epoch = metric_dict["epoch"]
        for metric_name, value in metric_dict.items():
            if metric_name == "epoch":
                continue
            metric = TrainingMetric(
                job_id=job_id,
                epoch=epoch,
                metric_name=metric_name,
                metric_value=value,
            )
            db_session.add(metric)
    db_session.commit()
    return job


class TestCompareJobs:
    """Tests for GET /model/compare endpoint."""

    def test_compare_two_jobs(self, db_session, client):
        """Compare two jobs returns correct aggregated metrics."""
        model = _seed_model(db_session)
        
        # Seed job 1
        _seed_job_with_metrics(
            db_session,
            "job-1",
            model.id,
            {"optimizer": "adam", "lr": 0.001, "batch_size": 32, "epochs": 3},
            [
                {"epoch": 1, "loss": 0.9, "val_loss": 0.95, "accuracy": 0.5, "val_accuracy": 0.48},
                {"epoch": 2, "loss": 0.7, "val_loss": 0.75, "accuracy": 0.6, "val_accuracy": 0.58},
                {"epoch": 3, "loss": 0.5, "val_loss": 0.55, "accuracy": 0.7, "val_accuracy": 0.68},
            ],
        )

        # Seed job 2
        _seed_job_with_metrics(
            db_session,
            "job-2",
            model.id,
            {"optimizer": "sgd", "lr": 0.01, "batch_size": 64, "epochs": 3},
            [
                {"epoch": 1, "loss": 0.8, "val_loss": 0.85, "accuracy": 0.55, "val_accuracy": 0.53},
                {"epoch": 2, "loss": 0.6, "val_loss": 0.65, "accuracy": 0.65, "val_accuracy": 0.63},
                {"epoch": 3, "loss": 0.4, "val_loss": 0.45, "accuracy": 0.75, "val_accuracy": 0.73},
            ],
        )

        response = client.get(f"{BASE}/compare?job_ids=job-1,job-2")

        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["jobs"]) == 2

        # Check job 1
        job1 = next(j for j in data["jobs"] if j["job_id"] == "job-1")
        assert job1["hyperparams"]["optimizer"] == "adam"
        assert len(job1["metrics"]) == 3
        assert job1["metrics"][0]["epoch"] == 1
        assert job1["metrics"][0]["loss"] == 0.9
        assert job1["metrics"][2]["val_accuracy"] == 0.68

        # Check job 2
        job2 = next(j for j in data["jobs"] if j["job_id"] == "job-2")
        assert job2["hyperparams"]["optimizer"] == "sgd"
        assert len(job2["metrics"]) == 3

    def test_compare_max_5_jobs(self, db_session, client):
        """Endpoint enforces max 5 jobs."""
        model = _seed_model(db_session)
        for i in range(6):
            _seed_job_with_metrics(
                db_session,
                f"job-{i}",
                model.id,
                {"optimizer": "adam", "lr": 0.001, "batch_size": 32, "epochs": 1},
                [{"epoch": 1, "loss": 0.5}],
            )

        response = client.get(f"{BASE}/compare?job_ids=job-0,job-1,job-2,job-3,job-4,job-5")

        assert response.status_code == 400
        assert "Maximum 5 jobs" in response.json()["message"]

    def test_compare_nonexistent_job_404(self, db_session, client):
        """Returns 404 if any job doesn't exist."""
        model = _seed_model(db_session)
        _seed_job_with_metrics(
            db_session,
            "job-1",
            model.id,
            {"optimizer": "adam", "lr": 0.001, "batch_size": 32, "epochs": 1},
            [{"epoch": 1, "loss": 0.5}],
        )

        response = client.get(f"{BASE}/compare?job_ids=job-1,nonexistent-job")

        assert response.status_code == 404
        assert "not found" in response.json()["message"].lower()

    def test_compare_empty_job_ids_400(self, client):
        """Returns 400 if no job IDs provided."""
        response = client.get(f"{BASE}/compare?job_ids=")

        assert response.status_code == 400
        assert "No job IDs provided" in response.json()["message"]

    def test_compare_single_job(self, db_session, client):
        """Can compare a single job (useful for consistency)."""
        model = _seed_model(db_session)
        _seed_job_with_metrics(
            db_session,
            "job-1",
            model.id,
            {"optimizer": "adam", "lr": 0.001, "batch_size": 32, "epochs": 2},
            [
                {"epoch": 1, "loss": 0.9, "val_loss": 0.95},
                {"epoch": 2, "loss": 0.7, "val_loss": 0.75},
            ],
        )

        response = client.get(f"{BASE}/compare?job_ids=job-1")

        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["job_id"] == "job-1"

    def test_compare_jobs_different_epoch_counts(self, db_session, client):
        """Can compare jobs with different numbers of epochs."""
        model = _seed_model(db_session)
        _seed_job_with_metrics(
            db_session,
            "job-1",
            model.id,
            {"optimizer": "adam", "lr": 0.001, "batch_size": 32, "epochs": 2},
            [
                {"epoch": 1, "loss": 0.9},
                {"epoch": 2, "loss": 0.7},
            ],
        )

        _seed_job_with_metrics(
            db_session,
            "job-2",
            model.id,
            {"optimizer": "sgd", "lr": 0.01, "batch_size": 64, "epochs": 4},
            [
                {"epoch": 1, "loss": 0.8},
                {"epoch": 2, "loss": 0.6},
                {"epoch": 3, "loss": 0.5},
                {"epoch": 4, "loss": 0.4},
            ],
        )

        response = client.get(f"{BASE}/compare?job_ids=job-1,job-2")

        assert response.status_code == 200
        data = response.json()["data"]
        job1 = next(j for j in data["jobs"] if j["job_id"] == "job-1")
        job2 = next(j for j in data["jobs"] if j["job_id"] == "job-2")
        assert len(job1["metrics"]) == 2
        assert len(job2["metrics"]) == 4

    def test_compare_includes_hyperparams(self, db_session, client):
        """Response includes hyperparameters for each job."""
        model = _seed_model(db_session)
        _seed_job_with_metrics(
            db_session,
            "job-1",
            model.id,
            {"optimizer": "adam", "lr": 0.001, "batch_size": 32, "epochs": 1},
            [{"epoch": 1, "loss": 0.5}],
        )

        _seed_job_with_metrics(
            db_session,
            "job-2",
            model.id,
            {"optimizer": "rmsprop", "lr": 0.0005, "batch_size": 16, "epochs": 1},
            [{"epoch": 1, "loss": 0.6}],
        )

        response = client.get(f"{BASE}/compare?job_ids=job-1,job-2")

        assert response.status_code == 200
        data = response.json()["data"]
        
        job1 = next(j for j in data["jobs"] if j["job_id"] == "job-1")
        assert job1["hyperparams"]["optimizer"] == "adam"
        assert job1["hyperparams"]["lr"] == 0.001
        assert job1["hyperparams"]["batch_size"] == 32

        job2 = next(j for j in data["jobs"] if j["job_id"] == "job-2")
        assert job2["hyperparams"]["optimizer"] == "rmsprop"
        assert job2["hyperparams"]["lr"] == 0.0005
        assert job2["hyperparams"]["batch_size"] == 16

    def test_compare_metrics_ordered_by_epoch(self, db_session, client):
        """Metrics are ordered by epoch number."""
        model = _seed_model(db_session)
        _seed_job_with_metrics(
            db_session,
            "job-1",
            model.id,
            {"optimizer": "adam", "lr": 0.001, "batch_size": 32, "epochs": 3},
            [
                {"epoch": 3, "loss": 0.5},
                {"epoch": 1, "loss": 0.9},
                {"epoch": 2, "loss": 0.7},
            ],
        )

        response = client.get(f"{BASE}/compare?job_ids=job-1")

        assert response.status_code == 200
        data = response.json()["data"]
        metrics = data["jobs"][0]["metrics"]
        
        assert metrics[0]["epoch"] == 1
        assert metrics[1]["epoch"] == 2
        assert metrics[2]["epoch"] == 3
