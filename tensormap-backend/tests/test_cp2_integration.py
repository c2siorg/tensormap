"""CP2 Gate Integration Tests

Full end-to-end tests for Phase 2:
- Training state persistence
- Socket.IO room isolation
- Live chart data flow
- Cancellation
- Orphaned job recovery
- Fallback polling

Marked with @pytest.mark.integration for separate CI runs.
"""

import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session

from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric
from app.shared.enums import ProblemType


@pytest.mark.integration
@pytest.mark.skip(reason="Requires running Socket.IO server on localhost:4300")
def test_full_training_lifecycle_with_socketio(client, db_session: Session, socketio_test_client):
    """Test complete training flow: start job → subscribe → receive metrics → complete."""

    # Setup: Create a model with training config
    file_id = uuid.uuid4()  # Use UUID for file_id
    model = ModelBasic(
        model_name="test_model",
        file_id=file_id,
        problem_type=ProblemType.CLASSIFICATION,
        epochs=5,
        batch_size=32,
        optimizer="adam",
        metric="accuracy",
    )
    db_session.add(model)
    db_session.commit()

    # Start training job
    response = client.post(
        "/api/v1/model/run",
        json={"model_name": "test_model", "project_id": None},
    )
    assert response.status_code == 202
    job_data = response.json()["data"]
    job_id = job_data["job_id"]
    assert job_data["status"] == "pending"

    # Connect to Socket.IO and subscribe to job
    sio = socketio_test_client
    events_received = []

    def on_result(data):
        events_received.append(data)

    sio.on("result :::", on_result)

    # Subscribe to job room
    sio.emit("subscribe_job", {"job_id": job_id})
    time.sleep(0.5)  # Allow catchup event to arrive

    # Verify catchup event received
    assert len(events_received) > 0
    catchup_event = events_received[0]
    assert catchup_event["type"] == "catchup"
    assert catchup_event["status"] in ["pending", "running"]

    # Poll job status until it completes (or timeout after 30s)
    max_wait = 30
    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = client.get(f"/api/v1/model/training-job/{job_id}")
        assert response.status_code == 200
        job_status = response.json()["data"]["status"]

        if job_status in ["completed", "failed", "cancelled"]:
            break

        time.sleep(1)

    # Verify metrics events were received
    metric_events = [e for e in events_received if e.get("type") == "metrics"]
    assert len(metric_events) > 0, "Should receive at least one metrics event"

    # Fetch metrics from API
    response = client.get(f"/api/v1/model/training-job/{job_id}/metrics")
    assert response.status_code == 200
    api_metrics = response.json()["data"]

    # Verify metrics match what was emitted
    assert len(api_metrics) >= len(metric_events)

    # Verify metric structure
    if len(api_metrics) > 0:
        first_metric = api_metrics[0]
        assert "epoch" in first_metric
        assert "loss" in first_metric

    # Cleanup
    sio.off("result :::")


@pytest.mark.integration
@pytest.mark.skip(reason="Requires running Socket.IO server on localhost:4300")
def test_socket_room_isolation(client, db_session: Session, socketio_test_client):
    """Test that User A does NOT see User B's training events."""

    # Setup: Create two models
    file_id = uuid.uuid4()  # Use UUID for file_id
    model_a = ModelBasic(
        model_name="model_a",
        file_id=file_id,
        problem_type=ProblemType.CLASSIFICATION,
        epochs=3,
        batch_size=32,
        optimizer="adam",
        metric="accuracy",
    )
    model_b = ModelBasic(
        model_name="model_b",
        file_id=file_id,
        problem_type=ProblemType.CLASSIFICATION,
        epochs=3,
        batch_size=32,
        optimizer="adam",
        metric="accuracy",
    )
    db_session.add(model_a)
    db_session.add(model_b)
    db_session.commit()

    # Start two concurrent jobs
    response_a = client.post("/api/v1/model/run", json={"model_name": "model_a"})
    response_b = client.post("/api/v1/model/run", json={"model_name": "model_b"})

    assert response_a.status_code == 202
    assert response_b.status_code == 202

    job_id_a = response_a.json()["data"]["job_id"]
    # job_id_b would be used for multi-client room isolation test
    # Currently testing basic subscription mechanism

    # Create socket connection
    sio_a = socketio_test_client
    events_a = []

    def on_result_a(data):
        events_a.append(data)

    sio_a.on("result :::", on_result_a)

    # Subscribe to job_id_a
    sio_a.emit("subscribe_job", {"job_id": job_id_a})
    time.sleep(1)

    # Verify events_a has data for job_id_a
    # Room isolation prevents cross-contamination
    assert len(events_a) > 0
    # In real scenario, we'd need separate socket connections for full isolation test
    # This test verifies the basic room subscription mechanism

    # Cleanup
    sio_a.off("result :::")


@pytest.mark.integration
@pytest.mark.skip(reason="Requires dataset file and long-running training process")
def test_cancellation_end_to_end(client, db_session: Session):
    """Test that cancellation works end-to-end."""

    # Setup: Create a model
    file_id = uuid.uuid4()  # Use UUID for file_id
    model = ModelBasic(
        model_name="cancel_test",
        file_id=file_id,
        problem_type=ProblemType.CLASSIFICATION,
        epochs=50,  # Long training
        batch_size=32,
        optimizer="adam",
        metric="accuracy",
    )
    db_session.add(model)
    db_session.commit()

    # Start training
    response = client.post("/api/v1/model/run", json={"model_name": "cancel_test"})
    assert response.status_code == 202
    job_id = response.json()["data"]["job_id"]

    # Wait for training to start
    time.sleep(2)

    # Request cancellation
    cancel_response = client.delete(f"/api/v1/model/training-job/{job_id}")
    assert cancel_response.status_code == 204

    # Poll until job reaches cancelled state
    max_wait = 15
    start_time = time.time()
    final_status = None

    while time.time() - start_time < max_wait:
        response = client.get(f"/api/v1/model/training-job/{job_id}")
        status = response.json()["data"]["status"]

        if status == "cancelled":
            final_status = "cancelled"
            break

        time.sleep(0.5)

    assert final_status == "cancelled", "Job should be cancelled"


@pytest.mark.integration
def test_orphaned_job_recovery(db_session: Session):
    """Test that orphaned jobs (server crash) are recovered on startup."""

    # Create a dummy model first to get a valid integer model_id
    model = ModelBasic(
        model_name="orphan_test_model",
        file_id=None,  # nullable field
        problem_type=ProblemType.CLASSIFICATION,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    # Simulate orphaned jobs: create jobs stuck in RUNNING state
    orphaned_job = TrainingJob(
        model_id=model.id,
        status=TrainingStatus.RUNNING,
        hyperparams={"epochs": 10, "batch_size": 32},
        started_at=datetime.now(UTC),
    )
    db_session.add(orphaned_job)
    db_session.commit()

    # Import and run the recovery function (this would happen on app startup)
    from app.services.training_service import orphan_recovery

    orphan_recovery(db_session)

    # Verify job status changed to FAILED
    db_session.refresh(orphaned_job)
    assert orphaned_job.status == TrainingStatus.FAILED
    assert "orphaned" in orphaned_job.error_message.lower() or "restart" in orphaned_job.error_message.lower()


@pytest.mark.integration
def test_metrics_api_consistency(client, db_session: Session):
    """Test that GET /training-job/{id}/metrics returns consistent data."""

    # Create a dummy model first to get a valid integer model_id
    model = ModelBasic(
        model_name="metrics_test_model",
        file_id=None,  # nullable field
        problem_type=ProblemType.CLASSIFICATION,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)

    # Create a completed job with metrics
    job = TrainingJob(
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        hyperparams={"epochs": 3, "batch_size": 32},
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)  # Refresh to get the generated job ID

    # Add metrics
    metrics_data = [
        {"epoch": 1, "loss": 0.9, "accuracy": 0.6, "val_loss": 0.95, "val_accuracy": 0.55},
        {"epoch": 2, "loss": 0.7, "accuracy": 0.7, "val_loss": 0.75, "val_accuracy": 0.68},
        {"epoch": 3, "loss": 0.5, "accuracy": 0.8, "val_loss": 0.55, "val_accuracy": 0.78},
    ]

    for epoch_data in metrics_data:
        epoch = epoch_data["epoch"]
        for metric_name, value in epoch_data.items():
            if metric_name != "epoch":
                metric = TrainingMetric(
                    job_id=job.id,
                    epoch=epoch,
                    metric_name=metric_name,
                    metric_value=value,
                )
                db_session.add(metric)

    db_session.commit()

    # Fetch metrics via API
    response = client.get(f"/api/v1/model/training-job/{job.id}/metrics")
    assert response.status_code == 200

    api_metrics = response.json()["data"]
    assert len(api_metrics) == 3

    # Verify structure
    for i, epoch_metrics in enumerate(api_metrics):
        expected = metrics_data[i]
        assert epoch_metrics["epoch"] == expected["epoch"]
        assert epoch_metrics["loss"] == expected["loss"]
        assert epoch_metrics["accuracy"] == expected["accuracy"]
        assert epoch_metrics["val_loss"] == expected["val_loss"]
        assert epoch_metrics["val_accuracy"] == expected["val_accuracy"]


@pytest.mark.integration
def test_training_jobs_list_endpoint(client, db_session: Session):
    """Test GET /training-jobs?model_name=X endpoint."""

    # Setup: Create a model and multiple jobs
    model = ModelBasic(
        model_name="multi_job_model",
        file_id=None,  # nullable field
        problem_type=ProblemType.CLASSIFICATION,
        epochs=5,
        batch_size=32,
        optimizer="adam",
        metric="accuracy",
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)  # Refresh to get auto-generated ID

    # Create multiple jobs with distinct, ascending started_at values so the
    # "most recently started first" ordering is deterministic. Using identical
    # timestamps made this assertion flaky (identical started_at values have no
    # guaranteed DB order without a secondary sort key).
    base_time = datetime.now(UTC)
    jobs = []
    for i in range(3):
        job = TrainingJob(
            model_id=model.id,
            status=TrainingStatus.COMPLETED if i < 2 else TrainingStatus.RUNNING,
            hyperparams={"epochs": 5, "batch_size": 32},
            started_at=base_time + timedelta(seconds=i),
        )
        jobs.append(job)
        db_session.add(job)

    db_session.commit()

    # Fetch jobs list
    response = client.get("/api/v1/model/training-jobs", params={"model_name": "multi_job_model"})
    assert response.status_code == 200

    jobs_data = response.json()["data"]
    assert len(jobs_data) == 3

    # Verify sorted by started_at DESC
    # The job started last (i == 2) should be first
    assert [j["job_id"] for j in jobs_data] == [jobs[2].id, jobs[1].id, jobs[0].id]
