"""Orphaned-job recovery: jobs left active by a crash are failed on startup."""

import pytest
from sqlmodel import Session, select

from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.training_service import orphan_recovery


@pytest.fixture
def model_id(db_session: Session) -> int:
    """A real model row so training_job's foreign key is satisfied.

    PostgreSQL (CI) enforces foreign keys, so jobs must reference a model that
    actually exists.
    """
    model = ModelBasic(model_name="orphan-test-model")
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model.id


def _seed_job(session: Session, model_id: int, status: TrainingStatus) -> TrainingJob:
    job = TrainingJob(model_id=model_id, status=status)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def test_recovers_running_and_pending_jobs(db_session, model_id):
    _seed_job(db_session, model_id, TrainingStatus.RUNNING)
    _seed_job(db_session, model_id, TrainingStatus.RUNNING)

    recovered = orphan_recovery(db_session)
    assert recovered == 2

    jobs = db_session.exec(select(TrainingJob)).all()
    assert all(j.status == TrainingStatus.FAILED for j in jobs)
    assert all(j.error_message == "Recovered orphaned job after restart" for j in jobs)
    assert all(j.completed_at is not None for j in jobs)


def test_does_not_touch_finished_jobs(db_session, model_id):
    completed = _seed_job(db_session, model_id, TrainingStatus.COMPLETED)
    cancelled = _seed_job(db_session, model_id, TrainingStatus.CANCELLED)
    failed = _seed_job(db_session, model_id, TrainingStatus.FAILED)

    recovered = orphan_recovery(db_session)
    assert recovered == 0

    db_session.refresh(completed)
    db_session.refresh(cancelled)
    db_session.refresh(failed)
    assert completed.status == TrainingStatus.COMPLETED
    assert cancelled.status == TrainingStatus.CANCELLED
    assert failed.status == TrainingStatus.FAILED


def test_recovery_is_idempotent(db_session, model_id):
    _seed_job(db_session, model_id, TrainingStatus.RUNNING)
    assert orphan_recovery(db_session) == 1
    # Second pass finds nothing left to recover.
    assert orphan_recovery(db_session) == 0
