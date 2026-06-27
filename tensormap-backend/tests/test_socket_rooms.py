"""Tests for Socket.IO per-job room isolation and catch-up.

We don't re-test socket.io's own room routing; we assert that our handlers pass
the right room/recipient and that catch-up returns the persisted history.
"""

from unittest.mock import AsyncMock, patch

import pytest

import app.socketio_instance as si
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric

NS = "/dl-result"


def _seed_job_with_metrics(factory, epochs: int) -> str:
    with factory() as session:
        job = TrainingJob(model_id=1, status=TrainingStatus.RUNNING)
        session.add(job)
        session.commit()
        session.refresh(job)
        for epoch in range(1, epochs + 1):
            session.add(TrainingMetric(job_id=job.id, epoch=epoch, metric_name="loss", metric_value=1.0 / epoch))
            session.add(TrainingMetric(job_id=job.id, epoch=epoch, metric_name="accuracy", metric_value=epoch / 10))
        session.commit()
        return job.id


async def test_subscribe_job_enters_room():
    with (
        patch.object(si, "sio", new=AsyncMock()) as mock_sio,
        patch.object(si, "_load_catchup", return_value=None),
    ):
        await si.subscribe_job("sid-1", {"job_id": "job-123"})

    mock_sio.enter_room.assert_awaited_once_with("sid-1", "job-123", namespace=NS)


async def test_subscribe_job_missing_id_is_noop():
    with patch.object(si, "sio", new=AsyncMock()) as mock_sio:
        await si.subscribe_job("sid-1", {})
    mock_sio.enter_room.assert_not_called()


async def test_subscribe_job_emits_catchup_to_sender_only():
    catchup = {"type": "catchup", "status": "running", "metrics": [{"epoch": 1, "loss": 0.5}]}
    with (
        patch.object(si, "sio", new=AsyncMock()) as mock_sio,
        patch.object(si, "_load_catchup", return_value=catchup),
    ):
        await si.subscribe_job("sid-1", {"job_id": "job-123"})

    mock_sio.emit.assert_awaited_once()
    args, kwargs = mock_sio.emit.call_args
    assert args[0] == "result :::"
    assert args[1] == catchup
    assert kwargs["to"] == "sid-1"  # directed to the subscriber, not broadcast
    assert kwargs["namespace"] == NS


async def test_unsubscribe_job_leaves_room():
    with patch.object(si, "sio", new=AsyncMock()) as mock_sio:
        await si.unsubscribe_job("sid-1", {"job_id": "job-123"})
    mock_sio.leave_room.assert_awaited_once_with("sid-1", "job-123", namespace=NS)


def test_catchup_returns_full_history(training_session_factory):
    job_id = _seed_job_with_metrics(training_session_factory, epochs=5)

    # _load_catchup uses training_service.make_session, repointed by the fixture.
    catchup = si._load_catchup(job_id)

    assert catchup is not None
    assert catchup["type"] == "catchup"
    assert catchup["status"] == "running"
    assert len(catchup["metrics"]) == 5
    assert [m["epoch"] for m in catchup["metrics"]] == [1, 2, 3, 4, 5]


def test_catchup_unknown_job_returns_none(training_session_factory):
    assert si._load_catchup("nonexistent") is None


@pytest.mark.parametrize("data", [None, {}, {"job_id": ""}])
async def test_unsubscribe_missing_id_is_noop(data):
    with patch.object(si, "sio", new=AsyncMock()) as mock_sio:
        await si.unsubscribe_job("sid-1", data)
    mock_sio.leave_room.assert_not_called()
