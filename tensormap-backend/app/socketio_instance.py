"""Shared Socket.IO server instance used for real-time training progress.

Progress is isolated per training job via Socket.IO rooms: a client subscribes
to ``job_id`` and only receives that job's metrics (no global broadcast). Late
or reconnecting subscribers get a one-shot catch-up of the persisted history.
"""

import asyncio

import socketio

from app.config import get_settings
from app.shared.constants import SOCKETIO_DL_NAMESPACE, SOCKETIO_LISTENER
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=get_settings().cors_allowed_origins_list,
)


@sio.on("connect", namespace=SOCKETIO_DL_NAMESPACE)
async def dl_connect(sid, environ):
    """Accept and log client connections to the training progress namespace."""
    client_ip = environ.get("REMOTE_ADDR", "unknown")
    logger.info("Client connected to training namespace: sid=%s, ip=%s", sid, client_ip)


def _load_catchup(job_id: str) -> dict | None:
    """Read a job's current status and full metric history (sync DB access).

    Returns None if the job does not exist. Runs in a worker thread because the
    DB layer is synchronous.
    """
    from app.models.training_job import TrainingJob
    from app.services.training_service import get_job_metrics_grouped, make_session

    with make_session() as session:
        job = session.get(TrainingJob, job_id)
        if job is None:
            return None
        return {
            "type": "catchup",
            "status": job.status.value,
            "metrics": get_job_metrics_grouped(job_id, session),
        }


@sio.on("subscribe_job", namespace=SOCKETIO_DL_NAMESPACE)
async def subscribe_job(sid, data):
    """Join the room for a job and replay its persisted metrics to this client."""
    job_id = (data or {}).get("job_id")
    if not job_id:
        return

    await sio.enter_room(sid, job_id, namespace=SOCKETIO_DL_NAMESPACE)

    # Catch-up: send everything persisted so far to this late/reconnecting client.
    catchup = await asyncio.to_thread(_load_catchup, job_id)
    if catchup is not None:
        await sio.emit(SOCKETIO_LISTENER, catchup, to=sid, namespace=SOCKETIO_DL_NAMESPACE)


@sio.on("unsubscribe_job", namespace=SOCKETIO_DL_NAMESPACE)
async def unsubscribe_job(sid, data):
    """Leave the room for a job so this client stops receiving its metrics."""
    job_id = (data or {}).get("job_id")
    if job_id:
        await sio.leave_room(sid, job_id, namespace=SOCKETIO_DL_NAMESPACE)
