"""Shared contract helpers for structured training Socket.IO events."""

from datetime import UTC, datetime

TRAINING_EVENT_VERSION = 1


def training_event(job_id: str, event_type: str, **fields) -> dict:
    """Build a versioned training event envelope with common metadata."""
    return {
        **fields,
        "version": TRAINING_EVENT_VERSION,
        "type": event_type,
        "job_id": job_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }
