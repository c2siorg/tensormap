"""Health check endpoints for Kubernetes and production monitoring."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlmodel import Session

from app.database import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    """Basic liveness probe."""
    return {"status": "ok"}


@router.get("/ready")
def readiness_check(db: Session = Depends(get_db)):
    """Readiness probe - checks database connectivity."""
    try:
        db.exec(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception:
        return {"status": "not ready", "database": "disconnected"}
