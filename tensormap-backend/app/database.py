"""Database engine and session dependency for FastAPI."""

from collections.abc import Generator

from sqlmodel import Session, create_engine

from app.config import get_settings


def _build_engine():
    """Create a SQLAlchemy engine with settings appropriate for the configured database."""
    settings = get_settings()
    url = settings.database_url
    kwargs: dict = {"pool_pre_ping": True}

    if not url.startswith("sqlite://"):
        # Apply explicit pool sizing only for non-SQLite engines;
        # SQLite typically uses different pooling semantics,
        # so we rely on SQLAlchemy's defaults for SQLite URLs.
        kwargs.update({"pool_size": 5, "max_overflow": 10, "pool_recycle": 1800})

    return create_engine(url, **kwargs)


engine = _build_engine()


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLModel session and close it when the request finishes."""
    with Session(engine) as session:
        yield session
