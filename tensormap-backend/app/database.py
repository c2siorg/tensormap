"""Database engine and session dependency for FastAPI."""

from collections.abc import Generator

from sqlmodel import Session, create_engine

from app.config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True)


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLModel session and close it when the request finishes."""
    with Session(engine) as session:
        yield session
