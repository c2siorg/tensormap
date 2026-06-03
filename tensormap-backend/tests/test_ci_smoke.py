"""
Smoke tests for CI/CD pipeline verification.

These tests ensure the basic infrastructure is working:
- Database connectivity is functional
"""

from sqlalchemy import text
from sqlmodel import Session, select


def test_db_connection(db_session: Session) -> None:
    """
    Test that the database connection is functional.

    This verifies that SQLModel can execute queries against the test database,
    whether it's PostgreSQL (CI) or SQLite (local).
    """
    result = db_session.exec(select(text("1"))).first()
    assert result is not None
