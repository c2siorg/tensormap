"""The Alembic chain must apply to a fresh SQLite database.

The other tests build the schema with SQLModel.metadata.create_all and skip Alembic entirely, so a
migration that only works on PostgreSQL (ALTER of a constraint, for example) went unnoticed until a
fresh `DATABASE_URL=sqlite:///...` setup refused to start. This runs the real chain.

Recent SQLite releases accept some ALTER COLUMN statements that older ones reject, so a migration
can pass on a developer machine and still fail in CI; run it against an older SQLite when in doubt.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

_SCRIPT = """
from alembic import command
from alembic.config import Config

cfg = Config("alembic.ini")
command.upgrade(cfg, "head")
command.downgrade(cfg, "h6i7j8k9l0m1")  # undo the tuning_session migration
command.upgrade(cfg, "head")
"""


def test_migrations_upgrade_downgrade_upgrade_on_fresh_sqlite(tmp_path):
    db_file = tmp_path / "fresh.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_file.as_posix()}", "SECRET_KEY": "test"}
    env.pop("TESTING", None)

    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT], cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=300
    )

    assert result.returncode == 0, result.stderr[-2000:]
    with sqlite3.connect(db_file) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        tuning_fk = [r for r in conn.execute("PRAGMA foreign_key_list(training_job)") if r[2] == "tuning_session"]
    assert {"model_basic", "training_job", "tuning_session"} <= tables
    assert tuning_fk, "training_job.tuning_session_id should reference tuning_session"
