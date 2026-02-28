"""Tests for get_column_stats_service and GET /data/process/stats/{file_id}."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.services.data_process import get_column_stats_service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUMERIC_CSV = "a,b\n1,4\n2,5\n3,6\n"
CATEGORICAL_CSV = "name,city\nalice,london\nbob,paris\nalice,london\n"
MIXED_CSV = "age,name\n25,alice\n30,bob\n22,charlie\n"
NULLS_CSV = "a,b\n1,\n2,5\n,6\n"


def _fake_file(tmp_path, csv_content, file_name="test"):
    """Write CSV to tmp_path and return a mock DataFile."""
    (tmp_path / f"{file_name}.csv").write_text(csv_content)
    f = MagicMock()
    f.file_name = file_name
    f.file_type = "csv"
    return f


def _run(tmp_path, csv_content, file_name="test"):
    """Call the service with a mocked DB session and patched settings."""
    fake_file = _fake_file(tmp_path, csv_content, file_name)
    db = MagicMock()
    db.exec.return_value.first.return_value = fake_file

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, status = get_column_stats_service(db, uuid.uuid4())

    return body, status


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_numeric_columns_returned(tmp_path):
    body, status = _run(tmp_path, NUMERIC_CSV)
    assert status == 200
    assert body["success"] is True
    column_names = [r["column"] for r in body["data"]["columns"]]
    assert "a" in column_names
    assert "b" in column_names


def test_summary_totals_present(tmp_path):
    body, _ = _run(tmp_path, NUMERIC_CSV)
    assert body["data"]["total_rows"] == 3
    assert body["data"]["total_cols"] == 2


def test_numeric_mean_correct(tmp_path):
    body, _ = _run(tmp_path, NUMERIC_CSV)
    rows = {r["column"]: r for r in body["data"]["columns"]}
    assert rows["a"]["mean"] == pytest.approx(2.0)
    assert rows["b"]["mean"] == pytest.approx(5.0)


def test_numeric_min_max_present(tmp_path):
    body, _ = _run(tmp_path, NUMERIC_CSV)
    rows = {r["column"]: r for r in body["data"]["columns"]}
    assert rows["a"]["min"] == pytest.approx(1.0)
    assert rows["b"]["max"] == pytest.approx(6.0)


def test_categorical_mean_min_max_are_none(tmp_path):
    """Non-numeric columns must return None for mean/min/max."""
    body, _ = _run(tmp_path, CATEGORICAL_CSV)
    rows = {r["column"]: r for r in body["data"]["columns"]}
    assert rows["name"]["mean"] is None
    assert rows["name"]["min"] is None
    assert rows["name"]["max"] is None


def test_null_count_correct(tmp_path):
    body, _ = _run(tmp_path, NULLS_CSV)
    rows = {r["column"]: r for r in body["data"]["columns"]}
    assert rows["a"]["null_count"] == 1
    assert rows["b"]["null_count"] == 1


def test_dtype_present(tmp_path):
    body, _ = _run(tmp_path, MIXED_CSV)
    for row in body["data"]["columns"]:
        assert "dtype" in row
        assert row["dtype"] != ""


def test_unknown_file_returns_400():
    db = MagicMock()
    db.exec.return_value.first.return_value = None
    body, status = get_column_stats_service(db, uuid.uuid4())
    assert status == 400
    assert body["success"] is False


def test_response_shape(tmp_path):
    body, _ = _run(tmp_path, NUMERIC_CSV)
    for row in body["data"]["columns"]:
        for key in ("column", "dtype", "count", "null_count", "mean", "min", "max"):
            assert key in row
