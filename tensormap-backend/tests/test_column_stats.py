"""Tests for get_column_stats_service and GET /data/process/stats/{file_id}."""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from starlette.responses import JSONResponse

from app.services.data_process import _safe_float, get_column_stats_service

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
    f.disk_name = f"{file_name}.csv"
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


def test_rejects_non_csv_file():
    fake_file = MagicMock()
    fake_file.file_name = "images.zip"
    fake_file.file_type = "zip"
    db = MagicMock()
    db.exec.return_value.first.return_value = fake_file

    body, status = get_column_stats_service(db, uuid.uuid4())
    assert status == 400
    assert body["success"] is False
    assert "CSV" in body["message"]


def test_safe_float_helper():
    """Verify that _safe_float properly converts numbers and returns None for non-finite values."""
    assert _safe_float(None) is None
    assert _safe_float(float("nan")) is None
    assert _safe_float(float("inf")) is None
    assert _safe_float(float("-inf")) is None
    assert _safe_float("not_a_number") is None
    assert _safe_float(10) == 10.0
    assert _safe_float(3.14) == 3.14
    assert _safe_float("42.5") == 42.5


def test_column_stats_handles_infinite_values_json_compliant(tmp_path):
    """Verify that infinite values in dataset columns are converted to None and pass strict JSON serialization."""
    inf_csv = "a,b\n1,10\ninf,20\n3,30\n"
    body, status = _run(tmp_path, inf_csv)
    assert status == 200
    assert body["success"] is True

    rows = {r["column"]: r for r in body["data"]["columns"]}
    assert rows["a"]["mean"] is None
    assert rows["a"]["max"] is None
    assert rows["a"]["min"] == pytest.approx(1.0)
    assert rows["b"]["mean"] == pytest.approx(20.0)

    # Must be strictly JSON compliant under allow_nan=False and FastAPI/Starlette JSONResponse
    dumped = json.dumps(body, allow_nan=False)
    assert "null" in dumped
    rendered = JSONResponse(content=body).render(body)
    assert isinstance(rendered, bytes)
