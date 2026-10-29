"""Tests for get_feature_distribution_service and GET /data/process/distribution/{file_id}."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.data_process import get_feature_distribution_service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUMERIC_CSV = "a,b\n10,100\n20,200\n30,300\n40,400\n"
OUTLIER_CSV = "feature\n10\n11\n12\n10\n11\n10\n12\n100\n"
CATEGORICAL_CSV = "color,size\nred,small\nblue,medium\nred,large\nred,small\ngreen,medium\n"
MIXED_CSV = "age,name,score\n25,alice,88.5\n30,bob,92.0\n22,charlie,75.0\n"
CONSTANT_CSV = "const_col\n5.0\n5.0\n5.0\n5.0\n"
NULLS_CSV = "col_a,col_b\n1,\n,\n3,\n"


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
        body, status = get_feature_distribution_service(db, uuid.uuid4())

    return body, status


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_numeric_feature_distribution_stats(tmp_path):
    """Verify numeric feature distribution computes standard metrics accurately."""
    body, status = _run(tmp_path, NUMERIC_CSV)
    assert status == 200
    assert body["success"] is True
    assert body["data"]["total_rows"] == 4
    assert body["data"]["total_cols"] == 2

    cols = {r["column"]: r for r in body["data"]["columns"]}
    assert "a" in cols
    col_a = cols["a"]
    assert col_a["is_numeric"] is True
    assert col_a["count"] == 4
    assert col_a["null_count"] == 0
    assert col_a["mean"] == pytest.approx(25.0)
    assert col_a["min"] == pytest.approx(10.0)
    assert col_a["max"] == pytest.approx(40.0)
    assert col_a["median"] == pytest.approx(25.0)
    assert col_a["q25"] == pytest.approx(17.5)
    assert col_a["q75"] == pytest.approx(32.5)
    assert col_a["iqr"] == pytest.approx(15.0)
    assert col_a["outlier_count"] == 0
    assert col_a["outlier_percentage"] == 0.0
    assert len(col_a["histogram"]) > 0


def test_outlier_detection_iqr(tmp_path):
    """Verify Tukey's IQR rule identifies anomalous data points as outliers."""
    body, status = _run(tmp_path, OUTLIER_CSV)
    assert status == 200
    col = body["data"]["columns"][0]
    assert col["column"] == "feature"
    assert col["count"] == 8
    # 100 is an extreme value compared to [10, 11, 12, 10, 11, 10, 12]
    assert col["outlier_count"] == 1
    assert col["outlier_percentage"] == pytest.approx(12.5)
    assert col["upper_bound"] < 100.0


def test_histogram_generation_bins(tmp_path):
    """Verify histogram bin structure and total count conservation."""
    body, status = _run(tmp_path, NUMERIC_CSV)
    col_a = body["data"]["columns"][0]
    histogram = col_a["histogram"]
    assert len(histogram) > 0
    for bin_entry in histogram:
        assert "bin_start" in bin_entry
        assert "bin_end" in bin_entry
        assert "bin_label" in bin_entry
        assert "count" in bin_entry
        assert bin_entry["bin_start"] <= bin_entry["bin_end"]
    # Total counts across histogram bins should match non-null count
    assert sum(b["count"] for b in histogram) == col_a["count"]


def test_constant_numeric_feature(tmp_path):
    """Verify single-valued numeric column produces 1 bin and zero outliers."""
    body, status = _run(tmp_path, CONSTANT_CSV)
    col = body["data"]["columns"][0]
    assert col["std"] == pytest.approx(0.0)
    assert col["iqr"] == pytest.approx(0.0)
    assert col["outlier_count"] == 0
    assert len(col["histogram"]) == 1
    assert col["histogram"][0]["count"] == 4


def test_all_null_numeric_feature(tmp_path):
    """Verify numeric column with zero non-null values returns None metrics and empty histogram."""
    body, status = _run(tmp_path, NULLS_CSV)
    cols = {r["column"]: r for r in body["data"]["columns"]}
    col_b = cols["col_b"]
    assert col_b["count"] == 0
    assert col_b["null_count"] == 3
    assert col_b["mean"] is None
    assert col_b["median"] is None
    assert col_b["histogram"] == []
    assert col_b["outlier_count"] == 0


def test_categorical_feature_distribution(tmp_path):
    """Verify categorical columns return unique count and top frequency categories."""
    body, status = _run(tmp_path, CATEGORICAL_CSV)
    assert status == 200
    cols = {r["column"]: r for r in body["data"]["columns"]}
    color_col = cols["color"]
    assert color_col["is_numeric"] is False
    assert color_col["unique_count"] == 3
    assert "top_categories" in color_col
    top_cat = color_col["top_categories"][0]
    assert top_cat["category"] == "red"
    assert top_cat["count"] == 3
    assert top_cat["percentage"] == pytest.approx(60.0)


def test_mixed_dataset(tmp_path):
    """Verify dataset containing both numeric and string columns profiles both correctly."""
    body, status = _run(tmp_path, MIXED_CSV)
    assert status == 200
    assert body["data"]["total_cols"] == 3
    cols = {r["column"]: r for r in body["data"]["columns"]}
    assert cols["age"]["is_numeric"] is True
    assert cols["name"]["is_numeric"] is False
    assert cols["score"]["is_numeric"] is True


def test_unknown_file_returns_400():
    """Verify non-existent database record returns 400."""
    db = MagicMock()
    db.exec.return_value.first.return_value = None
    body, status = get_feature_distribution_service(db, uuid.uuid4())
    assert status == 400
    assert body["success"] is False


def test_rejects_non_csv_file():
    """Verify non-CSV files return a 400 error."""
    fake_file = MagicMock()
    fake_file.file_name = "archive.zip"
    fake_file.file_type = "zip"
    db = MagicMock()
    db.exec.return_value.first.return_value = fake_file

    body, status = get_feature_distribution_service(db, uuid.uuid4())
    assert status == 400
    assert body["success"] is False
    assert "CSV" in body["message"]


def test_missing_disk_file_returns_500(tmp_path):
    """Verify missing file on disk returns a 500 error."""
    fake_file = MagicMock()
    fake_file.file_name = "missing"
    fake_file.file_type = "csv"
    fake_file.disk_name = "non_existent.csv"
    db = MagicMock()
    db.exec.return_value.first.return_value = fake_file

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, status = get_feature_distribution_service(db, uuid.uuid4())

    assert status == 500
    assert body["success"] is False


def test_distribution_api_endpoint(tmp_path):
    """Verify GET /api/v1/data/process/distribution/{file_id} via TestClient."""
    from app.database import get_db

    fake_file = _fake_file(tmp_path, NUMERIC_CSV, "endpoint_test")
    file_id = uuid.uuid4()

    mock_db = MagicMock()
    mock_db.exec.return_value.first.return_value = fake_file

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.services.data_process.get_settings") as mock_cfg:
            mock_cfg.return_value.upload_folder = str(tmp_path)

            client = TestClient(app)
            resp = client.get(f"/api/v1/data/process/distribution/{file_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "columns" in data["data"]
    finally:
        app.dependency_overrides.pop(get_db, None)
