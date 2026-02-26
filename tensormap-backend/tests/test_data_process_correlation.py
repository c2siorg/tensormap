"""Unit tests for the get_correlation_matrix service function.

Strategy
--------
* The database and file-system are both mocked so the suite is fast and
  self-contained (no postgres, no real CSV files required).
* We patch ``app.services.data_process._get_file_path`` to skip disk I/O
  and ``app.services.data_process.pd.read_csv`` to return fabricated DataFrames.
* Patching ``_get_file_path`` (rather than ``get_settings``) avoids the
  ``@lru_cache`` complication and the need for a valid ``.env`` file.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.data_process import get_correlation_matrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(file_obj):
    """Return a mock Session whose .exec().first() returns *file_obj*."""
    db = MagicMock()
    db.exec.return_value.first.return_value = file_obj
    return db


def _fake_file(name="test", ftype="csv"):
    return SimpleNamespace(file_name=name, file_type=ftype)


FILE_ID = uuid.uuid4()

_FILE_PATH_PATCH = "app.services.data_process._get_file_path"
_FAKE_PATH = "/tmp/fake_dataset.csv"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetCorrelationMatrix:
    """Unit tests for get_correlation_matrix()."""

    def test_file_not_found_in_db(self):
        """Returns 400 when the file_id is not in the database."""
        db = _make_db(None)
        body, status = get_correlation_matrix(db, file_id=FILE_ID)
        assert status == 400
        assert body["success"] is False

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_happy_path_returns_200(self, mock_read_csv, _mock_path):
        """Returns 200 with columns and matrix for a valid numeric CSV."""
        mock_read_csv.return_value = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        db = _make_db(_fake_file())

        body, status = get_correlation_matrix(db, file_id=FILE_ID)

        assert status == 200
        assert body["success"] is True
        assert body["data"]["columns"] == ["a", "b"]
        assert len(body["data"]["matrix"]) == 2
        assert len(body["data"]["matrix"][0]) == 2

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_diagonal_is_one(self, mock_read_csv, _mock_path):
        """Diagonal values of the correlation matrix must equal 1.0."""
        mock_read_csv.return_value = pd.DataFrame({"x": [1, 2, 3], "y": [3, 1, 2]})
        db = _make_db(_fake_file())

        body, _ = get_correlation_matrix(db, file_id=FILE_ID)
        matrix = body["data"]["matrix"]

        for i in range(len(matrix)):
            assert matrix[i][i] == pytest.approx(1.0)

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_values_clamped_in_range(self, mock_read_csv, _mock_path):
        """All non-null matrix values must be in [-1, 1]."""
        mock_read_csv.return_value = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [8, 6, 4, 2]})
        db = _make_db(_fake_file())

        body, _ = get_correlation_matrix(db, file_id=FILE_ID)
        matrix = body["data"]["matrix"]

        for row in matrix:
            for val in row:
                if val is not None:
                    assert -1.0 <= val <= 1.0

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_no_numeric_columns_returns_empty(self, mock_read_csv, _mock_path):
        """A CSV with only string columns returns empty columns and matrix."""
        mock_read_csv.return_value = pd.DataFrame({"name": ["alice", "bob"], "city": ["NY", "LA"]})
        db = _make_db(_fake_file())

        body, status = get_correlation_matrix(db, file_id=FILE_ID)

        assert status == 200
        assert body["data"]["columns"] == []
        assert body["data"]["matrix"] == []

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_non_numeric_columns_excluded(self, mock_read_csv, _mock_path):
        """String columns must not appear in the output."""
        mock_read_csv.return_value = pd.DataFrame(
            {"label": ["a", "b", "c"], "score": [1.0, 2.0, 3.0], "rank": [3, 2, 1]}
        )
        db = _make_db(_fake_file())

        body, _ = get_correlation_matrix(db, file_id=FILE_ID)
        columns = body["data"]["columns"]

        assert "label" not in columns
        assert "score" in columns
        assert "rank" in columns

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_nan_serialised_as_none(self, mock_read_csv, _mock_path):
        """NaN correlation values (e.g. constant column) are serialised as None."""
        mock_read_csv.return_value = pd.DataFrame({"constant": [5, 5, 5], "varying": [1, 2, 3]})
        db = _make_db(_fake_file())

        body, _ = get_correlation_matrix(db, file_id=FILE_ID)
        matrix = body["data"]["matrix"]
        columns = body["data"]["columns"]

        constant_idx = columns.index("constant")
        varying_idx = columns.index("varying")

        assert matrix[constant_idx][varying_idx] is None
        assert matrix[varying_idx][constant_idx] is None

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv", side_effect=FileNotFoundError("missing"))
    def test_file_not_found_on_disk(self, _mock_csv, _mock_path):
        """Returns 500 when the CSV file does not exist on disk."""
        db = _make_db(_fake_file())

        body, status = get_correlation_matrix(db, file_id=FILE_ID)

        assert status == 500
        assert body["success"] is False

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_values_rounded_to_six_decimal_places(self, mock_read_csv, _mock_path):
        """Numeric values are rounded to at most 6 decimal places."""
        mock_read_csv.return_value = pd.DataFrame({"p": [1, 2, 3, 4, 5], "q": [2, 3, 5, 4, 6]})
        db = _make_db(_fake_file())

        body, _ = get_correlation_matrix(db, file_id=FILE_ID)
        matrix = body["data"]["matrix"]

        for row in matrix:
            for val in row:
                if val is not None:
                    assert val == round(val, 6)

    @patch(_FILE_PATH_PATCH, return_value=_FAKE_PATH)
    @patch("app.services.data_process.pd.read_csv")
    def test_matrix_is_symmetric(self, mock_read_csv, _mock_path):
        """Correlation matrix must be symmetric: matrix[i][j] == matrix[j][i]."""
        mock_read_csv.return_value = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 5, 3]})
        db = _make_db(_fake_file())

        body, _ = get_correlation_matrix(db, file_id=FILE_ID)
        matrix = body["data"]["matrix"]
        n = len(matrix)

        for i in range(n):
            for j in range(n):
                vi, vj = matrix[i][j], matrix[j][i]
                if vi is None and vj is None:
                    continue
                assert vi == pytest.approx(vj)
