"""Regression tests for issue #470 — preprocessing pipeline data integrity fixes.

Covers three defects that caused silent data corruption or unhandled HTTP 500s:

1. Target column deletion/encoding guard:
   ``Drop Column`` and ``One Hot Encoding`` on the active target must return 422,
   not silently destroy the column that every trained model references.

2. Non-numeric column type checks:
   ``Min-Max Normalization``, ``Z-score Standardization``, ``Log Transform``,
   and ``Fill Missing Values`` with ``mean``/``median`` strategy must return 422
   when applied to a string/categorical column.

3. ``Categorical to Numerical`` NaN guard:
   Columns with NaN values must be rejected with 422 — letting pd.Categorical.codes
   encode NaN as -1 silently corrupts model training data.
"""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.models.data import DataFile, DataProcess
from app.services.data_process import (
    _categorical_to_numerical,
    _fill_missing,
    _log_transform,
    _min_max_normalize,
    _zscore_standardize,
    preprocess_data,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def file_id():
    return uuid.uuid4()


@pytest.fixture()
def mock_db():
    return MagicMock()


@pytest.fixture()
def csv_file(file_id):
    """A DataFile-shaped mock for a CSV file."""
    f = MagicMock(spec=DataFile)
    f.id = file_id
    f.file_name = "iris"
    f.file_type = "csv"
    f.disk_name = "iris.csv"
    f.row_count = None
    f.columns = None
    return f


@pytest.fixture()
def target_record(file_id):
    """A DataProcess-shaped mock whose target is 'species'."""
    t = MagicMock(spec=DataProcess)
    t.file_id = file_id
    t.target = "species"
    return t


@pytest.fixture()
def classification_csv(tmp_path):
    """Small classification CSV with a string target column."""
    df = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 7.0, 6.3],
            "sepal_width": [3.5, 3.0, 3.2, 3.3],
            "species": ["setosa", "setosa", "versicolor", "virginica"],
        }
    )
    df.to_csv(tmp_path / "iris.csv", index=False)
    return tmp_path


@pytest.fixture()
def csv_with_nan(tmp_path):
    """CSV that has NaN in the 'category' column."""
    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0],
            "category": ["a", None, "b", "a"],
        }
    )
    df.to_csv(tmp_path / "nan.csv", index=False)
    return tmp_path


def _make_transform(transformation: str, feature: str, params=None):
    t = MagicMock()
    t.transformation = transformation
    t.feature = feature
    t.params = params
    return t


def _mock_db_preprocess(mock_db, file_obj, target_obj):
    """Set up mock_db so that the first db.exec().first() returns file_obj,
    and the second (DataProcess lookup) returns target_obj."""
    mock_db.exec.side_effect = [
        MagicMock(first=MagicMock(return_value=file_obj)),
        MagicMock(first=MagicMock(return_value=target_obj)),
    ]


# ---------------------------------------------------------------------------
# 1. Target column guard
# ---------------------------------------------------------------------------


class TestTargetColumnGuard:
    @patch("app.services.data_process.get_settings")
    def test_drop_column_on_active_target_returns_422(
        self, mock_settings, mock_db, file_id, csv_file, target_record, classification_csv
    ):
        """Dropping the active target column must be rejected with 422."""
        mock_settings.return_value.upload_folder = str(classification_csv)
        _mock_db_preprocess(mock_db, csv_file, target_record)

        t = _make_transform("Drop Column", "species")
        body, status = preprocess_data(mock_db, file_id, [t])

        assert status == 422
        assert body["success"] is False
        assert "target" in body["message"].lower()
        assert "species" in body["message"]

    @patch("app.services.data_process.get_settings")
    def test_one_hot_encoding_on_active_target_returns_422(
        self, mock_settings, mock_db, file_id, csv_file, target_record, classification_csv
    ):
        """One Hot Encoding the active target column must be rejected with 422."""
        mock_settings.return_value.upload_folder = str(classification_csv)
        _mock_db_preprocess(mock_db, csv_file, target_record)

        t = _make_transform("One Hot Encoding", "species")
        body, status = preprocess_data(mock_db, file_id, [t])

        assert status == 422
        assert body["success"] is False
        assert "target" in body["message"].lower()
        assert "species" in body["message"]

    @patch("app.services.data_process.get_settings")
    def test_drop_non_target_column_is_allowed(
        self, mock_settings, mock_db, file_id, csv_file, target_record, classification_csv
    ):
        """Dropping a non-target column is still allowed."""
        mock_settings.return_value.upload_folder = str(classification_csv)
        _mock_db_preprocess(mock_db, csv_file, target_record)

        t = _make_transform("Drop Column", "sepal_width")
        body, status = preprocess_data(mock_db, file_id, [t])

        assert status == 200

    @patch("app.services.data_process.get_settings")
    def test_drop_column_without_active_target_is_allowed(
        self, mock_settings, mock_db, file_id, csv_file, classification_csv
    ):
        """If no target is set, Drop Column is allowed on any column."""
        mock_settings.return_value.upload_folder = str(classification_csv)
        _mock_db_preprocess(mock_db, csv_file, None)  # no target record

        t = _make_transform("Drop Column", "species")
        body, status = preprocess_data(mock_db, file_id, [t])

        assert status == 200

    @patch("app.services.data_process.get_settings")
    def test_categorical_to_numerical_on_active_target_is_allowed(
        self, mock_settings, mock_db, file_id, csv_file, target_record, classification_csv
    ):
        """Categorical to Numerical does not remove the target column, so it is not blocked."""
        mock_settings.return_value.upload_folder = str(classification_csv)
        _mock_db_preprocess(mock_db, csv_file, target_record)

        t = _make_transform("Categorical to Numerical", "species")
        body, status = preprocess_data(mock_db, file_id, [t])

        # Should succeed — target still exists under same name after encoding.
        assert status == 200


# ---------------------------------------------------------------------------
# 2. Non-numeric column type checks (unit tests for private helpers)
# ---------------------------------------------------------------------------


class TestNumericColumnTypeChecks:
    """Test the private transform helpers directly for fast, isolated coverage."""

    def _str_col_df(self):
        return pd.DataFrame({"name": ["alice", "bob", "carol"]})

    def _numeric_df(self):
        return pd.DataFrame({"val": [1.0, 2.0, 3.0]})

    def test_min_max_normalize_raises_on_string_column(self):
        df = self._str_col_df()
        with pytest.raises(ValueError, match="numeric"):
            _min_max_normalize(df, "name", None)

    def test_min_max_normalize_works_on_numeric_column(self):
        df = self._numeric_df()
        result = _min_max_normalize(df, "val", None)
        assert result["val"].max() <= 1.0
        assert result["val"].min() >= 0.0

    def test_zscore_standardize_raises_on_string_column(self):
        df = self._str_col_df()
        with pytest.raises(ValueError, match="numeric"):
            _zscore_standardize(df, "name", None)

    def test_zscore_standardize_works_on_numeric_column(self):
        df = self._numeric_df()
        result = _zscore_standardize(df, "val", None)
        assert pytest.approx(result["val"].mean(), abs=1e-9) == 0.0

    def test_log_transform_raises_on_string_column(self):
        df = self._str_col_df()
        with pytest.raises(ValueError, match="numeric"):
            _log_transform(df, "name", None)

    def test_log_transform_works_on_numeric_column(self):
        df = self._numeric_df()
        result = _log_transform(df, "val", None)
        expected = np.log1p([1.0, 2.0, 3.0])
        assert list(result["val"]) == pytest.approx(list(expected))

    def test_fill_missing_mean_raises_on_string_column(self):
        df = pd.DataFrame({"name": ["alice", None, "carol"]})
        with pytest.raises(ValueError, match="numeric"):
            _fill_missing(df, "name", {"strategy": "mean"})

    def test_fill_missing_median_raises_on_string_column(self):
        df = pd.DataFrame({"name": ["alice", None, "carol"]})
        with pytest.raises(ValueError, match="numeric"):
            _fill_missing(df, "name", {"strategy": "median"})

    def test_fill_missing_mode_works_on_string_column(self):
        df = pd.DataFrame({"name": ["alice", None, "alice"]})
        result = _fill_missing(df, "name", {"strategy": "mode"})
        assert result["name"].isnull().sum() == 0
        assert result["name"].iloc[1] == "alice"


# ---------------------------------------------------------------------------
# 3. Categorical to Numerical NaN guard
# ---------------------------------------------------------------------------


class TestCategoricalToNumericalNanGuard:
    def test_raises_value_error_on_nan_column(self):
        df = pd.DataFrame({"cat": ["a", None, "b"]})
        with pytest.raises(ValueError, match="missing value"):
            _categorical_to_numerical(df, "cat", None)

    def test_error_message_includes_nan_count(self):
        df = pd.DataFrame({"cat": [None, None, "b"]})
        with pytest.raises(ValueError, match="2 missing value"):
            _categorical_to_numerical(df, "cat", None)

    def test_succeeds_without_nan_values(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
        result = _categorical_to_numerical(df, "cat", None)
        # Result must be integer-coded with no -1 sentinel values.
        assert pd.api.types.is_integer_dtype(result["cat"])
        assert (result["cat"] >= 0).all()

    @patch("app.services.data_process.get_settings")
    def test_preprocess_data_categorical_nan_returns_422(self, mock_settings, mock_db, file_id, csv_file, csv_with_nan):
        """The 422 guard flows through preprocess_data end-to-end."""
        mock_settings.return_value.upload_folder = str(csv_with_nan)
        _mock_db_preprocess(mock_db, csv_file, None)
        csv_file.disk_name = "nan.csv"

        t = _make_transform("Categorical to Numerical", "category")
        body, status = preprocess_data(mock_db, file_id, [t])

        assert status == 422
        assert body["success"] is False
        assert "missing value" in body["message"].lower()
