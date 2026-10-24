"""Tests for target column protection and type guards in data preprocessing pipeline.

Validates that:
1. Active target columns (defined in DataProcess or linked ModelBasic) cannot be dropped
   or one-hot encoded, preventing broken training pipelines.
2. Numeric-only transforms (Min-Max Normalization, Z-score Standardization, Log Transform)
   cleanly reject non-numeric/string columns with 422 instead of crashing with 500 TypeError.
3. Imputation strategies 'mean' and 'median' reject non-numeric columns with 422.
4. Categorical to Numerical rejects columns containing NaNs with 422 to prevent silent
   data poisoning from pandas -1 categorical codes.
"""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.models import DataFile, DataProcess, ModelBasic
from app.schemas.data_process import TransformationItem
from app.services.data_process import (
    _categorical_to_numerical,
    _fill_missing,
    _log_transform,
    _min_max_normalize,
    _zscore_standardize,
    preprocess_data,
)


@pytest.fixture()
def test_file_id():
    return uuid.uuid4()


@pytest.fixture()
def mock_data_file(test_file_id):
    f = MagicMock(spec=DataFile)
    f.id = test_file_id
    f.file_name = "test_dataset"
    f.file_type = "csv"
    f.disk_name = "test_dataset.csv"
    f.columns = ["age", "salary", "city", "churn"]
    f.row_count = 5
    return f


@pytest.fixture()
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0, np.nan, 45.0],
            "salary": [50000.0, 60000.0, 75000.0, 80000.0, 90000.0],
            "city": ["New York", "London", "Paris", "London", np.nan],
            "churn": ["No", "Yes", "No", "No", "Yes"],
        }
    )
    file_path = tmp_path / "test_dataset.csv"
    df.to_csv(file_path, index=False)
    return tmp_path


def _setup_mock_db(mock_file, target=None, linked_models=None):
    db = MagicMock()

    def mock_exec(query):
        mock_result = MagicMock()
        query_str = str(query).lower()
        if "data_process" in query_str or "dataprocess" in query_str:
            if target:
                dp = MagicMock(spec=DataProcess)
                dp.file_id = mock_file.id
                dp.target = target
                mock_result.first.return_value = dp
            else:
                mock_result.first.return_value = None
        elif "model_basic" in query_str or "modelbasic" in query_str:
            mock_result.all.return_value = linked_models or []
        else:
            mock_result.first.return_value = mock_file
            mock_result.all.return_value = [mock_file]
        return mock_result

    db.exec.side_effect = mock_exec
    return db


# ===========================================================================
# 1. Target Column Protection Tests
# ===========================================================================


class TestTargetColumnProtection:
    @patch("app.services.data_process.get_settings")
    def test_cannot_drop_active_target_column(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file, target="churn")

        items = [TransformationItem(transformation="Drop Column", feature="churn")]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert "Cannot drop active target column 'churn'" in result["message"]

        # Confirm CSV on disk was not modified
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert "churn" in df_disk.columns

    @patch("app.services.data_process.get_settings")
    def test_cannot_one_hot_encode_active_target_column(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file, target="churn")

        items = [TransformationItem(transformation="One Hot Encoding", feature="churn")]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert "Cannot one-hot encode active target column 'churn'" in result["message"]

        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert "churn" in df_disk.columns

    @patch("app.services.data_process.get_settings")
    def test_cannot_drop_target_linked_via_model_basic(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        mb = MagicMock(spec=ModelBasic)
        mb.target_field = "churn"
        db = _setup_mock_db(mock_data_file, target=None, linked_models=[mb])

        items = [TransformationItem(transformation="Drop Column", feature="churn")]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 422
        assert "Cannot drop active target column 'churn'" in result["message"]

    @patch("app.services.data_process.get_settings")
    def test_cannot_one_hot_encode_target_linked_via_model_basic(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        mb = MagicMock(spec=ModelBasic)
        mb.target_field = "churn"
        db = _setup_mock_db(mock_data_file, target=None, linked_models=[mb])

        items = [TransformationItem(transformation="One Hot Encoding", feature="churn")]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 422
        assert "Cannot one-hot encode active target column 'churn'" in result["message"]

    @patch("app.services.data_process.get_settings")
    def test_dropping_non_target_feature_succeeds(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file, target="churn")

        items = [TransformationItem(transformation="Drop Column", feature="age")]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 200
        assert result["success"] is True
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert "age" not in df_disk.columns
        assert "churn" in df_disk.columns


# ===========================================================================
# 2. Numeric Transformation Type Guard Tests
# ===========================================================================


class TestNumericTransformTypeGuards:
    @pytest.mark.parametrize(
        "transform_name",
        ["Min-Max Normalization", "Z-score Standardization", "Log Transform"],
    )
    @patch("app.services.data_process.get_settings")
    def test_numeric_transforms_reject_string_column(
        self, mock_settings, test_file_id, mock_data_file, sample_csv, transform_name
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file)

        items = [TransformationItem(transformation=transform_name, feature="city")]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert f"Cannot apply {transform_name} to non-numeric column 'city'" in result["message"]

    @patch("app.services.data_process.get_settings")
    def test_numeric_transforms_succeed_on_numeric_column(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file)

        items = [TransformationItem(transformation="Min-Max Normalization", feature="salary")]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 200
        assert result["success"] is True
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert df_disk["salary"].min() >= 0.0
        assert df_disk["salary"].max() <= 1.0


# ===========================================================================
# 3. Fill Missing Strategy Guards
# ===========================================================================


class TestFillMissingStrategyGuards:
    @pytest.mark.parametrize("strategy", ["mean", "median"])
    @patch("app.services.data_process.get_settings")
    def test_mean_median_reject_non_numeric_column(
        self, mock_settings, test_file_id, mock_data_file, sample_csv, strategy
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file)

        items = [
            TransformationItem(
                transformation="Fill Missing Values",
                feature="city",
                params={"strategy": strategy},
            )
        ]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert f"Fill strategy '{strategy}' requires a numeric column" in result["message"]
        assert "Use 'mode' for non-numeric columns" in result["message"]

    @patch("app.services.data_process.get_settings")
    def test_mode_succeeds_on_non_numeric_column(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file)

        items = [
            TransformationItem(
                transformation="Fill Missing Values",
                feature="city",
                params={"strategy": "mode"},
            )
        ]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 200
        assert result["success"] is True
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert not df_disk["city"].isnull().any()

    @patch("app.services.data_process.get_settings")
    def test_mean_succeeds_on_numeric_column(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file)

        items = [
            TransformationItem(
                transformation="Fill Missing Values",
                feature="age",
                params={"strategy": "mean"},
            )
        ]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 200
        assert result["success"] is True
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert not df_disk["age"].isnull().any()


# ===========================================================================
# 4. Categorical to Numerical NaN Guard
# ===========================================================================


class TestCategoricalToNumericalNaNGuard:
    @patch("app.services.data_process.get_settings")
    def test_rejects_column_with_missing_values(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file)

        # "city" column in sample_csv has 1 missing value (np.nan)
        items = [TransformationItem(transformation="Categorical to Numerical", feature="city")]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert "contains 1 missing value(s)" in result["message"]
        assert "prevent invalid -1 code assignment" in result["message"]

    @patch("app.services.data_process.get_settings")
    def test_succeeds_when_column_has_no_missing_values(
        self, mock_settings, test_file_id, mock_data_file, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        db = _setup_mock_db(mock_data_file)

        # "churn" column has No/Yes and no missing values
        items = [TransformationItem(transformation="Categorical to Numerical", feature="churn")]
        result, status_code = preprocess_data(db, test_file_id, items)

        assert status_code == 200
        assert result["success"] is True
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert (df_disk["churn"] >= 0).all()
        assert df_disk["churn"].dtype.kind in ("i", "u")


# ===========================================================================
# 5. Direct Unit Tests for Transform Helper Functions
# ===========================================================================


class TestDirectTransformGuards:
    def test_min_max_normalize_non_numeric_raises_value_error(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="Cannot apply Min-Max Normalization to non-numeric"):
            _min_max_normalize(df, "text", None)

    def test_min_max_normalize_all_nan_raises_value_error(self):
        df = pd.DataFrame({"all_nan": [np.nan, np.nan]})
        with pytest.raises(ValueError, match="contains only missing values"):
            _min_max_normalize(df, "all_nan", None)

    def test_zscore_standardize_non_numeric_raises_value_error(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="Cannot apply Z-score Standardization to non-numeric"):
            _zscore_standardize(df, "text", None)

    def test_zscore_standardize_all_nan_raises_value_error(self):
        df = pd.DataFrame({"all_nan": [np.nan, np.nan]})
        with pytest.raises(ValueError, match="contains only missing values"):
            _zscore_standardize(df, "all_nan", None)

    def test_log_transform_non_numeric_raises_value_error(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="Cannot apply Log Transform to non-numeric"):
            _log_transform(df, "text", None)

    def test_fill_missing_non_numeric_mean_raises_value_error(self):
        df = pd.DataFrame({"text": ["a", None, "c"]})
        with pytest.raises(ValueError, match="Fill strategy 'mean' requires a numeric column"):
            _fill_missing(df, "text", {"strategy": "mean"})

    def test_fill_missing_non_numeric_median_raises_value_error(self):
        df = pd.DataFrame({"text": ["a", None, "c"]})
        with pytest.raises(ValueError, match="Fill strategy 'median' requires a numeric column"):
            _fill_missing(df, "text", {"strategy": "median"})

    def test_categorical_to_numerical_with_nan_raises_value_error(self):
        df = pd.DataFrame({"cat": ["x", np.nan, "y"]})
        with pytest.raises(ValueError, match="contains 1 missing value"):
            _categorical_to_numerical(df, "cat", None)
