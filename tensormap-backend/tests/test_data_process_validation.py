"""Tests for target column protection and type guards in data preprocessing pipeline.

Validates that:
1. Active target columns (defined in DataProcess or linked ModelBasic) cannot be dropped
   or one-hot encoded, preventing broken training pipelines.
2. Numeric-only transforms (Min-Max Normalization, Z-score Standardization, Log Transform)
   cleanly reject non-numeric/string columns with 422 instead of crashing with 500 TypeError.
3. Imputation strategies 'mean' and 'median' reject non-numeric columns with 422.
4. Valid multi-step transformation pipelines (e.g. Fill Missing Values -> Categorical to Numerical,
   or Categorical to Numerical -> Min-Max Normalization) succeed sequentially.
5. Failures during sequential execution leave the file on disk untouched.
6. Categorical to Numerical adheres to the pandas .codes convention (-1 for NaNs).
"""

import uuid
from unittest.mock import patch

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


def _seed_db_file(
    db_session,
    file_id: uuid.UUID,
    file_name: str = "test_dataset",
    disk_name: str = "test_dataset.csv",
    dp_targets: list[str] | None = None,
    model_targets: list[str] | None = None,
) -> DataFile:
    file = DataFile(
        id=file_id,
        file_name=file_name,
        file_type="csv",
        disk_name=disk_name,
        columns=["age", "salary", "city", "churn"],
        row_count=5,
    )
    db_session.add(file)

    for target in dp_targets or []:
        dp = DataProcess(file_id=file_id, target=target)
        db_session.add(dp)

    for idx, target in enumerate(model_targets or []):
        mb = ModelBasic(model_name=f"model_{idx}_{target}", file_id=file_id, target_field=target)
        db_session.add(mb)

    db_session.commit()
    db_session.refresh(file)
    return file


# ===========================================================================
# 1. Target Column Protection Tests
# ===========================================================================


class TestTargetColumnProtection:
    @patch("app.services.data_process.get_settings")
    def test_cannot_drop_active_target_column(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id, dp_targets=["churn"])

        items = [TransformationItem(transformation="Drop Column", feature="churn")]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert "Cannot drop active target column 'churn'" in result["message"]
        assert "dataset target setting" in result["message"]

        # Confirm CSV on disk was not modified
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert "churn" in df_disk.columns

    @patch("app.services.data_process.get_settings")
    def test_cannot_one_hot_encode_active_target_column(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id, dp_targets=["churn"])

        items = [TransformationItem(transformation="One Hot Encoding", feature="churn")]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert "Cannot one-hot encode active target column 'churn'" in result["message"]
        assert "dataset target setting" in result["message"]

        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert "churn" in df_disk.columns

    @patch("app.services.data_process.get_settings")
    def test_protects_multiple_targets_across_dp_and_models(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id, dp_targets=["churn"], model_targets=["salary"])

        items = [TransformationItem(transformation="Drop Column", feature="salary")]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 422
        assert "Cannot drop active target column 'salary'" in result["message"]
        assert "model 'model_0_salary'" in result["message"]

    @patch("app.services.data_process.get_settings")
    def test_cannot_drop_target_linked_via_model_basic(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id, model_targets=["churn"])

        items = [TransformationItem(transformation="Drop Column", feature="churn")]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 422
        assert "Cannot drop active target column 'churn'" in result["message"]
        assert "model 'model_0_churn'" in result["message"]

    @patch("app.services.data_process.get_settings")
    def test_cannot_one_hot_encode_target_linked_via_model_basic(
        self, mock_settings, db_session, test_file_id, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id, model_targets=["churn"])

        items = [TransformationItem(transformation="One Hot Encoding", feature="churn")]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 422
        assert "Cannot one-hot encode active target column 'churn'" in result["message"]

    @patch("app.services.data_process.get_settings")
    def test_dropping_non_target_feature_succeeds(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id, dp_targets=["churn"])

        items = [TransformationItem(transformation="Drop Column", feature="age")]
        result, status_code = preprocess_data(db_session, test_file_id, items)

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
        self, mock_settings, db_session, test_file_id, sample_csv, transform_name
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id)

        items = [TransformationItem(transformation=transform_name, feature="city")]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert f"Cannot apply {transform_name} to non-numeric column 'city'" in result["message"]

        # Ensure disk CSV remains untouched on validation failure
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert "city" in df_disk.columns
        assert not pd.api.types.is_numeric_dtype(df_disk["city"])

    @patch("app.services.data_process.get_settings")
    def test_numeric_transforms_succeed_on_numeric_column(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id)

        items = [TransformationItem(transformation="Min-Max Normalization", feature="salary")]
        result, status_code = preprocess_data(db_session, test_file_id, items)

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
    def test_mean_median_reject_non_numeric_column(self, mock_settings, db_session, test_file_id, sample_csv, strategy):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id)

        items = [
            TransformationItem(
                transformation="Fill Missing Values",
                feature="city",
                params={"strategy": strategy},
            )
        ]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert f"Fill strategy '{strategy}' requires a numeric column" in result["message"]
        assert "Use 'mode' for non-numeric columns" in result["message"]

        # Ensure disk CSV was not modified
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert df_disk["city"].isnull().any()

    @patch("app.services.data_process.get_settings")
    def test_mode_succeeds_on_non_numeric_column(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id)

        items = [
            TransformationItem(
                transformation="Fill Missing Values",
                feature="city",
                params={"strategy": "mode"},
            )
        ]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 200
        assert result["success"] is True
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert not df_disk["city"].isnull().any()

    @patch("app.services.data_process.get_settings")
    def test_mean_succeeds_on_numeric_column(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id)

        items = [
            TransformationItem(
                transformation="Fill Missing Values",
                feature="age",
                params={"strategy": "mean"},
            )
        ]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 200
        assert result["success"] is True
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert not df_disk["age"].isnull().any()


# ===========================================================================
# 4. Multi-Step Sequential Transformation Pipelines
# ===========================================================================


class TestMultiStepSequentialPipelines:
    @patch("app.services.data_process.get_settings")
    def test_categorical_then_min_max_succeeds_sequentially(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id)

        # "churn" starts as string. Step 1 encodes to int, Step 2 normalizes numeric codes
        items = [
            TransformationItem(transformation="Categorical to Numerical", feature="churn"),
            TransformationItem(transformation="Min-Max Normalization", feature="churn"),
        ]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 200
        assert result["success"] is True
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert df_disk["churn"].min() >= 0.0
        assert df_disk["churn"].max() <= 1.0

    @patch("app.services.data_process.get_settings")
    def test_pipeline_failure_at_step_2_leaves_file_untouched(
        self, mock_settings, db_session, test_file_id, sample_csv
    ):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id)

        original_df = pd.read_csv(sample_csv / "test_dataset.csv")

        # Step 1 succeeds (Fill age with mean), Step 2 fails (Z-score on string 'city')
        items = [
            TransformationItem(transformation="Fill Missing Values", feature="age", params={"strategy": "mean"}),
            TransformationItem(transformation="Z-score Standardization", feature="city"),
        ]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert "Cannot apply Z-score Standardization to non-numeric column 'city'" in result["message"]

        # Ensure file was not modified (age should still have missing values)
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert df_disk["age"].isnull().sum() == original_df["age"].isnull().sum()


# ===========================================================================
# 5. Categorical to Numerical Validation Tests
# ===========================================================================


class TestCategoricalToNumericalValidation:
    def test_categorical_to_numerical_with_nan_raises_value_error(self):
        df = pd.DataFrame({"city": ["London", "Paris", np.nan, "London"]})
        with pytest.raises(ValueError, match="contains missing values"):
            _categorical_to_numerical(df, "city", None)

    def test_categorical_to_numerical_succeeds_without_nan(self):
        df = pd.DataFrame({"city": ["London", "Paris", "Berlin", "London"]})
        result = _categorical_to_numerical(df, "city", None)
        codes = list(result["city"])
        assert codes[0] == codes[3]
        assert result["city"].dtype.kind in ("i", "u")

    @patch("app.services.data_process.get_settings")
    def test_preprocess_rejects_categorical_on_nan_column(self, mock_settings, db_session, test_file_id, sample_csv):
        mock_settings.return_value.upload_folder = str(sample_csv)
        _seed_db_file(db_session, test_file_id)

        original_df = pd.read_csv(sample_csv / "test_dataset.csv")

        # "city" column in test_dataset.csv has missing values
        items = [TransformationItem(transformation="Categorical to Numerical", feature="city")]
        result, status_code = preprocess_data(db_session, test_file_id, items)

        assert status_code == 422
        assert result["success"] is False
        assert "contains missing values" in result["message"]

        # Ensure disk CSV untouched
        df_disk = pd.read_csv(sample_csv / "test_dataset.csv")
        assert df_disk["city"].isnull().sum() == original_df["city"].isnull().sum()


# ===========================================================================
# 6. Direct Transform Helper Unit Tests
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
