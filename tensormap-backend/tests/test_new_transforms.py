import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.schemas.data_process import TransformationItem
from app.services.data_process import preprocess_data


def _make_db() -> MagicMock:
    file = MagicMock()
    file.file_name = "test"
    file.file_type = "csv"
    db = MagicMock()
    db.exec.return_value.first.return_value = file
    return db


def _run(df: pd.DataFrame, transformation: str, feature: str, params: dict = None):
    db = _make_db()
    items = [TransformationItem(transformation=transformation, feature=feature, params=params)]
    with (
        patch("app.services.data_process.select"),
        patch("app.services.data_process.pd.read_csv", return_value=df.copy()),
        patch("app.services.data_process.get_settings") as mock_settings,
        patch("pandas.DataFrame.to_csv"),
    ):
        mock_settings.return_value.upload_folder = "/tmp"
        result, status_code = preprocess_data(db, uuid.uuid4(), items)
    return result, status_code


def _run_df(df: pd.DataFrame, transformation: str, feature: str, params: dict = None) -> pd.DataFrame:
    """Like _run but captures and returns the processed DataFrame."""
    db = _make_db()
    items = [TransformationItem(transformation=transformation, feature=feature, params=params)]
    saved = {}

    def capture(self_df, path, index=False):
        saved["df"] = self_df.copy()

    with (
        patch("app.services.data_process.select"),
        patch("app.services.data_process.pd.read_csv", return_value=df.copy()),
        patch("app.services.data_process.get_settings") as mock_settings,
        patch.object(pd.DataFrame, "to_csv", capture),
    ):
        mock_settings.return_value.upload_folder = "/tmp"
        preprocess_data(db, uuid.uuid4(), items)
    return saved["df"]


class TestMinMaxNormalization:
    def test_returns_200(self):
        df = pd.DataFrame({"score": [10, 20, 30, 40, 50]})
        _, status_code = _run(df, "Min-Max Normalization", "score")
        assert status_code == 200

    def test_scales_to_zero_one(self):
        df = pd.DataFrame({"score": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result_df = _run_df(df, "Min-Max Normalization", "score")
        assert result_df["score"].min() == pytest.approx(0.0)
        assert result_df["score"].max() == pytest.approx(1.0)

    def test_constant_column_returns_zeros(self):
        df = pd.DataFrame({"score": [5.0, 5.0, 5.0]})
        result_df = _run_df(df, "Min-Max Normalization", "score")
        assert (result_df["score"] == 0.0).all()


class TestZscoreStandardization:
    def test_returns_200(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        _, status_code = _run(df, "Z-score Standardization", "val")
        assert status_code == 200

    def test_mean_near_zero_std_near_one(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result_df = _run_df(df, "Z-score Standardization", "val")
        assert result_df["val"].mean() == pytest.approx(0.0, abs=1e-10)
        assert result_df["val"].std() == pytest.approx(1.0, abs=1e-10)

    def test_constant_column_returns_zeros(self):
        df = pd.DataFrame({"val": [3.0, 3.0, 3.0]})
        result_df = _run_df(df, "Z-score Standardization", "val")
        assert (result_df["val"] == 0.0).all()


class TestLogTransform:
    def test_returns_200(self):
        df = pd.DataFrame({"price": [1.0, 10.0, 100.0, 1000.0]})
        _, status_code = _run(df, "Log Transform", "price")
        assert status_code == 200

    def test_applies_log1p(self):
        values = [1.0, 10.0, 100.0, 1000.0]
        df = pd.DataFrame({"price": values})
        result_df = _run_df(df, "Log Transform", "price")
        expected = [np.log1p(v) for v in values]
        for got, exp in zip(result_df["price"], expected, strict=False):
            assert got == pytest.approx(exp)

    def test_zero_input_gives_zero(self):
        df = pd.DataFrame({"price": [0.0]})
        result_df = _run_df(df, "Log Transform", "price")
        assert result_df["price"].iloc[0] == pytest.approx(0.0)


class TestFillMissingValues:
    def test_returns_200(self):
        df = pd.DataFrame({"age": [25.0, None, 35.0, None, 45.0]})
        _, status_code = _run(df, "Fill Missing Values", "age")
        assert status_code == 200

    def test_default_strategy_is_mean(self):
        df = pd.DataFrame({"age": [10.0, None, 30.0]})
        result_df = _run_df(df, "Fill Missing Values", "age")
        assert result_df["age"].isna().sum() == 0
        assert result_df["age"].iloc[1] == pytest.approx(20.0)

    def test_median_strategy(self):
        df = pd.DataFrame({"age": [10.0, None, 30.0, 40.0, 50.0]})
        result_df = _run_df(df, "Fill Missing Values", "age", params={"strategy": "median"})
        assert result_df["age"].isna().sum() == 0
        assert result_df["age"].iloc[1] == pytest.approx(df["age"].median())

    def test_mode_strategy(self):
        df = pd.DataFrame({"val": [1.0, 1.0, None, 2.0]})
        result_df = _run_df(df, "Fill Missing Values", "val", params={"strategy": "mode"})
        assert result_df["val"].isna().sum() == 0
        assert result_df["val"].iloc[2] == pytest.approx(1.0)

    def test_params_none_defaults_to_mean(self):
        df = pd.DataFrame({"age": [10.0, None, 30.0]})
        result_df = _run_df(df, "Fill Missing Values", "age", params=None)
        assert result_df["age"].isna().sum() == 0
        assert result_df["age"].iloc[1] == pytest.approx(20.0)
