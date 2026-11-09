"""Tests for interpretability service data consistency with stratified splitting."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.interpretability import InterpretabilityService
from app.shared.enums import ProblemType


@pytest.fixture
def mock_session():
    """Create a mock database session with job, model, and data file."""
    session = MagicMock()
    job = MagicMock()
    job.model_id = 1
    model = MagicMock()
    model.model_id = 1
    model.file_id = "test-file-id"
    model.target_field = "label"
    model.training_split = 80
    data_file = MagicMock()
    data_file.disk_name = "test_data.csv"
    session.get.side_effect = lambda cls, id: {
        "job-123": job,
        1: model,
        "test-file-id": data_file,
    }.get(id)
    return session, model


def test_classification_uses_stratified_split(mock_session):
    """Verify classification models use stratified split."""
    session, model = mock_session
    model.model_type = ProblemType.CLASSIFICATION
    test_data = pd.DataFrame(
        {
            "feature": range(100),
            "label": [0] * 90 + [1] * 10,  # Imbalanced
        }
    )
    service = InterpretabilityService()
    with (
        patch("app.config.get_settings") as mock_settings,
        patch("pandas.read_csv", return_value=test_data),
        patch("app.services.interpretability.train_test_split") as mock_split,
    ):
        mock_settings.return_value.upload_folder = "/uploads"
        X_train = test_data.drop("label", axis=1).iloc[:80]
        X_test = test_data.drop("label", axis=1).iloc[80:]
        y_train = pd.Series([0] * 72 + [1] * 8)
        y_test = pd.Series([0] * 18 + [1] * 2)
        mock_split.return_value = (X_train, X_test, y_train, y_test)
        X_result, y_result, _ = service._load_test_data("job-123", session)
        mock_split.assert_called_once()
        assert "stratify" in mock_split.call_args.kwargs
        assert mock_split.call_args.kwargs["train_size"] == 0.8
        assert mock_split.call_args.kwargs["random_state"] == 42


def test_regression_uses_simple_split(mock_session):
    """Verify regression models use simple split."""
    session, model = mock_session
    model.model_type = ProblemType.REGRESSION
    model.target_field = "target"
    test_data = pd.DataFrame(
        {
            "feature": range(50),
            "target": range(100, 150),
        }
    )
    service = InterpretabilityService()
    with (
        patch("app.config.get_settings") as mock_settings,
        patch("pandas.read_csv", return_value=test_data),
        patch("app.services.interpretability.train_test_split") as mock_split,
    ):
        mock_settings.return_value.upload_folder = "/uploads"
        X_test, y_test, _ = service._load_test_data("job-123", session)
        mock_split.assert_not_called()
        assert len(X_test) == 10
        assert len(y_test) == 10


def test_classification_preserves_class_distribution(mock_session):
    """Verify class distribution is preserved."""
    session, model = mock_session
    model.model_type = ProblemType.CLASSIFICATION
    test_data = pd.DataFrame(
        {
            "feature": range(100),
            "label": [0] * 50 + [1] * 50,
        }
    )
    service = InterpretabilityService()
    with (
        patch("app.config.get_settings") as mock_settings,
        patch("pandas.read_csv", return_value=test_data),
    ):
        mock_settings.return_value.upload_folder = "/uploads"
        X_test, y_test, _ = service._load_test_data("job-123", session)
        unique, counts = np.unique(y_test, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1] == 10


def test_missing_values_dropped_before_split(mock_session):
    """Verify missing values are dropped."""
    session, model = mock_session
    model.model_type = ProblemType.CLASSIFICATION
    test_data = pd.DataFrame(
        {
            "feature": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,
            "label": [0, 1] * 50,
        }
    )
    service = InterpretabilityService()
    with (
        patch("app.config.get_settings") as mock_settings,
        patch("pandas.read_csv", return_value=test_data),
    ):
        mock_settings.return_value.upload_folder = "/uploads"
        X_test, y_test, _ = service._load_test_data("job-123", session)
        assert not np.isnan(X_test).any()
        assert not np.isnan(y_test).any()


def test_string_labels_encoded(mock_session):
    """Verify string labels are encoded."""
    session, model = mock_session
    model.model_type = ProblemType.CLASSIFICATION
    test_data = pd.DataFrame(
        {
            "feature": range(40),
            "label": ["setosa"] * 10 + ["versicolor"] * 15 + ["virginica"] * 15,
        }
    )
    service = InterpretabilityService()
    with (
        patch("app.config.get_settings") as mock_settings,
        patch("pandas.read_csv", return_value=test_data),
    ):
        mock_settings.return_value.upload_folder = "/uploads"
        X_test, y_test, class_names = service._load_test_data("job-123", session)
        assert y_test.dtype in [np.int8, np.int16, np.int32, np.int64, int]
        assert "setosa" in class_names
        assert len(class_names) == 3


def test_deterministic_classification_split(mock_session):
    """Verify classification split is deterministic."""
    session, model = mock_session
    model.model_type = ProblemType.CLASSIFICATION
    test_data = pd.DataFrame(
        {
            "feature": range(100),
            "label": [0] * 60 + [1] * 40,
        }
    )
    service = InterpretabilityService()
    with (
        patch("app.config.get_settings") as mock_settings,
        patch("pandas.read_csv", return_value=test_data),
    ):
        mock_settings.return_value.upload_folder = "/uploads"
        X_test1, y_test1, _ = service._load_test_data("job-123", session)
        X_test2, y_test2, _ = service._load_test_data("job-123", session)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_test1, y_test2)


def test_deterministic_regression_split(mock_session):
    """Verify regression split is deterministic."""
    session, model = mock_session
    model.model_type = ProblemType.REGRESSION
    model.target_field = "y"
    test_data = pd.DataFrame(
        {
            "x": range(50),
            "y": range(100, 150),
        }
    )
    service = InterpretabilityService()
    with (
        patch("app.config.get_settings") as mock_settings,
        patch("pandas.read_csv", return_value=test_data),
    ):
        mock_settings.return_value.upload_folder = "/uploads"
        X_test1, y_test1, _ = service._load_test_data("job-123", session)
        X_test2, y_test2, _ = service._load_test_data("job-123", session)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_test1, y_test2)
