"""Integration tests verifying training and interpretability use identical test data.

These tests verify that the exact same test samples are used for both model training
and post-training interpretability analysis, ensuring consistency and accuracy.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.interpretability import InterpretabilityService
from app.services.model_run import _prepare_training_data
from app.shared.enums import ProblemType


@pytest.fixture
def mock_interpretability_session():
    """Create mock session for interpretability service."""
    session = MagicMock()
    job = MagicMock()
    job.model_id = 1
    model = MagicMock()
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


class TestTrainingInterpretabilityConsistency:
    """Verify training and interpretability produce identical test sets."""

    def test_classification_imbalanced_exact_match(self, mock_interpretability_session):
        """Verify training and interpretability produce EXACT same test data for imbalanced classification."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.CLASSIFICATION

        # Create imbalanced dataset (90% class 0, 10% class 1)
        test_data = pd.DataFrame(
            {
                "feat1": range(100),
                "feat2": range(100, 200),
                "label": [0] * 90 + [1] * 10,
            }
        )

        # Get training split
        x_train_tr, y_train_tr, x_test_tr, y_test_tr = _prepare_training_data(
            test_data.copy(), "label", 80, ProblemType.CLASSIFICATION
        )

        # Get interpretability split
        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test_interp, y_test_interp, class_names = service._load_test_data("job-123", session)

        # Verify exact match
        np.testing.assert_array_equal(x_test_tr.values, x_test_interp, "X_test arrays must match exactly")
        np.testing.assert_array_equal(y_test_tr.values, y_test_interp, "y_test arrays must match exactly")

        # Verify class distribution preserved (9:1 ratio)
        assert len(y_test_interp) == 20
        unique, counts = np.unique(y_test_interp, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == 18  # 90% of 20
        assert counts[1] == 2  # 10% of 20

    def test_classification_multiclass_exact_match(self, mock_interpretability_session):
        """Verify training and interpretability match for multiclass with different proportions."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.CLASSIFICATION

        # Create 3-class dataset with proportions 60:30:10
        test_data = pd.DataFrame(
            {
                "x1": range(100),
                "x2": range(200, 300),
                "label": [0] * 60 + [1] * 30 + [2] * 10,
            }
        )

        # Training split
        x_train_tr, y_train_tr, x_test_tr, y_test_tr = _prepare_training_data(
            test_data.copy(), "label", 80, ProblemType.CLASSIFICATION
        )

        # Interpretability split
        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test_interp, y_test_interp, _ = service._load_test_data("job-123", session)

        # Exact match verification
        np.testing.assert_array_equal(x_test_tr.values, x_test_interp)
        np.testing.assert_array_equal(y_test_tr.values, y_test_interp)

        # Verify proportions preserved (60:30:10 = 12:6:2 in 20-sample test set)
        counts = np.bincount(y_test_interp)
        assert counts[0] == 12  # 60% of 20
        assert counts[1] == 6  # 30% of 20
        assert counts[2] == 2  # 10% of 20

    def test_regression_exact_match(self, mock_interpretability_session):
        """Verify training and interpretability produce exact same test data for regression."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.REGRESSION
        model.target_field = "target"

        test_data = pd.DataFrame(
            {
                "feature1": range(50),
                "feature2": range(50, 100),
                "target": np.linspace(10, 60, 50),
            }
        )

        # Training split
        x_train_tr, y_train_tr, x_test_tr, y_test_tr = _prepare_training_data(
            test_data.copy(), "target", 80, ProblemType.REGRESSION
        )

        # Interpretability split
        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test_interp, y_test_interp, _ = service._load_test_data("job-123", session)

        # Exact match verification
        np.testing.assert_array_equal(x_test_tr.values, x_test_interp)
        np.testing.assert_array_equal(y_test_tr.values, y_test_interp)

        # Verify test set size
        assert len(x_test_interp) == 10  # 20% of 50

    def test_string_labels_consistent_encoding(self, mock_interpretability_session):
        """Verify string labels produce identical integer encoding."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.CLASSIFICATION

        test_data = pd.DataFrame(
            {
                "feature": range(60),
                "label": ["cat"] * 30 + ["dog"] * 20 + ["bird"] * 10,
            }
        )

        # Training split
        x_train_tr, y_train_tr, x_test_tr, y_test_tr = _prepare_training_data(
            test_data.copy(), "label", 80, ProblemType.CLASSIFICATION
        )

        # Interpretability split
        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test_interp, y_test_interp, class_names = service._load_test_data("job-123", session)

        # Exact match (same encoding)
        np.testing.assert_array_equal(y_test_tr.values, y_test_interp)

        # Verify class names preserved
        assert "cat" in class_names
        assert "dog" in class_names
        assert "bird" in class_names

    def test_with_missing_values_exact_match(self, mock_interpretability_session):
        """Verify missing values handled identically."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.CLASSIFICATION

        # Data with NaN values
        test_data = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,  # 20 NaN rows
                "feat2": [10.0] * 100,
                "label": [0, 1] * 50,
            }
        )

        # Training split (will drop NaN)
        x_train_tr, y_train_tr, x_test_tr, y_test_tr = _prepare_training_data(
            test_data.copy(), "label", 80, ProblemType.CLASSIFICATION
        )

        # Interpretability split (will drop NaN)
        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test_interp, y_test_interp, _ = service._load_test_data("job-123", session)

        # Both should have dropped same rows
        np.testing.assert_array_equal(x_test_tr.values, x_test_interp)
        np.testing.assert_array_equal(y_test_tr.values, y_test_interp)

        # No NaN in results
        assert not np.isnan(x_test_interp).any()


class TestAllInterpretabilityMethods:
    """Test all three interpretability data loading methods."""

    def test_load_test_data_with_features_consistency(self, mock_interpretability_session):
        """Verify _load_test_data_with_features produces same split."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.CLASSIFICATION

        test_data = pd.DataFrame(
            {
                "temp": range(100),
                "humidity": range(100, 200),
                "label": [0] * 70 + [1] * 30,
            }
        )

        # Get split from training
        x_train_tr, y_train_tr, x_test_tr, y_test_tr = _prepare_training_data(
            test_data.copy(), "label", 80, ProblemType.CLASSIFICATION
        )

        # Get split from interpretability with features
        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test_interp, y_test_interp, feature_names = service._load_test_data_with_features("job-123", session)

        # Verify exact match
        np.testing.assert_array_equal(x_test_tr.values, x_test_interp)
        np.testing.assert_array_equal(y_test_tr.values, y_test_interp)

        # Verify feature names
        assert feature_names == ["temp", "humidity"]

    def test_load_test_data_complete_consistency(self, mock_interpretability_session):
        """Verify _load_test_data_complete produces same split."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.CLASSIFICATION

        test_data = pd.DataFrame(
            {
                "x": range(80),
                "y": range(80, 160),
                "class": ["A"] * 40 + ["B"] * 40,
            }
        )
        model.target_field = "class"

        # Training split
        x_train_tr, y_train_tr, x_test_tr, y_test_tr = _prepare_training_data(
            test_data.copy(), "class", 80, ProblemType.CLASSIFICATION
        )

        # Interpretability split with complete metadata
        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test_interp, y_test_interp, class_names, feature_names = service._load_test_data_complete(
                "job-123", session
            )

        # Exact match
        np.testing.assert_array_equal(x_test_tr.values, x_test_interp)
        np.testing.assert_array_equal(y_test_tr.values, y_test_interp)

        # Verify metadata
        assert class_names == ["A", "B"]
        assert feature_names == ["x", "y"]

    def test_all_three_methods_produce_identical_data(self, mock_interpretability_session):
        """Verify all three loading methods return identical X_test and y_test."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.CLASSIFICATION

        test_data = pd.DataFrame(
            {
                "f1": range(50),
                "f2": range(50, 100),
                "target": [0] * 30 + [1] * 20,
            }
        )
        model.target_field = "target"

        service = InterpretabilityService()

        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"

            # Load via all three methods
            x1, y1, _ = service._load_test_data("job-123", session)
            x2, y2, _ = service._load_test_data_with_features("job-123", session)
            x3, y3, _, _ = service._load_test_data_complete("job-123", session)

        # All three must produce identical results
        np.testing.assert_array_equal(x1, x2, "Method 1 vs 2 mismatch")
        np.testing.assert_array_equal(x1, x3, "Method 1 vs 3 mismatch")
        np.testing.assert_array_equal(y1, y2, "Method 1 vs 2 labels mismatch")
        np.testing.assert_array_equal(y1, y3, "Method 1 vs 3 labels mismatch")


class TestActualStratificationBehavior:
    """Test actual stratification behavior without mocking train_test_split."""

    def test_imbalanced_multiclass_preserves_proportions(self, mock_interpretability_session):
        """Verify actual stratification preserves class proportions without mocking."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.CLASSIFICATION

        # 4 classes with proportions 50:30:15:5
        test_data = pd.DataFrame(
            {
                "feature": range(200),
                "label": [0] * 100 + [1] * 60 + [2] * 30 + [3] * 10,
            }
        )

        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test, y_test, _ = service._load_test_data("job-123", session)

        # Test set is 20% = 40 samples
        assert len(y_test) == 40

        # Count each class
        counts = np.bincount(y_test, minlength=4)

        # Proportions should be preserved: 50:30:15:5 = 20:12:6:2 in 40-sample test set
        assert counts[0] == 20  # 50% of 40
        assert counts[1] == 12  # 30% of 40
        assert counts[2] == 6  # 15% of 40
        assert counts[3] == 2  # 5% of 40

    def test_severe_imbalance_stratifies_correctly(self, mock_interpretability_session):
        """Test stratification with severe class imbalance (95:5)."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.CLASSIFICATION

        # Severely imbalanced: 95% class 0, 5% class 1
        test_data = pd.DataFrame(
            {
                "x": range(200),
                "label": [0] * 190 + [1] * 10,
            }
        )

        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test, y_test, _ = service._load_test_data("job-123", session)

        # Test set is 20% = 40 samples
        # With stratification: 95:5 ratio = 38:2
        counts = np.bincount(y_test)
        assert counts[0] == 38  # 95% of 40
        assert counts[1] == 2  # 5% of 40


class TestRegressionBackwardCompatibility:
    """Verify regression behavior matches previous shuffle-and-index approach."""

    def test_regression_exact_split_indices(self, mock_interpretability_session):
        """Verify regression produces exact same split as before (deterministic shuffle + index)."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.REGRESSION
        model.target_field = "y"

        # Create ordered data
        test_data = pd.DataFrame(
            {
                "x1": list(range(100)),
                "x2": list(range(100, 200)),
                "y": list(range(1000, 1100)),
            }
        )

        # Training behavior: shuffle then split
        x_train_tr, y_train_tr, x_test_tr, y_test_tr = _prepare_training_data(
            test_data.copy(), "y", 80, ProblemType.REGRESSION
        )

        # Interpretability behavior
        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"
            x_test_interp, y_test_interp, _ = service._load_test_data("job-123", session)

        # Must match exactly (same shuffle seed, same split index)
        np.testing.assert_array_equal(x_test_tr.values, x_test_interp)
        np.testing.assert_array_equal(y_test_tr.values, y_test_interp)

        # Should NOT be last 20 rows of original (because shuffled)
        # Original y values for last 20: 1080-1099
        assert not np.array_equal(y_test_interp, np.arange(1080, 1100))

    def test_regression_deterministic_across_calls(self, mock_interpretability_session):
        """Verify regression split is fully deterministic."""
        session, model = mock_interpretability_session
        model.model_type = ProblemType.REGRESSION
        model.target_field = "value"

        test_data = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
                "value": np.random.randn(100),
            }
        )

        service = InterpretabilityService()
        with (
            patch("app.config.get_settings") as mock_settings,
            patch("pandas.read_csv", return_value=test_data.copy()),
        ):
            mock_settings.return_value.upload_folder = "/uploads"

            # Multiple calls should produce identical results
            x1, y1, _ = service._load_test_data("job-123", session)
            x2, y2, _ = service._load_test_data("job-123", session)
            x3, y3, _ = service._load_test_data("job-123", session)

        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(x1, x3)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(y1, y3)
