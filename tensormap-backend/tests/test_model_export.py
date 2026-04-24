"""Unit tests for the deep_learning service — export_model_service.

Tests verify export_model_service returns correct responses for different scenarios.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from app.models.ml import ModelBasic

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("flatten_json", MagicMock())


@pytest.fixture()
def mock_db():
    return MagicMock()


@pytest.fixture()
def sample_model():
    m = MagicMock(spec=ModelBasic)
    m.id = 1
    m.model_name = "test_model"
    m.project_id = None
    return m


class TestExportModelService:
    """Tests for export_model_service."""

    @pytest.mark.parametrize("export_format,expected_format", [
        ("savedmodel", "savedmodel"),
        ("tflite", "tflite"),
        ("onnx", "onnx"),
    ])
    def test_supported_formats(self, mock_db, sample_model, export_format, expected_format):
        """All supported formats are accepted in the regex."""
        from app.services.deep_learning import export_model_service

        mock_db.exec.return_value.first.return_value = sample_model
        mock_model = MagicMock(save=MagicMock())

        with (
            patch("app.services.deep_learning.os.path.exists", return_value=True),
            patch("tensorflow.keras.models.load_model", return_value=mock_model),
            patch("app.services.deep_learning.os.path.join", return_value="/export/test_model"),
            patch("app.services.deep_learning.os.makedirs"),
            patch("builtins.open", MagicMock()),
        ):
            body, status = export_model_service(mock_db, "test_model", export_format)

        assert status == 200
        assert body["success"] is True
        assert body["data"]["format"] == expected_format

    def test_model_not_found_returns_404(self, mock_db):
        """Returns 404 when model doesn't exist."""
        from app.services.deep_learning import export_model_service

        mock_db.exec.return_value.first.return_value = None

        body, status = export_model_service(mock_db, "nonexistent", "savedmodel")

        assert status == 404
        assert body["success"] is False

    def test_unsupported_format_returns_400(self, mock_db, sample_model):
        """Returns 400 for unsupported format."""
        from app.services.deep_learning import export_model_service

        mock_db.exec.return_value.first.return_value = sample_model
        mock_model = MagicMock()

        with (
            patch("app.services.deep_learning.os.path.exists", return_value=True),
            patch("tensorflow.keras.models.load_model", return_value=mock_model),
        ):
            body, status = export_model_service(mock_db, "test_model", "invalid")

        assert status == 400
        assert "Unsupported format" in body["message"]

    def test_missing_file_returns_404(self, mock_db, sample_model):
        """Returns 404 when model file is missing from disk."""
        from app.services.deep_learning import export_model_service

        mock_db.exec.return_value.first.return_value = sample_model

        with patch("app.services.deep_learning.os.path.exists", return_value=False):
            body, status = export_model_service(mock_db, "test_model", "savedmodel")

        assert status == 404
        assert "not found on disk" in body["message"]