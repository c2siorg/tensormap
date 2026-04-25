"""Unit tests for model export service."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from app.models.ml import ModelBasic

sys.modules.setdefault("tensorflow", MagicMock())
sys.modules.setdefault("flatten_json", MagicMock())


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def sample_model():
    m = MagicMock(spec=ModelBasic)
    m.id = 1
    m.model_name = "test_model"
    return m


class TestExportModelService:
    @pytest.mark.parametrize("fmt", ["savedmodel", "tflite", "onnx"])
    def test_supported_formats(self, mock_db, sample_model, fmt):
        from app.services.deep_learning import export_model_service

        mock_db.exec.return_value.first.return_value = sample_model
        mock_model = MagicMock(save=MagicMock())

        with (
            patch("app.services.deep_learning.os.path.exists", return_value=True),
            patch("tensorflow.keras.models.load_model", return_value=mock_model),
            patch("app.services.deep_learning._validate_model_path", return_value="/path"),
            patch("app.services.deep_learning.os.makedirs"),
            patch("builtins.open", MagicMock()),
        ):
            body, status = export_model_service(mock_db, "test_model", fmt)

        assert status == 200
        assert body["success"] is True

    def test_model_not_found(self, mock_db):
        from app.services.deep_learning import export_model_service

        mock_db.exec.return_value.first.return_value = None
        body, status = export_model_service(mock_db, "nonexistent", "savedmodel")

        assert status == 404
        assert body["success"] is False

    def test_file_not_on_disk(self, mock_db, sample_model):
        from app.services.deep_learning import export_model_service

        mock_db.exec.return_value.first.return_value = sample_model

        with patch("app.services.deep_learning.os.path.exists", return_value=False):
            body, status = export_model_service(mock_db, "test_model", "savedmodel")

        assert status == 404
        assert "not found" in body["message"]

    def test_unsupported_format(self, mock_db, sample_model):
        from app.services.deep_learning import export_model_service

        mock_db.exec.return_value.first.return_value = sample_model
        mock_model = MagicMock()

        with (
            patch("app.services.deep_learning.os.path.exists", return_value=True),
            patch("tensorflow.keras.models.load_model", return_value=mock_model),
        ):
            body, status = export_model_service(mock_db, "test_model", "invalid")

        assert status == 400
        assert "Unsupported" in body["message"]
