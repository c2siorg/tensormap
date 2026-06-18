"""Unit tests for null check handling in model_run and code_generation services.

These tests verify that missing database records produce clear error messages
instead of AttributeError crashes.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy third-party modules before importing the services
_tf_stub = MagicMock()
sys.modules.setdefault("tensorflow", _tf_stub)
sys.modules.setdefault("flatten_json", MagicMock())

from app.services.code_generation import (  # noqa: E402
    CodeGenerationError,
    _file_location,
    generate_code,
)
from app.services.model_run import (  # noqa: E402
    ModelRunError,
    _helper_generate_file_location,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    # Default: queries return None (record not found)
    db.exec.return_value.first.return_value = None
    return db


# ---------------------------------------------------------------------------
# model_run.py — _helper_generate_file_location
# ---------------------------------------------------------------------------


class TestHelperGenerateFileLocation:
    def test_raises_error_when_file_not_found(self, mock_db):
        """Missing DataFile raises ModelRunError with descriptive message."""
        mock_db.exec.return_value.first.return_value = None

        with pytest.raises(ModelRunError) as exc_info:
            _helper_generate_file_location(mock_db, file_id=999)

        assert "Dataset file not found" in str(exc_info.value)
        assert "999" in str(exc_info.value)

    def test_returns_path_when_file_exists(self, mock_db):
        """Existing DataFile returns correct path."""
        mock_file = MagicMock()
        mock_file.file_type = "csv"
        mock_file.file_name = "test_data"
        mock_file.disk_name = "test_data.csv"
        mock_db.exec.return_value.first.return_value = mock_file

        with patch("app.services.model_run.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = "/uploads"
            result = _helper_generate_file_location(mock_db, file_id=1)

        assert result == "/uploads/test_data.csv"

    def test_returns_path_for_zip_files(self, mock_db):
        """Zip files return the extraction directory (disk_name without .zip)."""
        mock_file = MagicMock()
        mock_file.file_type = "zip"
        mock_file.file_name = "images"
        mock_file.disk_name = "images_abc123.zip"
        mock_db.exec.return_value.first.return_value = mock_file

        with patch("app.services.model_run.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = "/uploads"
            result = _helper_generate_file_location(mock_db, file_id=1)

        assert result == "/uploads/images_abc123"


# ---------------------------------------------------------------------------
# code_generation.py — _file_location
# ---------------------------------------------------------------------------


class TestCodeGenFileLocation:
    def test_returns_filename_for_csv(self):
        """Existing CSV DataFile returns its original filename."""
        mock_file = MagicMock()
        mock_file.file_type = "csv"
        mock_file.file_name = "iris.csv"
        mock_file.disk_name = "abc123.csv"

        result = _file_location(mock_file)

        assert result == "iris.csv"

    def test_returns_dirname_for_zip(self):
        """Zip file returns extraction directory name (filename without .zip)."""
        mock_file = MagicMock()
        mock_file.file_type = "zip"
        mock_file.file_name = "images.zip"
        mock_file.disk_name = "def456.zip"

        result = _file_location(mock_file)

        assert result == "images"


# ---------------------------------------------------------------------------
# code_generation.py — generate_code
# ---------------------------------------------------------------------------


class TestGenerateCode:
    def test_raises_error_when_model_not_found(self, mock_db):
        """Missing ModelBasic raises CodeGenerationError."""
        mock_db.exec.return_value.first.return_value = None

        with pytest.raises(CodeGenerationError) as exc_info:
            generate_code("nonexistent_model", mock_db)

        assert "Model not found" in str(exc_info.value)
        assert "nonexistent_model" in str(exc_info.value)

    def test_raises_error_when_file_not_found(self, mock_db):
        """Missing DataFile (after model found) raises CodeGenerationError."""
        mock_model = MagicMock()
        mock_model.file_id = 123

        # First call returns model, second call returns None (file not found)
        mock_db.exec.return_value.first.side_effect = [mock_model, None]

        with pytest.raises(CodeGenerationError) as exc_info:
            generate_code("test_model", mock_db)

        assert "Dataset file not found" in str(exc_info.value)
        assert "123" in str(exc_info.value)
