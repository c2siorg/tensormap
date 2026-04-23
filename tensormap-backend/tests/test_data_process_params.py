"""Parametrized tests for data_process service."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.models import DataFile
from app.services.data_process import get_file_data


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_settings():
    with patch("app.services.data_process.get_settings") as mock:
        mock.return_value.upload_folder = "/tmp"
        mock.return_value.max_content_length = 200 * 1024 * 1024
        yield mock


@pytest.mark.parametrize(
    "file_type,expected_status",
    [
        ("json", 500),
        ("parquet", 500),
    ],
)
def test_get_file_data_unsupported_file_type(mock_db, mock_settings, file_type, expected_status):
    mock_file = MagicMock(spec=DataFile)
    mock_file.id = uuid.uuid4()
    mock_file.file_name = "test"
    mock_file.file_type = file_type
    mock_db.exec.return_value.first.return_value = mock_file
    with patch("builtins.open", side_effect=FileNotFoundError):
        body, status = get_file_data(mock_db, uuid.uuid4())
    assert status == expected_status


def test_get_file_data_csv_success(tmp_path, mock_db, mock_settings):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\n1,2\n")

    mock_file = MagicMock(spec=DataFile)
    mock_file.id = uuid.uuid4()
    mock_file.file_name = "test"
    mock_file.file_type = "csv"
    mock_db.exec.return_value.first.return_value = mock_file

    body, status = get_file_data(mock_db, mock_file.id)

    assert status == 200
    assert body.get("success") is True
    assert "data" in body
