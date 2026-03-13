"""Unit tests for the data_process service.

All database interactions use MagicMock — no running DB required.
CSV-based tests use pytest's tmp_path fixture for realistic temp files.
"""

import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.models.data import DataFile, DataProcess
from app.services.data_process import (
    add_target_service,
    delete_one_target_by_id_service,
    get_all_targets_service,
    get_data_metrics,
    get_file_data,
    get_one_target_by_id_service,
    preprocess_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def file_id():
    return uuid.uuid4()


@pytest.fixture()
def mock_db():
    return MagicMock()


@pytest.fixture()
def sample_file(file_id):
    """A DataFile-shaped mock."""
    f = MagicMock(spec=DataFile)
    f.id = file_id
    f.file_name = "iris"
    f.file_type = "csv"
    return f


@pytest.fixture()
def sample_target(file_id):
    """A DataProcess-shaped mock."""
    t = MagicMock(spec=DataProcess)
    t.id = 1
    t.file_id = file_id
    t.target = "species"
    return t


@pytest.fixture()
def classification_csv(tmp_path):
    """Create a small classification CSV and return the directory."""
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
def regression_csv(tmp_path):
    """Create a small regression CSV and return the directory."""
    df = pd.DataFrame(
        {
            "size": [1500, 2000, 2500, 3000],
            "bedrooms": [3, 4, 4, 5],
            "price": [300000, 400000, 500000, 600000],
        }
    )
    df.to_csv(tmp_path / "housing.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# add_target_service
# ---------------------------------------------------------------------------


class TestAddTargetService:
    def test_success(self, mock_db, file_id, sample_file):
        mock_db.exec.return_value.first.return_value = sample_file

        body, status = add_target_service(mock_db, file_id, "species")

        assert status == 201
        assert body["success"] is True
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_file_not_found(self, mock_db, file_id):
        mock_db.exec.return_value.first.return_value = None

        body, status = add_target_service(mock_db, file_id, "species")

        assert status == 400
        assert body["success"] is False
        assert "doesn't exist" in body["message"]


# ---------------------------------------------------------------------------
# get_one_target_by_id_service
# ---------------------------------------------------------------------------


class TestGetOneTargetByIdService:
    def test_success(self, mock_db, file_id, sample_file, sample_target):
        mock_db.exec.side_effect = [
            MagicMock(first=MagicMock(return_value=sample_file)),
            MagicMock(first=MagicMock(return_value=sample_target)),
        ]

        body, status = get_one_target_by_id_service(mock_db, file_id)

        assert status == 200
        assert body["success"] is True
        assert body["data"]["target_field"] == "species"
        assert body["data"]["file_name"] == "iris"

    def test_file_not_found(self, mock_db, file_id):
        mock_db.exec.return_value.first.return_value = None

        body, status = get_one_target_by_id_service(mock_db, file_id)

        assert status == 400
        assert body["success"] is False

    def test_target_not_found(self, mock_db, file_id, sample_file):
        mock_db.exec.side_effect = [
            MagicMock(first=MagicMock(return_value=sample_file)),
            MagicMock(first=MagicMock(return_value=None)),
        ]

        body, status = get_one_target_by_id_service(mock_db, file_id)

        assert status == 400
        assert "doesn't exist" in body["message"]


# ---------------------------------------------------------------------------
# delete_one_target_by_id_service
# ---------------------------------------------------------------------------


class TestDeleteOneTargetByIdService:
    def test_success(self, mock_db, file_id, sample_file, sample_target):
        mock_db.exec.side_effect = [
            MagicMock(first=MagicMock(return_value=sample_file)),
            MagicMock(first=MagicMock(return_value=sample_target)),
        ]

        body, status = delete_one_target_by_id_service(mock_db, file_id)

        assert status == 200
        assert body["success"] is True
        mock_db.delete.assert_called_once_with(sample_target)
        mock_db.commit.assert_called_once()

    def test_file_not_found(self, mock_db, file_id):
        mock_db.exec.return_value.first.return_value = None

        body, status = delete_one_target_by_id_service(mock_db, file_id)

        assert status == 400
        assert body["success"] is False

    def test_target_not_found(self, mock_db, file_id, sample_file):
        mock_db.exec.side_effect = [
            MagicMock(first=MagicMock(return_value=sample_file)),
            MagicMock(first=MagicMock(return_value=None)),
        ]

        body, status = delete_one_target_by_id_service(mock_db, file_id)

        assert status == 400
        assert "doesn't exist" in body["message"]


# ---------------------------------------------------------------------------
# get_all_targets_service
# ---------------------------------------------------------------------------


class TestGetAllTargetsService:
    def test_success(self, mock_db, file_id):
        row = MagicMock()
        row.file_id = file_id
        row.file_name = "iris"
        row.file_type = "csv"
        row.target = "species"

        mock_db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=1)),
            MagicMock(all=MagicMock(return_value=[row])),
        ]

        body, status = get_all_targets_service(mock_db)

        assert status == 200
        assert body["success"] is True
        assert len(body["data"]) == 1
        assert body["data"][0]["target_field"] == "species"
        assert body["pagination"]["total"] == 1

    def test_empty(self, mock_db):
        mock_db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=0)),
            MagicMock(all=MagicMock(return_value=[])),
        ]

        body, status = get_all_targets_service(mock_db)

        assert status == 200
        assert body["data"] == []
        assert body["pagination"]["total"] == 0


# ---------------------------------------------------------------------------
# get_data_metrics
# ---------------------------------------------------------------------------


class TestGetDataMetrics:
    @patch("app.services.data_process.get_settings")
    def test_classification_csv(self, mock_settings, mock_db, file_id, sample_file, classification_csv):
        mock_settings.return_value.upload_folder = str(classification_csv)
        mock_db.exec.return_value.first.return_value = sample_file

        body, status = get_data_metrics(mock_db, file_id)

        assert status == 200
        assert body["success"] is True
        data = body["data"]
        assert "data_types" in data
        assert "correlation_matrix" in data
        assert "metric" in data
        assert "sepal_length" in data["data_types"]

    @patch("app.services.data_process.get_settings")
    def test_regression_csv(self, mock_settings, mock_db, file_id, regression_csv):
        reg_file = MagicMock(spec=DataFile)
        reg_file.id = file_id
        reg_file.file_name = "housing"
        reg_file.file_type = "csv"

        mock_settings.return_value.upload_folder = str(regression_csv)
        mock_db.exec.return_value.first.return_value = reg_file

        body, status = get_data_metrics(mock_db, file_id)

        assert status == 200
        assert "size" in body["data"]["correlation_matrix"]
        assert "price" in body["data"]["metric"]

    def test_file_not_in_db(self, mock_db, file_id):
        mock_db.exec.return_value.first.return_value = None

        body, status = get_data_metrics(mock_db, file_id)

        assert status == 400
        assert body["success"] is False

    @patch("app.services.data_process.get_settings")
    def test_file_not_on_disk(self, mock_settings, mock_db, file_id, sample_file, tmp_path):
        mock_settings.return_value.upload_folder = str(tmp_path / "nonexistent")
        mock_db.exec.return_value.first.return_value = sample_file

        body, status = get_data_metrics(mock_db, file_id)

        assert status == 500
        assert body["success"] is False

    @patch("app.services.data_process.get_settings")
    def test_empty_csv(self, mock_settings, mock_db, file_id, tmp_path):
        """Metrics on a CSV with headers but no data rows."""
        pd.DataFrame(columns=["a", "b"]).to_csv(tmp_path / "empty.csv", index=False)

        empty_file = MagicMock(spec=DataFile)
        empty_file.file_name = "empty"
        empty_file.file_type = "csv"

        mock_settings.return_value.upload_folder = str(tmp_path)
        mock_db.exec.return_value.first.return_value = empty_file

        body, status = get_data_metrics(mock_db, file_id)

        assert status == 200
        assert body["data"]["metric"]["a"]["count"] == "0"


# ---------------------------------------------------------------------------
# get_file_data
# ---------------------------------------------------------------------------


class TestGetFileData:
    @patch("app.services.data_process.get_settings")
    def test_success(self, mock_settings, mock_db, file_id, sample_file, classification_csv):
        mock_settings.return_value.upload_folder = str(classification_csv)
        mock_db.exec.return_value.first.return_value = sample_file

        body, status = get_file_data(mock_db, file_id)

        assert status == 200
        assert body["success"] is True
        assert body["data"] is not None

    def test_file_not_in_db(self, mock_db, file_id):
        mock_db.exec.return_value.first.return_value = None

        body, status = get_file_data(mock_db, file_id)

        assert status == 400
        assert body["success"] is False

    @patch("app.services.data_process.get_settings")
    def test_file_missing_on_disk(self, mock_settings, mock_db, file_id, sample_file, tmp_path):
        mock_settings.return_value.upload_folder = str(tmp_path / "missing")
        mock_db.exec.return_value.first.return_value = sample_file

        body, status = get_file_data(mock_db, file_id)

        assert status == 500
        assert body["success"] is False


# ---------------------------------------------------------------------------
# preprocess_data
# ---------------------------------------------------------------------------


class TestPreprocessData:
    @patch("app.services.data_process.get_settings")
    def test_drop_column(self, mock_settings, mock_db, file_id, sample_file, classification_csv):
        mock_settings.return_value.upload_folder = str(classification_csv)
        mock_db.exec.return_value.first.return_value = sample_file

        t = MagicMock()
        t.transformation = "Drop Column"
        t.feature = "species"

        body, status = preprocess_data(mock_db, file_id, [t])

        assert status == 200
        df = pd.read_csv(classification_csv / "iris.csv")
        assert "species" not in df.columns

    @patch("app.services.data_process.get_settings")
    def test_categorical_to_numerical(self, mock_settings, mock_db, file_id, sample_file, classification_csv):
        mock_settings.return_value.upload_folder = str(classification_csv)
        mock_db.exec.return_value.first.return_value = sample_file

        t = MagicMock()
        t.transformation = "Categorical to Numerical"
        t.feature = "species"

        body, status = preprocess_data(mock_db, file_id, [t])

        assert status == 200
        df = pd.read_csv(classification_csv / "iris.csv")
        assert pd.api.types.is_integer_dtype(df["species"])

    @patch("app.services.data_process.get_settings")
    def test_one_hot_encoding(self, mock_settings, mock_db, file_id, sample_file, classification_csv):
        mock_settings.return_value.upload_folder = str(classification_csv)
        mock_db.exec.return_value.first.return_value = sample_file

        t = MagicMock()
        t.transformation = "One Hot Encoding"
        t.feature = "species"

        body, status = preprocess_data(mock_db, file_id, [t])

        assert status == 200
        df = pd.read_csv(classification_csv / "iris.csv")
        assert "species" not in df.columns
        assert any("species" in col for col in df.columns)

    def test_file_not_in_db(self, mock_db, file_id):
        mock_db.exec.return_value.first.return_value = None

        body, status = preprocess_data(mock_db, file_id, [])

        assert status == 400
        assert body["success"] is False

    @patch("app.services.data_process.get_settings")
    def test_missing_column(self, mock_settings, mock_db, file_id, sample_file, classification_csv):
        """Dropping a column that doesn't exist returns a 422 error."""
        mock_settings.return_value.upload_folder = str(classification_csv)
        mock_db.exec.return_value.first.return_value = sample_file

        t = MagicMock()
        t.transformation = "Drop Column"
        t.feature = "nonexistent_column"

        body, status = preprocess_data(mock_db, file_id, [t])

        assert status == 422
        assert body["success"] is False
