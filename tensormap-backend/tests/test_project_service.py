import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.project import Project
from app.services.project import delete_project_service


@pytest.fixture()
def project_id():
    return uuid.uuid4()


@pytest.fixture()
def mock_db():
    return MagicMock()


@pytest.fixture()
def sample_project(project_id):
    p = MagicMock(spec=Project)
    p.id = project_id
    p.name = "test-project"
    return p


@pytest.fixture()
def sample_csv_file():
    f = MagicMock(spec=DataFile)
    f.disk_name = "data.csv"
    f.file_type = "csv"
    return f


@pytest.fixture()
def sample_zip_file():
    f = MagicMock(spec=DataFile)
    f.disk_name = "images.zip"
    f.file_type = "zip"
    return f


@pytest.fixture()
def sample_model():
    m = MagicMock(spec=ModelBasic)
    m.model_name = "my_model"
    return m


class TestDeleteProjectService:
    def test_returns_404_when_not_found(self, mock_db, project_id):
        mock_db.get.return_value = None
        body, status = delete_project_service(mock_db, project_id)
        assert status == 404
        assert body["success"] is False

    def test_deletes_db_and_removes_files(
        self, mock_db, project_id, sample_project, sample_csv_file, sample_model, tmp_path
    ):
        mock_db.get.return_value = sample_project
        mock_db.exec.return_value.all.side_effect = [
            [sample_csv_file],
            [sample_model],
        ]

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("col1,col2\n1,2")

        json_path = tmp_path / "my_model.json"
        json_path.write_text("{}")

        with (
            patch("app.services.project.get_settings") as mock_settings,
            patch("app.services.project.MODEL_GENERATION_LOCATION", str(tmp_path) + "/"),
            patch("app.services.project.MODEL_GENERATION_TYPE", ".json"),
        ):
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = delete_project_service(mock_db, project_id)

        assert status == 200
        assert body["success"] is True
        mock_db.delete.assert_called_once_with(sample_project)
        mock_db.commit.assert_called_once()
        assert not csv_path.exists()
        assert not json_path.exists()

    def test_path_traversal_blocked_on_disk_name(self, mock_db, project_id, sample_project, tmp_path):
        malicious = MagicMock(spec=DataFile)
        malicious.disk_name = "../../etc/passwd"
        malicious.file_type = "csv"

        mock_db.get.return_value = sample_project
        mock_db.exec.return_value.all.side_effect = [
            [malicious],
            [],
        ]

        targeted = tmp_path / "etc" / "passwd"
        targeted.parent.mkdir(parents=True)
        targeted.write_text("root:x:0:0:")

        with (
            patch("app.services.project.get_settings") as mock_settings,
            patch("app.services.project.os.remove") as mock_remove,
        ):
            mock_settings.return_value.upload_folder = str(tmp_path)
            delete_project_service(mock_db, project_id)

        mock_remove.assert_not_called()

    def test_missing_files_dont_cause_error(self, mock_db, project_id, sample_project, sample_csv_file, tmp_path):
        mock_db.get.return_value = sample_project
        mock_db.exec.return_value.all.side_effect = [
            [sample_csv_file],
            [],
        ]

        with (
            patch("app.services.project.get_settings") as mock_settings,
        ):
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = delete_project_service(mock_db, project_id)

        assert status == 200
        assert body["success"] is True

    def test_removes_zip_extraction_dir(self, mock_db, project_id, sample_project, sample_zip_file, tmp_path):
        mock_db.get.return_value = sample_project
        mock_db.exec.return_value.all.side_effect = [
            [sample_zip_file],
            [],
        ]

        zip_path = tmp_path / "images.zip"
        zip_path.write_text("not a real zip")
        extract_dir = tmp_path / "images"
        extract_dir.mkdir()
        (extract_dir / "file.txt").write_text("data")

        with (
            patch("app.services.project.get_settings") as mock_settings,
        ):
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = delete_project_service(mock_db, project_id)

        assert status == 200
        assert not extract_dir.exists()
