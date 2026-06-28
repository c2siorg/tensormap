import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.project import Project
from app.schemas.project import ProjectCreateRequest, ProjectUpdateRequest
from app.services.project import (
    create_project_service,
    delete_project_service,
    get_all_projects_service,
    get_project_by_id_service,
    update_project_service,
)


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
            patch("app.services.project.MODEL_GENERATION_LOCATION", str(tmp_path)),
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

    def test_path_traversal_blocked_on_model_name(self, mock_db, project_id, sample_project, tmp_path):
        malicious = MagicMock(spec=ModelBasic)
        malicious.model_name = "../../etc/bad"

        mock_db.get.return_value = sample_project
        mock_db.exec.return_value.all.side_effect = [
            [],
            [malicious],
        ]

        targeted = tmp_path / "etc" / "bad.json"
        targeted.parent.mkdir(parents=True)
        targeted.write_text("{}")

        with (
            patch("app.services.project.get_settings") as mock_settings,
            patch("app.services.project.MODEL_GENERATION_LOCATION", str(tmp_path)),
            patch("app.services.project.MODEL_GENERATION_TYPE", ".json"),
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


class TestCreateProjectService:
    def _make_project(self, name="Test", description="Desc"):
        p = MagicMock(spec=Project)
        p.id = uuid.uuid4()
        p.name = name
        p.description = description
        p.created_on = datetime(2024, 1, 1, tzinfo=UTC)
        p.updated_on = datetime(2024, 1, 2, tzinfo=UTC)
        return p

    def test_happy_path_returns_201(self, mock_db):
        project = self._make_project()
        mock_db.refresh.side_effect = lambda p: None

        with patch("app.services.project.Project", return_value=project):
            body, status = create_project_service(mock_db, ProjectCreateRequest(name="Test", description="Desc"))

        assert status == 201
        assert body["success"] is True
        assert body["data"]["name"] == project.name
        mock_db.add.assert_called_once_with(project)
        mock_db.commit.assert_called_once()

    def test_db_error_returns_500_and_rollback(self, mock_db):
        mock_db.commit.side_effect = SQLAlchemyError("fail")

        with patch("app.services.project.Project", return_value=MagicMock()):
            body, status = create_project_service(mock_db, ProjectCreateRequest(name="X", description="Y"))

        assert status == 500
        assert body["success"] is False
        mock_db.rollback.assert_called_once()

    def test_response_includes_all_fields(self, mock_db):
        project = self._make_project(name="MyProj", description="My desc")
        mock_db.refresh.side_effect = lambda p: None

        with patch("app.services.project.Project", return_value=project):
            body, status = create_project_service(mock_db, ProjectCreateRequest(name="MyProj", description="My desc"))

        data = body["data"]
        assert "id" in data
        assert "name" in data
        assert "description" in data
        assert "created_on" in data
        assert "updated_on" in data


class TestGetAllProjectsService:
    def _make_row(self):
        row = MagicMock()
        row.id = uuid.uuid4()
        row.name = "Project A"
        row.description = "Desc A"
        row.created_on = datetime(2024, 1, 1, tzinfo=UTC)
        row.updated_on = datetime(2024, 1, 2, tzinfo=UTC)
        row.file_count = 3
        row.model_count = 1
        return row

    def test_returns_paginated_list(self, mock_db):
        row = self._make_row()
        mock_db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=1)),
            MagicMock(all=MagicMock(return_value=[row])),
        ]

        body, status = get_all_projects_service(mock_db)

        assert status == 200
        assert body["success"] is True
        assert len(body["data"]) == 1
        assert body["data"][0]["name"] == "Project A"
        assert body["data"][0]["file_count"] == 3
        assert "pagination" in body

    def test_empty_list_returns_200(self, mock_db):
        mock_db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=0)),
            MagicMock(all=MagicMock(return_value=[])),
        ]

        body, status = get_all_projects_service(mock_db)

        assert status == 200
        assert body["data"] == []
        assert body["pagination"]["total"] == 0

    def test_db_error_returns_500(self, mock_db):
        mock_db.exec.side_effect = SQLAlchemyError("fail")

        body, status = get_all_projects_service(mock_db)

        assert status == 500
        assert body["success"] is False

    def test_pagination_params_forwarded(self, mock_db):
        mock_db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=100)),
            MagicMock(all=MagicMock(return_value=[])),
        ]

        body, status = get_all_projects_service(mock_db, offset=10, limit=5)

        assert body["pagination"]["offset"] == 10
        assert body["pagination"]["limit"] == 5


class TestGetProjectByIdService:
    def _make_project(self):
        p = MagicMock(spec=Project)
        p.id = uuid.uuid4()
        p.name = "Test"
        p.description = "Desc"
        p.created_on = datetime(2024, 1, 1, tzinfo=UTC)
        p.updated_on = datetime(2024, 1, 2, tzinfo=UTC)
        return p

    def test_returns_404_when_not_found(self, mock_db, project_id):
        mock_db.get.return_value = None

        body, status = get_project_by_id_service(mock_db, project_id)

        assert status == 404
        assert body["success"] is False

    def test_returns_project_when_found(self, mock_db, project_id):
        project = self._make_project()
        mock_db.get.return_value = project

        body, status = get_project_by_id_service(mock_db, project_id)

        assert status == 200
        assert body["success"] is True
        assert body["data"]["name"] == project.name

    def test_db_error_returns_500(self, mock_db, project_id):
        mock_db.get.side_effect = SQLAlchemyError("fail")

        body, status = get_project_by_id_service(mock_db, project_id)

        assert status == 500
        assert body["success"] is False


class TestUpdateProjectService:
    def _make_project(self):
        p = MagicMock(spec=Project)
        p.id = uuid.uuid4()
        p.name = "Old Name"
        p.description = "Old Desc"
        p.created_on = datetime(2024, 1, 1, tzinfo=UTC)
        p.updated_on = datetime(2024, 1, 2, tzinfo=UTC)
        return p

    def test_returns_404_when_not_found(self, mock_db, project_id):
        mock_db.get.return_value = None

        body, status = update_project_service(mock_db, project_id, ProjectUpdateRequest(name="New"))

        assert status == 404
        assert body["success"] is False

    def test_updates_fields_and_returns_200(self, mock_db, project_id):
        project = self._make_project()
        mock_db.get.return_value = project
        mock_db.refresh.side_effect = lambda p: None

        body, status = update_project_service(mock_db, project_id, ProjectUpdateRequest(name="New Name"))

        assert status == 200
        assert body["success"] is True
        assert project.name == "New Name"
        mock_db.add.assert_called_once_with(project)
        mock_db.commit.assert_called_once()

    def test_db_error_returns_500_and_rollback(self, mock_db, project_id):
        project = self._make_project()
        mock_db.get.return_value = project
        mock_db.commit.side_effect = SQLAlchemyError("fail")

        body, status = update_project_service(mock_db, project_id, ProjectUpdateRequest(name="New"))

        assert status == 500
        assert body["success"] is False
        mock_db.rollback.assert_called_once()

    def test_partial_update_only_sets_provided_fields(self, mock_db, project_id):
        project = self._make_project()
        project.description = "Keep This"
        mock_db.get.return_value = project
        mock_db.refresh.side_effect = lambda p: None

        update_project_service(mock_db, project_id, ProjectUpdateRequest(name="New Name"))

        assert project.name == "New Name"
        assert project.description == "Keep This"
