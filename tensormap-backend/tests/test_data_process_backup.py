import os
import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.models.data import DataFile
from app.services.data_process import (
    get_dataset_backup_status_service,
    preprocess_data,
    revert_data_process_service,
)
from app.services.data_upload import delete_one_file_by_id_service


@pytest.fixture()
def mock_db():
    return MagicMock()


@pytest.fixture()
def sample_file():
    f = MagicMock(spec=DataFile)
    f.id = uuid.uuid4()
    f.file_name = "iris"
    f.file_type = "csv"
    f.disk_name = "iris.csv"
    f.columns = ["col"]
    return f


class TestPreprocessBackup:
    def test_backup_created_on_happy_path(self, mock_db, sample_file, tmp_path):
        csv = tmp_path / "iris.csv"
        pd.DataFrame({"col": [1.0, 2.0]}).to_csv(csv, index=False)

        mock_db.exec.return_value.first.return_value = sample_file

        with (
            patch("app.services.data_process.get_settings") as mock_settings,
            patch("app.services.data_process.pd.read_csv") as mock_read_csv,
        ):
            mock_read_csv.return_value = pd.DataFrame({"col": [1.0, 2.0]})
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = preprocess_data(
                mock_db,
                uuid.uuid4(),
                [MagicMock(transformation="Drop Column", feature="col", params=None)],
            )

        assert status == 200
        assert (tmp_path / "iris.csv.bak").exists()
        assert csv.exists()

    def test_backup_failure_does_not_block_write(self, mock_db, sample_file, tmp_path):
        csv = tmp_path / "iris.csv"
        pd.DataFrame({"col": [1.0, 2.0]}).to_csv(csv, index=False)

        mock_db.exec.return_value.first.return_value = sample_file

        with (
            patch("app.services.data_process.get_settings") as mock_settings,
            patch("app.services.data_process.pd.read_csv") as mock_read_csv,
            patch("app.services.data_process.shutil.copy2", side_effect=OSError("Permission denied")),
        ):
            mock_read_csv.return_value = pd.DataFrame({"col": [1.0, 2.0]})
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = preprocess_data(
                mock_db,
                uuid.uuid4(),
                [MagicMock(transformation="Drop Column", feature="col", params=None)],
            )

        assert status == 200
        assert csv.read_text() != ""


class TestRevertPreprocessService:
    def test_revert_happy_path_restores_file_and_updates_db(self, mock_db, sample_file, tmp_path):
        """Revert restores the original CSV from .bak, deletes .bak, and updates columns."""
        file_path = tmp_path / "iris.csv"
        bak_path = tmp_path / "iris.csv.bak"

        # Transformed CSV (1 column) vs Original Backup CSV (2 columns)
        pd.DataFrame({"a": [1, 2]}).to_csv(file_path, index=False)
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bak_path, index=False)

        mock_db.exec.return_value.first.return_value = sample_file

        with patch("app.services.data_process.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = revert_data_process_service(mock_db, sample_file.id)

        assert status == 200
        assert body["success"] is True
        assert not bak_path.exists(), "Backup file should be removed after revert"
        assert file_path.exists(), "Restored CSV must exist"

        # Content must now match the backup
        restored_df = pd.read_csv(file_path)
        assert list(restored_df.columns) == ["a", "b"]
        assert sample_file.columns == ["a", "b"]
        mock_db.commit.assert_called_once()

    def test_revert_fails_when_no_bak_file(self, mock_db, sample_file, tmp_path):
        """Revert returns 400 when no .bak backup exists for the dataset."""
        file_path = tmp_path / "iris.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(file_path, index=False)

        mock_db.exec.return_value.first.return_value = sample_file

        with patch("app.services.data_process.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = revert_data_process_service(mock_db, sample_file.id)

        assert status == 400
        assert body["success"] is False
        assert "No preprocessing backup found" in body["message"]

    def test_revert_fails_when_file_not_found_in_db(self, mock_db):
        """Revert returns 400 when file is missing from database."""
        mock_db.exec.return_value.first.return_value = None
        body, status = revert_data_process_service(mock_db, uuid.uuid4())
        assert status == 400
        assert body["success"] is False

    def test_revert_fails_for_non_csv_file(self, mock_db, sample_file):
        """Revert returns 400 when attempting to revert a non-CSV file."""
        sample_file.file_type = "zip"
        mock_db.exec.return_value.first.return_value = sample_file
        body, status = revert_data_process_service(mock_db, sample_file.id)
        assert status == 400
        assert "only supported for CSV" in body["message"]


class TestGetDatasetBackupStatusService:
    def test_backup_status_when_bak_exists(self, mock_db, sample_file, tmp_path):
        """Returns has_backup True and correct byte size when .bak exists."""
        bak_path = tmp_path / "iris.csv.bak"
        pd.DataFrame({"x": [10, 20]}).to_csv(bak_path, index=False)
        expected_size = os.path.getsize(bak_path)

        mock_db.exec.return_value.first.return_value = sample_file

        with patch("app.services.data_process.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = get_dataset_backup_status_service(mock_db, sample_file.id)

        assert status == 200
        assert body["success"] is True
        assert body["data"]["has_backup"] is True
        assert body["data"]["backup_size_bytes"] == expected_size

    def test_backup_status_when_no_bak_exists(self, mock_db, sample_file, tmp_path):
        """Returns has_backup False and backup_size_bytes None when .bak is absent."""
        mock_db.exec.return_value.first.return_value = sample_file

        with patch("app.services.data_process.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = get_dataset_backup_status_service(mock_db, sample_file.id)

        assert status == 200
        assert body["success"] is True
        assert body["data"]["has_backup"] is False
        assert body["data"]["backup_size_bytes"] is None


class TestStorageLeakCleanupOnDelete:
    def test_delete_dataset_removes_orphaned_bak_file(self, mock_db, sample_file, tmp_path):
        """Deleting a dataset also removes any leftover .bak backup file."""
        file_path = tmp_path / "iris.csv"
        bak_path = tmp_path / "iris.csv.bak"

        pd.DataFrame({"x": [1]}).to_csv(file_path, index=False)
        pd.DataFrame({"x": [1]}).to_csv(bak_path, index=False)

        r1, r2 = MagicMock(), MagicMock()
        r1.first.return_value = sample_file
        r2.all.return_value = []
        mock_db.exec.side_effect = [r1, r2]

        with patch("app.services.data_upload.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(tmp_path)
            body, status = delete_one_file_by_id_service(mock_db, sample_file.id)

        assert status == 200
        assert body["success"] is True
        assert not file_path.exists(), "Original file must be removed"
        assert not bak_path.exists(), "Orphaned .bak file must also be removed"


class TestRevertAndBackupEndpoints:
    def test_revert_endpoint_200(self, client, db_session, tmp_path):
        """POST /data/process/revert/{file_id} restores CSV and returns 200."""
        from app.models.project import Project

        project = Project(name="revert_proj")
        db_session.add(project)
        db_session.commit()
        db_session.refresh(project)

        disk_name = "revert_test.csv"
        file_path = tmp_path / disk_name
        bak_path = tmp_path / f"{disk_name}.bak"

        pd.DataFrame({"a": [1]}).to_csv(file_path, index=False)
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(bak_path, index=False)

        data_file = DataFile(
            file_name="revert_test",
            file_type="csv",
            disk_name=disk_name,
            project_id=project.id,
            columns=["a"],
        )
        db_session.add(data_file)
        db_session.commit()
        db_session.refresh(data_file)

        with patch("app.services.data_process.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(tmp_path)
            resp = client.post(f"/api/v1/data/process/revert/{data_file.id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["columns"] == ["a", "b"]

    def test_backup_status_endpoint_200(self, client, db_session, tmp_path):
        """GET /data/process/backup/{file_id} returns backup status."""
        from app.models.project import Project

        project = Project(name="status_proj")
        db_session.add(project)
        db_session.commit()
        db_session.refresh(project)

        disk_name = "status_test.csv"
        file_path = tmp_path / disk_name
        pd.DataFrame({"a": [1]}).to_csv(file_path, index=False)

        data_file = DataFile(
            file_name="status_test",
            file_type="csv",
            disk_name=disk_name,
            project_id=project.id,
        )
        db_session.add(data_file)
        db_session.commit()
        db_session.refresh(data_file)

        with patch("app.services.data_process.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = str(tmp_path)
            # Before backup exists
            resp1 = client.get(f"/api/v1/data/process/backup/{data_file.id}")
            assert resp1.status_code == 200
            assert resp1.json()["data"]["has_backup"] is False

            # Create backup
            (tmp_path / f"{disk_name}.bak").write_text("a,b\n1,2")
            resp2 = client.get(f"/api/v1/data/process/backup/{data_file.id}")
            assert resp2.status_code == 200
            assert resp2.json()["data"]["has_backup"] is True
