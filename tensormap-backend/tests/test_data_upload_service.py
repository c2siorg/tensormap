import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.models.data import DataFile
from app.services.data_upload import delete_one_file_by_id_service


@pytest.fixture()
def file_id():
    return uuid.uuid4()


@pytest.fixture()
def mock_db():
    return MagicMock()


@pytest.fixture()
def sample_file(file_id):
    f = MagicMock(spec=DataFile)
    f.id = file_id
    f.disk_name = "some_file.csv"
    f.file_type = "csv"
    return f


class TestDeleteOneFileById:
    def test_returns_400_when_file_not_found(self, mock_db, file_id):
        mock_db.exec.return_value.first.return_value = None
        body, status = delete_one_file_by_id_service(mock_db, file_id)
        assert status == 400
        assert body["success"] is False

    def test_happy_path_deletes_db_and_disk(self, mock_db, file_id, sample_file):
        mock_db.exec.return_value.first.return_value = sample_file
        mock_db.exec.return_value.all.return_value = []

        with (
            patch("app.services.data_upload.os.remove") as mock_remove,
            patch("app.services.data_upload.get_settings") as mock_settings,
        ):
            mock_settings.return_value.upload_folder = "/tmp"
            body, status = delete_one_file_by_id_service(mock_db, file_id)

        assert status == 200
        assert body["success"] is True
        mock_db.delete.assert_called_once_with(sample_file)
        mock_db.commit.assert_called_once()
        mock_remove.assert_called_once()

    def test_disk_removed_after_db_commit(self, mock_db, file_id, sample_file):
        mock_db.exec.return_value.first.return_value = sample_file
        mock_db.exec.return_value.all.return_value = []

        call_order = []
        mock_db.commit.side_effect = lambda: call_order.append("commit")

        with (
            patch("app.services.data_upload.os.remove", side_effect=lambda p: call_order.append("remove")),
            patch("app.services.data_upload.get_settings") as mock_settings,
        ):
            mock_settings.return_value.upload_folder = "/tmp"
            delete_one_file_by_id_service(mock_db, file_id)

        assert call_order == ["commit", "remove"], f"Expected commit before remove, got {call_order}"

    def test_db_delete_failure_does_not_remove_disk(self, mock_db, file_id, sample_file):
        """If db.commit() raises, the disk file must NOT be removed."""
        mock_db.exec.return_value.first.return_value = sample_file
        mock_db.exec.return_value.all.return_value = []
        mock_db.commit.side_effect = SQLAlchemyError("DB commit failed")

        removed = []

        with (
            patch("app.services.data_upload.os.remove", side_effect=lambda p: removed.append(p)),
            patch("app.services.data_upload.get_settings") as mock_settings,
        ):
            mock_settings.return_value.upload_folder = "/tmp"
            body, status = delete_one_file_by_id_service(mock_db, file_id)

        assert status == 500
        assert len(removed) == 0, f"os.remove was called despite commit failure: {removed}"

    def test_nulls_all_model_basic_file_ids(self, mock_db, file_id, sample_file):
        mb1, mb2 = MagicMock(), MagicMock()
        mb1.file_id = file_id
        mb2.file_id = file_id
        mock_db.exec.return_value.first.return_value = sample_file
        mock_db.exec.return_value.all.return_value = [mb1, mb2]

        with (
            patch("app.services.data_upload.os.remove"),
            patch("app.services.data_upload.get_settings") as mock_settings,
        ):
            mock_settings.return_value.upload_folder = "/tmp"
            delete_one_file_by_id_service(mock_db, file_id)

        assert mb1.file_id is None
        assert mb2.file_id is None
        assert mock_db.add.call_count >= 2
