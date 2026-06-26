import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.models.data import DataFile
from app.services.data_upload import (
    add_file_service,
    delete_one_file_by_id_service,
    get_all_files_service,
)


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
        r1, r2 = MagicMock(), MagicMock()
        r1.first.return_value = sample_file
        r2.all.return_value = []
        mock_db.exec.side_effect = [r1, r2]

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
        r1, r2 = MagicMock(), MagicMock()
        r1.first.return_value = sample_file
        r2.all.return_value = []
        mock_db.exec.side_effect = [r1, r2]

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
        r1, r2 = MagicMock(), MagicMock()
        r1.first.return_value = sample_file
        r2.all.return_value = []
        mock_db.exec.side_effect = [r1, r2]
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
        r1, r2 = MagicMock(), MagicMock()
        r1.first.return_value = sample_file
        r2.all.return_value = [mb1, mb2]
        mock_db.exec.side_effect = [r1, r2]

        with (
            patch("app.services.data_upload.os.remove"),
            patch("app.services.data_upload.get_settings") as mock_settings,
        ):
            mock_settings.return_value.upload_folder = "/tmp"
            delete_one_file_by_id_service(mock_db, file_id)

        assert mb1.file_id is None
        assert mb2.file_id is None
        assert mock_db.add.call_count >= 2


class TestAddFileService:
    def _make_fw(self, filename: str, size: int = 100) -> MagicMock:
        fw = MagicMock()
        fw.filename = filename
        fw.file.seek = MagicMock()
        fw.file.tell = MagicMock(return_value=size)
        return fw

    def test_rejects_oversized_file(self):
        db = MagicMock()
        fw = self._make_fw("data.csv", size=300 * 1024 * 1024)
        with patch("app.services.data_upload.get_settings") as ms:
            ms.return_value.max_content_length = 200 * 1024 * 1024
            ms.return_value.upload_folder = "/tmp/tm_test"
            body, status = add_file_service(db, fw)
        assert status == 413
        assert body["success"] is False

    def test_rejects_when_file_size_unreadable(self):
        db = MagicMock()
        fw = self._make_fw("data.csv")
        fw.file.seek.side_effect = OSError("seek failed")
        with patch("app.services.data_upload.get_settings") as ms:
            ms.return_value.max_content_length = 200 * 1024 * 1024
            ms.return_value.upload_folder = "/tmp/tm_test"
            body, status = add_file_service(db, fw)
        assert status == 413
        assert body["success"] is False

    def test_csv_happy_path_creates_db_record(self, tmp_path):
        db = MagicMock()
        fw = MagicMock()
        fw.filename = "iris.csv"
        fw.file.seek = MagicMock()
        fw.file.tell = MagicMock(return_value=100)
        fw.save = MagicMock(side_effect=lambda p: None)

        with (
            patch("app.services.data_upload.get_settings") as ms,
            patch("app.services.data_upload.os.makedirs"),
            patch(
                "app.services.data_upload.pd.read_csv",
                side_effect=[
                    pd.DataFrame(columns=["a", "b"]),
                    iter([pd.DataFrame({"a": [1, 2], "b": [3, 4]})]),
                ],
            ),
        ):
            ms.return_value.max_content_length = 200 * 1024 * 1024
            ms.return_value.upload_folder = str(tmp_path)
            body, status = add_file_service(db, fw)

        assert status == 201
        assert body["success"] is True
        db.add.assert_called_once()
        db.commit.assert_called_once()

    def test_zip_happy_path_creates_db_record_and_image_properties(self, tmp_path):
        db = MagicMock()
        fw = MagicMock()
        fw.filename = "dataset.zip"
        fw.file.seek = MagicMock()
        fw.file.tell = MagicMock(return_value=500)
        fw.save = MagicMock(side_effect=lambda p: None)

        with (
            patch("app.services.data_upload.get_settings") as ms,
            patch("app.services.data_upload.os.makedirs"),
            patch("app.services.data_upload.zipfile.ZipFile") as mzf,
        ):
            ms.return_value.max_content_length = 200 * 1024 * 1024
            ms.return_value.upload_folder = str(tmp_path)

            mock_zf = MagicMock()
            mock_zf.__enter__ = MagicMock(return_value=mock_zf)
            mock_zf.__exit__ = MagicMock(return_value=False)
            mock_zf.namelist.return_value = ["img/cat.png"]
            mock_info = MagicMock()
            mock_info.file_size = 500
            mock_zf.getinfo.return_value = mock_info
            mzf.return_value = mock_zf

            body, status = add_file_service(db, fw)

        assert status == 201
        assert body["success"] is True
        assert db.add.call_count == 2
        db.flush.assert_called_once()
        db.commit.assert_called_once()

    def test_zip_path_traversal_rejected(self, tmp_path):
        db = MagicMock()
        fw = MagicMock()
        fw.filename = "evil.zip"
        fw.file.seek = MagicMock()
        fw.file.tell = MagicMock(return_value=100)
        fw.save = MagicMock()

        with (
            patch("app.services.data_upload.get_settings") as ms,
            patch("app.services.data_upload.os.makedirs"),
            patch("app.services.data_upload.zipfile.ZipFile") as mzf,
            patch("app.services.data_upload.os.remove"),
            patch("app.services.data_upload.shutil.rmtree"),
        ):
            ms.return_value.max_content_length = 200 * 1024 * 1024
            ms.return_value.upload_folder = str(tmp_path)

            mock_zf = MagicMock()
            mock_zf.__enter__ = MagicMock(return_value=mock_zf)
            mock_zf.__exit__ = MagicMock(return_value=False)
            mock_zf.namelist.return_value = ["../../../etc/passwd"]
            mock_info = MagicMock()
            mock_info.file_size = 50
            mock_zf.getinfo.return_value = mock_info
            mzf.return_value = mock_zf

            body, status = add_file_service(db, fw)

        assert status == 400
        assert body["success"] is False
        assert "traversal" in body["message"].lower()


class TestGetAllFilesService:
    def test_returns_paginated_files(self):
        db = MagicMock()
        f = MagicMock()
        f.id = uuid.uuid4()
        f.file_name = "train"
        f.file_type = "csv"
        f.disk_name = "train_abc.csv"
        f.columns = ["x", "y"]
        f.row_count = 10

        db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=1)),
            MagicMock(all=MagicMock(return_value=[f])),
        ]

        with patch("app.services.data_upload.get_settings") as ms:
            ms.return_value.upload_folder = "/tmp"
            body, status = get_all_files_service(db)

        assert status == 200
        assert body["success"] is True
        assert len(body["data"]) == 1
        assert body["data"][0]["fields"] == ["x", "y"]
        assert body["data"][0]["row_count"] == 10
        assert "pagination" in body

    def test_backfills_columns_for_csv_with_no_cache(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv_path, index=False)

        db = MagicMock()
        f = MagicMock()
        f.id = uuid.uuid4()
        f.file_name = "data"
        f.file_type = "csv"
        f.disk_name = "data.csv"
        f.columns = None
        f.row_count = None

        db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=1)),
            MagicMock(all=MagicMock(return_value=[f])),
        ]

        with patch("app.services.data_upload.get_settings") as ms:
            ms.return_value.upload_folder = str(tmp_path)
            body, status = get_all_files_service(db)

        assert status == 200
        assert body["data"][0]["fields"] == ["a", "b"]
        assert body["data"][0]["row_count"] == 1
        db.add.assert_called_once_with(f)
        db.commit.assert_called_once()

    def test_zip_file_has_no_fields(self):
        db = MagicMock()
        f = MagicMock()
        f.id = uuid.uuid4()
        f.file_name = "images"
        f.file_type = "zip"
        f.disk_name = "images.zip"
        f.columns = None
        f.row_count = None

        db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=1)),
            MagicMock(all=MagicMock(return_value=[f])),
        ]

        with patch("app.services.data_upload.get_settings") as ms:
            ms.return_value.upload_folder = "/tmp"
            body, status = get_all_files_service(db)

        assert status == 200
        assert body["data"][0]["fields"] == []
        assert body["data"][0]["row_count"] == 0

    def test_empty_result_set(self):
        db = MagicMock()
        db.exec.side_effect = [
            MagicMock(one=MagicMock(return_value=0)),
            MagicMock(all=MagicMock(return_value=[])),
        ]

        with patch("app.services.data_upload.get_settings") as ms:
            ms.return_value.upload_folder = "/tmp"
            body, status = get_all_files_service(db)

        assert status == 200
        assert body["data"] == []
        assert body["pagination"]["total"] == 0
