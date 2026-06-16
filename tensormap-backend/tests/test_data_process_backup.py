import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.data_process import preprocess_data


@pytest.fixture()
def mock_db():
    return MagicMock()


@pytest.fixture()
def sample_file():
    f = MagicMock()
    f.file_name = "iris"
    f.file_type = "csv"
    f.disk_name = "iris.csv"
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
