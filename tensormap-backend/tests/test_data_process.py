import uuid
from unittest.mock import MagicMock, patch

import pandas as pd

from app.schemas.data_process import TransformationItem
from app.services.data_process import preprocess_data


def _make_db(file_name: str = "test_file", file_type: str = "csv") -> MagicMock:
    file = MagicMock()
    file.file_name = file_name
    file.file_type = file_type
    db = MagicMock()
    db.exec.return_value.first.return_value = file
    return db


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
            "category": ["A", "B", "A"],
        }
    )


class TestUnknownTransformation:
    def test_returns_422(self):
        db = _make_db()
        with patch("app.services.data_process.pd.read_csv", return_value=_sample_df()):
            items = [TransformationItem(transformation="INVALID_OP", feature="age")]
            _, status_code = preprocess_data(db, uuid.uuid4(), items)

        assert status_code == 422

    def test_error_message_contains_bad_name(self):
        db = _make_db()
        with patch("app.services.data_process.pd.read_csv", return_value=_sample_df()):
            items = [TransformationItem(transformation="INVALID_OP", feature="age")]
            result, _ = preprocess_data(db, uuid.uuid4(), items)

        assert "INVALID_OP" in result["message"]

    def test_error_message_lists_valid_options(self):
        db = _make_db()
        with patch("app.services.data_process.pd.read_csv", return_value=_sample_df()):
            items = [TransformationItem(transformation="INVALID_OP", feature="age")]
            result, _ = preprocess_data(db, uuid.uuid4(), items)

        assert "Drop Column" in result["message"]
        assert "One Hot Encoding" in result["message"]
        assert "Categorical to Numerical" in result["message"]


class TestColumnNotFound:
    def test_returns_422(self):
        db = _make_db()
        with patch("app.services.data_process.pd.read_csv", return_value=_sample_df()):
            items = [TransformationItem(transformation="Drop Column", feature="nonexistent_col")]
            _, status_code = preprocess_data(db, uuid.uuid4(), items)

        assert status_code == 422

    def test_error_message_contains_missing_column(self):
        db = _make_db()
        with patch("app.services.data_process.pd.read_csv", return_value=_sample_df()):
            items = [TransformationItem(transformation="Drop Column", feature="nonexistent_col")]
            result, _ = preprocess_data(db, uuid.uuid4(), items)

        assert "nonexistent_col" in result["message"]

    def test_error_message_lists_available_columns(self):
        db = _make_db()
        with patch("app.services.data_process.pd.read_csv", return_value=_sample_df()):
            items = [TransformationItem(transformation="Drop Column", feature="nonexistent_col")]
            result, _ = preprocess_data(db, uuid.uuid4(), items)

        assert "age" in result["message"]
        assert "income" in result["message"]
        assert "category" in result["message"]

    def test_second_item_with_bad_column_also_caught(self):
        db = _make_db()
        with patch("app.services.data_process.pd.read_csv", return_value=_sample_df()):
            items = [
                TransformationItem(transformation="Drop Column", feature="age"),
                TransformationItem(transformation="Drop Column", feature="ghost_col"),
            ]
            result, status_code = preprocess_data(db, uuid.uuid4(), items)

        assert status_code == 422
        assert "ghost_col" in result["message"]

    def test_file_not_found_in_db_returns_400(self):
        db = MagicMock()
        db.exec.return_value.first.return_value = None
        items = [TransformationItem(transformation="Drop Column", feature="age")]
        _, status_code = preprocess_data(db, uuid.uuid4(), items)

        assert status_code == 400
