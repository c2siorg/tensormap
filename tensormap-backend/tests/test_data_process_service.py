"""Tests for app/services/data_process.py.

All DB interactions use MagicMock – no real database required.
File-system interactions use pytest's tmp_path fixture.
"""

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.models.data import DataFile, DataProcess
from app.schemas.data_process import TransformationItem
from app.services.data_process import (
    add_target_service,
    delete_one_target_by_id_service,
    get_all_targets_service,
    get_data_metrics,
    get_file_data,
    get_one_target_by_id_service,
    preprocess_data,
)

# ── shared helpers ─────────────────────────────────────────────────────────────


def _make_csv(
    tmp_path: Path,
    file_name: str = "dataset",
    content: str = "sepal_length,sepal_width,species\n5.1,3.5,setosa\n6.0,2.9,virginica\n",
) -> tuple[DataFile, Path]:
    """Write a CSV to *tmp_path* and return a matching DataFile stub + its path."""
    csv_path = tmp_path / f"{file_name}.csv"
    csv_path.write_text(content)
    file = DataFile(id=uuid.uuid4(), file_name=file_name, file_type="csv")
    return file, csv_path


def _db_returning(first_value) -> MagicMock:
    """Return a Session mock whose first exec().first() yields *first_value*."""
    db = MagicMock()
    db.exec.return_value.first.return_value = first_value
    return db


def _db_returning_sequence(values: list) -> MagicMock:
    """Return a Session mock whose successive exec().first() calls yield *values*."""
    db = MagicMock()
    db.exec.return_value.first.side_effect = values
    return db


# ── add_target_service ─────────────────────────────────────────────────────────


def test_add_target_success():
    """Creates a DataProcess record and returns 201 when file exists."""
    file = DataFile(id=uuid.uuid4(), file_name="iris", file_type="csv")
    db = _db_returning(file)

    body, code = add_target_service(db, file_id=file.id, target="species")

    assert code == 201
    assert body["success"] is True
    db.add.assert_called_once()
    db.commit.assert_called_once()


def test_add_target_file_not_found():
    """Returns 400 when file_id is absent from the database."""
    db = _db_returning(None)

    body, code = add_target_service(db, file_id=uuid.uuid4(), target="label")

    assert code == 400
    assert body["success"] is False
    db.add.assert_not_called()


# ── get_one_target_by_id_service ───────────────────────────────────────────────


def test_get_one_target_success():
    """Returns 200 with target details when both file and target record exist."""
    file = DataFile(id=uuid.uuid4(), file_name="data", file_type="csv")
    record = DataProcess(file_id=file.id, target="species")
    db = _db_returning_sequence([file, record])

    body, code = get_one_target_by_id_service(db, file_id=file.id)

    assert code == 200
    assert body["success"] is True
    assert body["data"]["target_field"] == "species"
    assert body["data"]["file_name"] == "data"


def test_get_one_target_file_not_found():
    """Returns 400 when the requested file_id is absent from the DB."""
    db = _db_returning(None)

    body, code = get_one_target_by_id_service(db, file_id=uuid.uuid4())

    assert code == 400
    assert body["success"] is False


def test_get_one_target_record_not_found():
    """Returns 400 when file exists but its target record is missing."""
    file = DataFile(id=uuid.uuid4(), file_name="data", file_type="csv")
    db = _db_returning_sequence([file, None])

    body, code = get_one_target_by_id_service(db, file_id=file.id)

    assert code == 400
    assert body["success"] is False


# ── delete_one_target_by_id_service ───────────────────────────────────────────


def test_delete_target_success():
    """Deletes the DataProcess record and returns 200."""
    file = DataFile(id=uuid.uuid4(), file_name="data", file_type="csv")
    record = DataProcess(file_id=file.id, target="species")
    db = _db_returning_sequence([file, record])

    body, code = delete_one_target_by_id_service(db, file_id=file.id)

    assert code == 200
    assert body["success"] is True
    db.delete.assert_called_once_with(record)
    db.commit.assert_called_once()


def test_delete_target_file_not_found():
    """Returns 400 when the file is absent from the DB."""
    db = _db_returning(None)

    body, code = delete_one_target_by_id_service(db, file_id=uuid.uuid4())

    assert code == 400
    assert body["success"] is False


def test_delete_target_record_not_found():
    """Returns 400 when file exists but the target record is absent."""
    file = DataFile(id=uuid.uuid4(), file_name="data", file_type="csv")
    db = _db_returning_sequence([file, None])

    body, code = delete_one_target_by_id_service(db, file_id=file.id)

    assert code == 400
    assert body["success"] is False


# ── get_all_targets_service ────────────────────────────────────────────────────


def test_get_all_targets_returns_paginated_data():
    """Returns 200 with data list and correct pagination metadata."""
    file_id = uuid.uuid4()
    row = SimpleNamespace(file_id=file_id, file_name="iris", file_type="csv", target="species")

    count_mock = MagicMock()
    count_mock.one.return_value = 1
    rows_mock = MagicMock()
    rows_mock.all.return_value = [row]

    db = MagicMock()
    db.exec.side_effect = [count_mock, rows_mock]

    body, code = get_all_targets_service(db, offset=0, limit=10)

    assert code == 200
    assert body["pagination"]["total"] == 1
    assert body["pagination"]["offset"] == 0
    assert body["pagination"]["limit"] == 10
    assert len(body["data"]) == 1
    assert body["data"][0]["target_field"] == "species"
    assert body["data"][0]["file_name"] == "iris"


# ── get_data_metrics ───────────────────────────────────────────────────────────


def test_get_data_metrics_success(tmp_path):
    """Returns 200 with data_types, correlation_matrix, and metric keys."""
    file, _ = _make_csv(tmp_path, content="a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
    db = _db_returning(file)

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, code = get_data_metrics(db, file_id=file.id)

    assert code == 200
    assert body["success"] is True
    assert "data_types" in body["data"]
    assert "correlation_matrix" in body["data"]
    assert "metric" in body["data"]


def test_get_data_metrics_file_not_in_db():
    """Returns 400 when file_id is absent from the DB."""
    db = _db_returning(None)

    body, code = get_data_metrics(db, file_id=uuid.uuid4())

    assert code == 400
    assert body["success"] is False


def test_get_data_metrics_csv_missing_on_disk(tmp_path):
    """Returns 500 when the DB record exists but the CSV is absent on disk."""
    file = DataFile(id=uuid.uuid4(), file_name="ghost_file", file_type="csv")
    db = _db_returning(file)

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, code = get_data_metrics(db, file_id=file.id)

    assert code == 500
    assert body["success"] is False


# ── get_file_data ──────────────────────────────────────────────────────────────


def test_get_file_data_success(tmp_path):
    """Returns 200 with JSON-encoded records for a valid CSV."""
    file, _ = _make_csv(tmp_path)
    db = _db_returning(file)

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, code = get_file_data(db, file_id=file.id)

    assert code == 200
    assert body["success"] is True
    records = json.loads(body["data"])
    assert len(records) == 2
    assert "sepal_length" in records[0]
    assert "species" in records[0]


def test_get_file_data_file_not_in_db():
    """Returns 400 when file_id is absent from the DB."""
    db = _db_returning(None)

    body, code = get_file_data(db, file_id=uuid.uuid4())

    assert code == 400
    assert body["success"] is False


# ── preprocess_data ────────────────────────────────────────────────────────────


def test_preprocess_file_not_in_db():
    """Returns 400 when file_id is absent from the DB."""
    db = _db_returning(None)
    t = TransformationItem(transformation="Drop Column", feature="x")

    body, code = preprocess_data(db, file_id=uuid.uuid4(), transformations=[t])

    assert code == 400
    assert body["success"] is False


def test_preprocess_one_hot_encoding(tmp_path):
    """One Hot Encoding expands a categorical column into dummy columns."""
    file, csv_path = _make_csv(
        tmp_path,
        file_name="colors",
        content="color,val\nred,1\nblue,2\nred,3\n",
    )
    db = _db_returning(file)
    t = TransformationItem(transformation="One Hot Encoding", feature="color")

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, code = preprocess_data(db, file_id=file.id, transformations=[t])

    assert code == 200
    assert body["success"] is True
    result = pd.read_csv(csv_path)
    assert "color" not in result.columns
    assert any(col.startswith("color_") for col in result.columns)


def test_preprocess_categorical_to_numerical(tmp_path):
    """Categorical to Numerical replaces a string column with integer codes."""
    file, csv_path = _make_csv(
        tmp_path,
        file_name="iris",
        content="species,petal_length\nsetosa,1.4\nvirginica,5.6\nsetosa,1.3\n",
    )
    db = _db_returning(file)
    t = TransformationItem(transformation="Categorical to Numerical", feature="species")

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, code = preprocess_data(db, file_id=file.id, transformations=[t])

    assert code == 200
    result = pd.read_csv(csv_path)
    assert pd.api.types.is_integer_dtype(result["species"])


def test_preprocess_drop_column(tmp_path):
    """Drop Column removes the specified column and preserves the rest."""
    file, csv_path = _make_csv(
        tmp_path,
        file_name="tabular",
        content="a,b,c\n1,2,3\n4,5,6\n",
    )
    db = _db_returning(file)
    t = TransformationItem(transformation="Drop Column", feature="b")

    with patch("app.services.data_process.get_settings") as mock_cfg:
        mock_cfg.return_value.upload_folder = str(tmp_path)
        body, code = preprocess_data(db, file_id=file.id, transformations=[t])

    assert code == 200
    result = pd.read_csv(csv_path)
    assert "b" not in result.columns
    assert list(result.columns) == ["a", "c"]
