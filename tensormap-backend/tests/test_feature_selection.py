import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.feature_selection import _normalise_scores, analyze_features, apply_feature_selection


def _db(file_obj):
    db = MagicMock()
    db.exec.return_value.first.return_value = file_obj
    return db


def _file(file_type="csv"):
    return SimpleNamespace(file_type=file_type, disk_name="dataset.csv", columns=["signal", "noise", "label"])


def _frame():
    return pd.DataFrame(
        {
            "signal": [0, 0, 1, 1, 2, 2, 3, 3],
            "noise": [3, 2, 3, 2, 3, 2, 3, 2],
            "label": ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )


def test_analyze_ranks_target_related_features_and_excludes_target():
    with (
        patch("app.services.feature_selection._get_csv", return_value=(_file(), _frame(), None)),
    ):
        body, status = analyze_features(_db(_file()), uuid.uuid4(), "label")

    assert status == 200
    assert body["data"]["task"] == "classification"
    assert body["data"]["sample_rows"] == 8
    assert [score["feature"] for score in body["data"]["scores"]][0] == "signal"
    assert {score["feature"] for score in body["data"]["scores"]} == {"signal", "noise"}
    assert all(0 <= score["composite_score"] <= 1 for score in body["data"]["scores"])


def test_analyze_rejects_missing_target():
    with patch("app.services.feature_selection._get_csv", return_value=(_file(), _frame(), None)):
        body, status = analyze_features(_db(_file()), uuid.uuid4(), "missing")

    assert status == 422
    assert "Target column" in body["message"]


def test_analyze_auto_detects_numeric_classification_target():
    frame = _frame()
    frame["label"] = [0, 0, 1, 1, 2, 2, 3, 3]
    with patch("app.services.feature_selection._get_csv", return_value=(_file(), frame, None)):
        body, status = analyze_features(_db(_file()), uuid.uuid4(), "label")

    assert status == 200
    assert body["data"]["task"] == "classification"


def test_positive_infinite_metric_receives_maximum_normalized_score():
    scores = _normalise_scores(pd.Series([1.0, float("inf"), 3.0]).to_numpy())

    assert scores.tolist() == [0.0, 1.0, 1.0]


def test_apply_refuses_to_drop_target():
    with patch("app.services.feature_selection._get_csv", return_value=(_file(), _frame(), None)):
        body, status = apply_feature_selection(_db(_file()), uuid.uuid4(), "label", ["label"])

    assert status == 422
    assert "target column cannot be removed" in body["message"]


def test_apply_requires_selected_columns():
    with patch("app.services.feature_selection._get_csv", return_value=(_file(), _frame(), None)):
        body, status = apply_feature_selection(_db(_file()), uuid.uuid4(), "label", [])

    assert status == 422
    assert "Select at least one" in body["message"]


def test_apply_requires_at_least_one_predictor():
    with patch("app.services.feature_selection._get_csv", return_value=(_file(), _frame(), None)):
        body, status = apply_feature_selection(_db(_file()), uuid.uuid4(), "label", ["signal", "noise"])

    assert status == 422
    assert "At least one predictor" in body["message"]


def test_apply_creates_backup_and_updates_column_cache(tmp_path):
    dataset = tmp_path / "dataset.csv"
    original = _frame()
    original.to_csv(dataset, index=False)
    file = _file()
    db = _db(file)
    with (
        patch("app.services.feature_selection.get_settings") as settings,
        patch("app.services.feature_selection.select"),
        patch("app.services.feature_selection.pd.read_csv", return_value=original),
    ):
        settings.return_value.upload_folder = str(tmp_path)
        body, status = apply_feature_selection(db, uuid.uuid4(), "label", ["noise"])

    assert status == 200
    assert pd.read_csv(str(dataset) + ".bak").equals(original)
    assert list(pd.read_csv(dataset).columns) == ["signal", "label"]
    assert file.columns == ["signal", "label"]
    assert body["data"]["removed_columns"] == ["noise"]


def test_apply_does_not_prune_if_backup_fails(tmp_path):
    dataset = tmp_path / "dataset.csv"
    original = _frame()
    original.to_csv(dataset, index=False)
    before = dataset.read_bytes()
    existing_backup = tmp_path / "dataset.csv.bak"
    existing_backup.write_text("previous backup", encoding="utf-8")
    file = _file()
    with (
        patch("app.services.feature_selection.get_settings") as settings,
        patch("app.services.feature_selection.select"),
        patch("app.services.feature_selection.pd.read_csv", return_value=original),
        patch("app.services.feature_selection.shutil.copy2", side_effect=OSError("read-only")),
    ):
        settings.return_value.upload_folder = str(tmp_path)
        body, status = apply_feature_selection(_db(file), uuid.uuid4(), "label", ["noise"])

    assert status == 500
    assert dataset.read_bytes() == before
    assert existing_backup.read_text(encoding="utf-8") == "previous backup"


def test_apply_restores_dataset_when_database_commit_fails(tmp_path):
    dataset = tmp_path / "dataset.csv"
    original = _frame()
    original.to_csv(dataset, index=False)
    file = _file()
    db = _db(file)
    db.commit.side_effect = RuntimeError("database unavailable")
    with (
        patch("app.services.feature_selection.get_settings") as settings,
        patch("app.services.feature_selection.pd.read_csv", return_value=original),
    ):
        settings.return_value.upload_folder = str(tmp_path)
        body, status = apply_feature_selection(db, uuid.uuid4(), "label", ["noise"])

    assert status == 500
    assert "database unavailable" in body["message"]
    assert pd.read_csv(dataset).equals(original)
    db.rollback.assert_called_once()


@pytest.mark.parametrize("file_type", ["zip", "parquet"])
def test_apply_rejects_non_csv(file_type):
    with patch(
        "app.services.feature_selection._get_csv",
        return_value=(None, None, ({"success": False, "message": "CSV only", "data": None}, 400)),
    ):
        _, status = apply_feature_selection(_db(_file(file_type)), uuid.uuid4(), "label", ["noise"])

    assert status == 400
