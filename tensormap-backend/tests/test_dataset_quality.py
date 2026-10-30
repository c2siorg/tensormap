"""Tests for read-only training dataset quality reports."""

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.models.data import DataFile
from app.services.dataset_quality import analyze_dataset_quality


def _analyze(
    tmp_path: Path,
    content: str | bytes,
    target: str | None = "label",
    problem_type: int = 1,
    training_split: float | None = None,
):
    """Write a small CSV and call the service with a mock DataFile row."""
    path = tmp_path / "uploaded.csv"
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    file = MagicMock(spec=DataFile)
    file.file_type = "csv"
    file.disk_name = path.name
    file.row_count = None
    file.columns = None
    db = MagicMock()
    db.exec.return_value.first.return_value = file
    with patch("app.services.dataset_quality.get_settings") as settings:
        settings.return_value.upload_folder = str(tmp_path)
        settings.return_value.max_content_length = 200 * 1024 * 1024
        body, status = analyze_dataset_quality(db, uuid.uuid4(), target, problem_type, training_split)
    return body, status


def _issues(report):
    return {issue["code"]: issue for issue in report["issues"]}


@pytest.mark.parametrize("problem_type", [1, 2])
def test_clean_classification_and_regression_datasets_are_allowed(tmp_path, problem_type):
    body, status = _analyze(tmp_path, "feature,label\n0,1\n1,2\n2,3\n", problem_type=problem_type)
    assert status == 200
    assert body["data"]["status"] == "clean"
    assert body["data"]["can_train"] is True
    assert body["data"]["issues"] == []
    assert body["data"]["summary"]["rows"] == 3


@pytest.mark.parametrize(
    "target_values,problem_type",
    [
        ("0\n1\n0\n1\n", 1),
        ("setosa\nversicolor\nvirginica\n", 1),
    ],
)
def test_binary_and_multiclass_targets_are_allowed(tmp_path, target_values, problem_type):
    rows = "".join(f"{i},{value}\n" for i, value in enumerate(target_values.splitlines()))
    body, status = _analyze(tmp_path, f"feature,label\n{rows}", problem_type=problem_type)
    assert status == 200
    assert body["data"]["can_train"] is True
    assert "single_class_target" not in _issues(body["data"])


def test_single_class_classification_target_is_blocking(tmp_path):
    body, status = _analyze(tmp_path, "feature,label\n0,cat\n1,cat\n2,cat\n", problem_type=1)
    issue = _issues(body["data"])["single_class_target"]
    assert status == 200
    assert body["data"]["can_train"] is False
    assert issue["blocking"] is True
    assert issue["columns"] == ["label"]
    assert "at least two classes" in issue["message"]


def test_target_class_collapse_after_dropping_missing_rows_is_blocking(tmp_path):
    # 'cat' row has missing f1, so it gets dropped, leaving only 'dog' in usable rows
    body, status = _analyze(
        tmp_path,
        "f1,f2,label\n,2,cat\n1,2,dog\n3,4,dog\n",
        target="label",
        problem_type=1,
    )
    assert status == 200
    assert body["data"]["can_train"] is False
    issue = _issues(body["data"])["single_class_target"]
    assert issue["blocking"] is True
    assert issue["distinct_count"] == 1
    assert "after dropping rows with missing feature values" in issue["message"]


def test_constant_regression_target_remains_allowed(tmp_path):
    body, status = _analyze(
        tmp_path,
        "feature,target\n0,5.0\n1,5.0\n2,5.0\n",
        target="target",
        problem_type=2,
    )
    assert status == 200
    assert body["data"]["can_train"] is True
    assert "single_class_target" not in _issues(body["data"])


def test_missing_values_report_counts_percentage_and_valid_zero(tmp_path):
    body, _ = _analyze(tmp_path, "feature,label\n0,1\n1,\n2,3\n")
    missing_issues = [issue for issue in body["data"]["issues"] if issue["code"] == "missing_values"]
    assert [(issue["columns"][0], issue["count"]) for issue in missing_issues] == [("label", 1)]
    assert missing_issues[0]["percentage"] == pytest.approx(33.33)
    assert body["data"]["summary"]["missing_values"] == 1
    assert body["data"]["can_train"] is True
    assert all(issue["severity"] == "warning" for issue in body["data"]["issues"])


def test_entirely_empty_column_is_reported_separately(tmp_path):
    body, _ = _analyze(tmp_path, "feature,label,empty\n0,1,\n1,2,\n")
    issues = _issues(body["data"])
    assert issues["empty_column"]["columns"] == ["empty"]
    assert issues["empty_column"]["distinct_count"] == 0
    assert issues["no_usable_rows"]["blocking"] is True


def test_constant_column_counts_nonmissing_values_only(tmp_path):
    body, _ = _analyze(tmp_path, "constant,label\n7,0\n,1\n7,2\n")
    issues = _issues(body["data"])
    assert issues["constant_column"]["columns"] == ["constant"]
    assert issues["constant_column"]["distinct_count"] == 1
    assert body["data"]["can_train"] is True


@pytest.mark.parametrize("content,expected_columns", [("feature,label\n", 2), ("", 0)])
def test_empty_dataset_is_blocking(tmp_path, content, expected_columns):
    body, _ = _analyze(tmp_path, content)
    report = body["data"]
    assert report["status"] == "blocked"
    assert report["can_train"] is False
    assert report["summary"]["rows"] == 0
    assert report["summary"]["columns"] == expected_columns
    assert "empty_dataset" in _issues(report)


def test_all_missing_rows_leave_no_usable_training_rows(tmp_path):
    body, _ = _analyze(tmp_path, "feature,label\n,\n,\n")
    issues = _issues(body["data"])
    assert issues["no_usable_rows"]["blocking"] is True
    assert body["data"]["summary"]["usable_rows"] == 0


def test_duplicate_header_limitation_is_reported(tmp_path):
    body, _ = _analyze(tmp_path, "feature,feature,label\n1,2,0\n2,3,1\n")
    assert body["data"]["summary"]["columns"] == 3
    assert body["data"]["limitations"]
    assert "Duplicate CSV headers" in body["data"]["limitations"][0]


def test_target_column_name_is_matched_without_trimming(tmp_path):
    body, _ = _analyze(tmp_path, "feature, label \n0,1\n1,2\n", target=" label ")
    assert body["data"]["status"] == "clean"
    assert body["data"]["can_train"] is True


def test_whitespace_only_target_is_treated_as_missing(tmp_path):
    body, _ = _analyze(tmp_path, "feature,label\n0,1\n1,2\n", target="  ")
    assert _issues(body["data"])["target_required"]["blocking"] is True


@pytest.mark.parametrize(
    "target,expected",
    [(None, "target_required"), ("missing", "target_not_found"), ("empty", "empty_target")],
)
def test_invalid_target_is_blocking(tmp_path, target, expected):
    body, _ = _analyze(tmp_path, "feature,empty,label\n0,,1\n1,,2\n", target=target)
    assert _issues(body["data"])[expected]["blocking"] is True
    assert body["data"]["can_train"] is False


def test_all_missing_classification_target_is_blocking_as_empty_target(tmp_path):
    body, _ = _analyze(tmp_path, "feature,label\n0,\n1,\n", problem_type=1)
    assert _issues(body["data"])["empty_target"]["blocking"] is True
    assert "single_class_target" not in _issues(body["data"])
    assert body["data"]["can_train"] is False


def test_target_only_dataset_is_blocked_for_tabular_training(tmp_path):
    body, _ = _analyze(tmp_path, "label\n0\n1\n")
    assert _issues(body["data"])["no_feature_columns"]["blocking"] is True


def test_split_that_produces_an_empty_partition_is_blocking(tmp_path):
    body, _ = _analyze(tmp_path, "feature,label\n0,1\n", training_split=80)
    issue = _issues(body["data"])["empty_training_split"]
    assert issue["blocking"] is True
    assert issue["training_rows"] == 0
    assert issue["evaluation_rows"] == 1


def test_mixed_warnings_and_blocking_errors_are_separated(tmp_path):
    body, _ = _analyze(tmp_path, "constant,target\n1,\n1,\n")
    report = body["data"]
    assert report["status"] == "blocked"
    assert report["can_train"] is False
    assert any(issue["severity"] == "warning" for issue in report["issues"])
    assert any(issue["blocking"] for issue in report["issues"])


def test_image_archive_report_is_not_applicable_and_does_not_require_target():
    file = MagicMock(spec=DataFile)
    file.file_type = "zip"
    file.row_count = None
    file.columns = None
    db = MagicMock()
    db.exec.return_value.first.return_value = file
    body, status = analyze_dataset_quality(db, uuid.uuid4(), None, problem_type_id=3)
    assert status == 200
    assert body["data"]["status"] == "not_applicable"
    assert body["data"]["can_train"] is True


def test_malformed_csv_returns_safe_client_error(tmp_path):
    body, status = _analyze(tmp_path, b"feature,label\n\xff,1\n")
    assert status == 400
    assert body["success"] is False
    assert "uploaded.csv" not in body["message"]


def test_missing_file_returns_404():
    db = MagicMock()
    db.exec.return_value.first.return_value = None
    body, status = analyze_dataset_quality(db, uuid.uuid4())
    assert status == 404
    assert body["success"] is False


def test_dataset_path_cannot_escape_upload_folder(tmp_path):
    outside = tmp_path.parent / "outside.csv"
    outside.write_text("feature,label\n0,1\n1,2\n")
    file = MagicMock(spec=DataFile)
    file.file_type = "csv"
    file.disk_name = "../outside.csv"
    db = MagicMock()
    db.exec.return_value.first.return_value = file
    with patch("app.services.dataset_quality.get_settings") as settings:
        settings.return_value.upload_folder = str(tmp_path)
        settings.return_value.max_content_length = 200 * 1024 * 1024
        body, status = analyze_dataset_quality(db, uuid.uuid4(), "label")
    assert status == 400
    assert "outside.csv" not in body["message"]


def test_quality_report_api_uses_standard_response_envelope(client, db_session, tmp_path, monkeypatch):
    file_id = uuid.uuid4()
    (tmp_path / "api.csv").write_text("feature,label\n0,1\n1,2\n")
    db_session.add(
        DataFile(id=file_id, file_name="api", file_type="csv", disk_name="api.csv", columns=["feature", "label"])
    )
    db_session.commit()
    monkeypatch.setattr(
        "app.services.dataset_quality.get_settings",
        lambda: SimpleNamespace(upload_folder=str(tmp_path), max_content_length=200 * 1024 * 1024),
    )

    response = client.get(
        f"/api/v1/data/process/quality/{file_id}",
        params={"target_field": "label", "problem_type_id": 2, "training_split": 80},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["can_train"] is True
    assert body["data"]["status"] == "clean"
