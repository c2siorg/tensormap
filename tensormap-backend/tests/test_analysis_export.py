"""HTTP-level coverage for cached model analysis exports."""

import csv
import io
import json
from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import Session

from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus


def create_completed_job(db_session: Session, *, model_type: int = 1, cache=None) -> str:
    model = ModelBasic(model_name=f"export-{uuid4()}", graph_ir={}, model_type=model_type)
    db_session.add(model)
    db_session.commit()
    job_id = str(uuid4())
    db_session.add(
        TrainingJob(
            id=job_id,
            model_id=model.id,
            status=TrainingStatus.COMPLETED,
            completed_at=datetime.now(UTC),
            analysis_cache=cache,
        )
    )
    db_session.commit()
    return job_id


def parse_csv_results(response_text: str) -> dict:
    """Parse the documented typed JSON Pointer rows back into cached values."""
    nodes = {}
    for row in csv.DictReader(io.StringIO(response_text, newline="")):
        field = row["field"]
        if field == "":
            path = ()
        else:
            assert field.startswith("/")
            path = tuple(segment.replace("~1", "/").replace("~0", "~") for segment in field[1:].split("/"))
        nodes[path] = (row["type"], row["value"])

    def decode(path):
        node_type, value = nodes[path]
        if node_type == "scalar":
            return json.loads(value)
        children = sorted(
            candidate for candidate in nodes if len(candidate) == len(path) + 1 and candidate[:-1] == path
        )
        if not children:
            return json.loads(value)
        if node_type == "array":
            return [decode(child) for child in sorted(children, key=lambda child: int(child[-1]))]
        return {child[-1]: decode(child) for child in children}

    return decode(())


def test_json_export_contains_only_requested_job_cached_results(client, db_session):
    first_id = create_completed_job(
        db_session,
        cache={"confusion_matrix": {"confusion_matrix": [[2, 1]], "class_names": ["A"]}},
    )
    create_completed_job(db_session, cache={"confusion_matrix": {"confusion_matrix": [[99]]}})

    response = client.get(f"/api/v1/model/analysis/{first_id}/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.headers["content-disposition"] == f'attachment; filename="analysis-{first_id}.json"'
    assert response.json() == {
        "job_id": first_id,
        "analysis_type": "classification",
        "results": {"confusion_matrix": {"confusion_matrix": [[2, 1]], "class_names": ["A"]}},
        "unavailable_results": ["roc_curve", "feature_importance", "predictions"],
    }


def test_csv_export_flattens_nested_results_and_keeps_numeric_values(client, db_session):
    job_id = create_completed_job(
        db_session,
        cache={"feature_importance": {"features": ["age", "income"], "importances_mean": [0.4, -0.1]}},
    )

    response = client.get(f"/api/v1/model/analysis/{job_id}/export?format=csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == f'attachment; filename="analysis-{job_id}.csv"'
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert rows == [
        {"analysis": "feature_importance", "field": "", "type": "object", "value": ""},
        {"analysis": "feature_importance", "field": "/features", "type": "array", "value": ""},
        {"analysis": "feature_importance", "field": "/features/0", "type": "scalar", "value": '"age"'},
        {"analysis": "feature_importance", "field": "/features/1", "type": "scalar", "value": '"income"'},
        {"analysis": "feature_importance", "field": "/importances_mean", "type": "array", "value": ""},
        {"analysis": "feature_importance", "field": "/importances_mean/0", "type": "scalar", "value": "0.4"},
        {"analysis": "feature_importance", "field": "/importances_mean/1", "type": "scalar", "value": "-0.1"},
    ]


def test_csv_export_represents_empty_root_collections(client, db_session):
    for value, expected_type, expected_json in (({}, "object", "{}"), ([], "array", "[]")):
        job_id = create_completed_job(db_session, cache={"confusion_matrix": value})
        response = client.get(f"/api/v1/model/analysis/{job_id}/export?format=csv")

        rows = list(csv.DictReader(io.StringIO(response.text, newline="")))
        assert rows == [
            {
                "analysis": "confusion_matrix",
                "field": "",
                "type": expected_type,
                "value": expected_json,
            }
        ]
        assert parse_csv_results(response.text) == value


def test_csv_export_preserves_nested_empty_null_empty_string_and_missing(client, db_session):
    source = {"nested": {"empty_object": {}, "empty_list": [], "null_value": None, "empty_string": ""}}
    job_id = create_completed_job(db_session, cache={"confusion_matrix": source})

    response = client.get(f"/api/v1/model/analysis/{job_id}/export?format=csv")
    rows = list(csv.DictReader(io.StringIO(response.text, newline="")))
    values_by_path = {row["field"]: row for row in rows}

    assert values_by_path["/nested/empty_object"]["value"] == "{}"
    assert values_by_path["/nested/empty_list"]["value"] == "[]"
    assert values_by_path["/nested/null_value"]["value"] == "null"
    assert values_by_path["/nested/empty_string"]["value"] == '""'
    assert "/nested/missing" not in values_by_path
    assert parse_csv_results(response.text) == source


def test_csv_pointer_paths_and_csv_quoting_round_trip_arbitrary_keys(client, db_session):
    source = {
        "a.b": 'comma, quote " and newline\nΩ',
        "a": {"b": 1},
        "a/b": True,
        "a~b": None,
        "items": [10, 20],
        "bracket[key]": {'quote"': "line,\nnext"},
        "Ω雪": "unicode",
        "": "empty-key",
        "numeric": {"0": "object-zero"},
    }
    job_id = create_completed_job(db_session, cache={"confusion_matrix": source})

    response = client.get(f"/api/v1/model/analysis/{job_id}/export?format=csv")
    rows = list(csv.DictReader(io.StringIO(response.text, newline="")))
    fields = {row["field"] for row in rows}

    assert "/a.b" in fields
    assert "/a/b" in fields
    assert "/a~1b" in fields
    assert "/a~0b" in fields
    assert "/items/0" in fields and "/items/1" in fields
    assert '/bracket[key]/quote"' in fields
    assert "/Ω雪" in fields
    assert "/" in fields
    assert next(row for row in rows if row["field"] == "/numeric")["type"] == "object"
    assert parse_csv_results(response.text) == source


def test_export_reports_missing_results_and_rejects_unknown_format(client, db_session):
    empty_job = create_completed_job(db_session)
    assert client.get(f"/api/v1/model/analysis/{empty_job}/export").status_code == 409

    populated_job = create_completed_job(db_session, cache={"confusion_matrix": {"matrix": [[1]]}})
    assert client.get(f"/api/v1/model/analysis/{populated_job}/export?format=xml").status_code == 422
    assert client.get(f"/api/v1/model/analysis/{uuid4()}/export").status_code == 404


def test_export_omits_task_unsupported_cached_outputs(client, db_session):
    regression_id = create_completed_job(
        db_session,
        model_type=2,
        cache={
            "regression_analysis": {"mae": 0.25, "residuals": [0.1]},
            "confusion_matrix": {"secret_unrelated": True},
        },
    )

    response = client.get(f"/api/v1/model/analysis/{regression_id}/export")

    assert response.status_code == 200
    assert response.json()["results"] == {"regression_analysis": {"mae": 0.25, "residuals": [0.1]}}
    assert "confusion_matrix" not in response.json()["results"]
