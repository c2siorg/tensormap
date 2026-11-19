"""Unit and API integration tests for CSV semantic column type inference engine."""

import uuid
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.config import get_settings
from app.models import DataFile, Project
from app.schemas.schema_inference import SemanticType
from app.services.schema_inference_service import (
    infer_dataframe_schema,
    infer_dataset_schema_service,
    infer_series_semantic_type,
)


def test_infer_series_semantic_types():
    """Tests semantic categorization across numeric, boolean, datetime, and key columns."""
    # 1. Numeric continuous
    s_cont = pd.Series([1.23, 4.56, 7.89, 10.11, 12.13], name="sensor_reading")
    assert infer_series_semantic_type(s_cont)[0] == SemanticType.NUMERIC_CONTINUOUS

    # 2. Boolean strings
    s_bool_str = pd.Series(["yes", "no", "yes", "no", "yes"], name="is_active")
    assert infer_series_semantic_type(s_bool_str)[0] == SemanticType.BOOLEAN

    # 3. Boolean numeric
    s_bool_num = pd.Series([0, 1, 1, 0, 1], name="flag")
    assert infer_series_semantic_type(s_bool_num)[0] == SemanticType.BOOLEAN

    # 4. Datetime strings
    s_dt = pd.Series(["2024-01-01 10:00:00", "2024-01-02 12:30:00", "2024-01-03 09:15:00"], name="timestamp")
    assert infer_series_semantic_type(s_dt)[0] == SemanticType.DATETIME

    # 5. Identifier key
    s_id = pd.Series([f"USR_{i:04d}" for i in range(15)], name="user_id")
    assert infer_series_semantic_type(s_id)[0] == SemanticType.IDENTIFIER_KEY

    # 6. Categorical
    s_cat = pd.Series(["low", "medium", "high", "low", "medium"], name="risk_level")
    assert infer_series_semantic_type(s_cat)[0] == SemanticType.CATEGORICAL


def test_infer_dataframe_schema():
    """Tests full DataFrame schema compilation and aggregation counts."""
    df = pd.DataFrame(
        {
            "order_id": [f"ORD_{i}" for i in range(20)],
            "amount": [19.99 + i for i in range(20)],
            "is_paid": ["true", "false"] * 10,
            "country": ["US", "CA", "UK", "DE"] * 5,
        }
    )
    resp = infer_dataframe_schema(df)
    assert resp.total_columns == 4
    assert resp.total_rows == 20
    assert resp.semantic_counts[SemanticType.IDENTIFIER_KEY.value] == 1
    assert resp.semantic_counts[SemanticType.NUMERIC_CONTINUOUS.value] == 1
    assert resp.semantic_counts[SemanticType.BOOLEAN.value] == 1
    assert resp.semantic_counts[SemanticType.CATEGORICAL.value] == 1


def test_infer_dataset_schema_service_not_found(db_session: Session):
    """Tests 404 response for non-existent file ID."""
    body, status = infer_dataset_schema_service(db_session, file_id=uuid.uuid4())
    assert status == 404
    assert body["success"] is False


def test_infer_dataset_schema_service_and_api(db_session: Session, client: TestClient):
    """Tests end-to-end database service and FastAPI HTTP GET endpoint."""
    project = Project(name="Schema Test Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    disk_name = f"test_schema_{uuid.uuid4().hex[:8]}.csv"
    upload_dir = Path(get_settings().upload_folder).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    csv_file = upload_dir / disk_name
    csv_file.write_text(
        "patient_id,created_at,bmi,diabetic\n"
        + "\n".join(f"P_{i:03d},2024-05-{i:02d},2{i}.5,{'yes' if i % 2 == 0 else 'no'}" for i in range(1, 15))
    )

    try:
        data_file = DataFile(
            file_name="test_data.csv",
            file_type="csv",
            disk_name=disk_name,
            project_id=project.id,
        )
        db_session.add(data_file)
        db_session.commit()
        db_session.refresh(data_file)

        body, status = infer_dataset_schema_service(db_session, file_id=data_file.id)
        assert status == 200
        assert body["success"] is True
        assert body["data"]["total_columns"] == 4

        resp = client.get(f"/api/v1/data/process/schema/{data_file.id}?sample_size=100")
        assert resp.status_code == 200
        resp_data = resp.json()["data"]
        col_map = {c["name"]: c["semantic_type"] for c in resp_data["columns"]}
        assert col_map["patient_id"] == "identifier_key"
        assert col_map["created_at"] == "datetime"
        assert col_map["bmi"] == "numeric_continuous"
        assert col_map["diabetic"] == "boolean"
    finally:
        if csv_file.exists():
            csv_file.unlink()


def test_paid_and_valid_not_misclassified_as_identifier():
    """Tests that words ending with 'id' such as 'paid', 'grid', 'android' are not treated as identifiers."""
    s_paid = pd.Series([10.0 + i for i in range(15)], name="paid")
    assert infer_series_semantic_type(s_paid)[0] != SemanticType.IDENTIFIER_KEY

    s_grid = pd.Series([f"G_{i}" for i in range(15)], name="grid")
    assert infer_series_semantic_type(s_grid)[0] != SemanticType.IDENTIFIER_KEY


def test_constant_column_not_misclassified_as_boolean():
    """Tests that single-value constant column is not treated as boolean."""
    s_const = pd.Series([1, 1, 1, 1, 1], name="all_ones")
    assert infer_series_semantic_type(s_const)[0] != SemanticType.BOOLEAN


def test_empty_csv_returns_400(db_session: Session):
    """Tests that an empty CSV file returns 400."""
    project = Project(name="Empty Test Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    disk_name = f"empty_{uuid.uuid4().hex[:8]}.csv"
    upload_dir = Path(get_settings().upload_folder).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    csv_file = upload_dir / disk_name
    csv_file.write_text("")

    try:
        data_file = DataFile(
            file_name="empty.csv",
            file_type="csv",
            disk_name=disk_name,
            project_id=project.id,
        )
        db_session.add(data_file)
        db_session.commit()
        db_session.refresh(data_file)

        body, status = infer_dataset_schema_service(db_session, file_id=data_file.id)
        assert status == 400
        assert body["success"] is False
    finally:
        if csv_file.exists():
            csv_file.unlink()
