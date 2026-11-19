"""Tests for class imbalance diagnosis, Shannon entropy, and balanced weight calculation."""

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models.data import DataFile, DataProcess


@pytest.fixture
def make_csv_file(tmp_path: Path):
    """Fixture to persist sample dataframe to disk."""

    def _create(df: pd.DataFrame) -> tuple[Path, str]:
        upload_folder = tmp_path / "uploads"
        upload_folder.mkdir(parents=True, exist_ok=True)
        disk_name = f"test_{uuid4().hex[:8]}.csv"
        df.to_csv(upload_folder / disk_name, index=False)
        return upload_folder, disk_name

    return _create


def test_class_imbalance_balanced_binary(client: TestClient, db_session: Session, make_csv_file):
    """Tests balanced binary dataset diagnostics."""
    df = pd.DataFrame({"x": range(100), "label": ["A"] * 50 + ["B"] * 50})
    upload_folder, disk_name = make_csv_file(df)
    data_file = DataFile(file_name="b.csv", file_type="csv", disk_name=disk_name, columns=["x", "label"], row_count=100)
    db_session.add(data_file)
    db_session.commit()

    with patch("app.services.class_imbalance_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(upload_folder)
        resp = client.get(f"/api/v1/data/process/imbalance/{data_file.id}?target_field=label")
        assert resp.status_code == 200
        data = resp.json()
        assert data["imbalance_ratio"] == 1.0
        assert data["severity"] == "balanced"
        assert data["is_imbalanced"] is False
        assert data["normalized_entropy"] == 1.0


def test_class_imbalance_severe_ratio_and_weights(client: TestClient, db_session: Session, make_csv_file):
    """Tests severe class imbalance (95/5) and balanced weight computation."""
    df = pd.DataFrame({"x": range(100), "target": [0] * 95 + [1] * 5})
    upload_folder, disk_name = make_csv_file(df)
    data_file = DataFile(
        file_name="i.csv", file_type="csv", disk_name=disk_name, columns=["x", "target"], row_count=100
    )
    db_session.add(data_file)
    db_session.commit()

    with patch("app.services.class_imbalance_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(upload_folder)
        resp = client.get(f"/api/v1/data/process/imbalance/{data_file.id}?target_field=target")
        assert resp.status_code == 200
        data = resp.json()
        assert data["imbalance_ratio"] == 19.0
        assert data["severity"] == "severe"
        assert data["is_imbalanced"] is True
        dist = {str(item["class_name"]): item for item in data["class_distribution"]}
        assert dist["1"]["balanced_weight"] == 10.0
        assert round(dist["0"]["balanced_weight"], 2) == 0.53


def test_class_imbalance_auto_detect_target(client: TestClient, db_session: Session, make_csv_file):
    """Tests that omitting target_field automatically resolves from DataProcess setting."""
    df = pd.DataFrame({"feat": [1, 2, 3, 4], "category": ["yes", "yes", "no", "yes"]})
    upload_folder, disk_name = make_csv_file(df)
    data_file = DataFile(
        file_name="auto.csv", file_type="csv", disk_name=disk_name, columns=["feat", "category"], row_count=4
    )
    db_session.add(data_file)
    db_session.commit()
    db_session.add(DataProcess(file_id=data_file.id, target="category"))
    db_session.commit()

    with patch("app.services.class_imbalance_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(upload_folder)
        resp = client.get(f"/api/v1/data/process/imbalance/{data_file.id}")
        assert resp.status_code == 200
        assert resp.json()["target_field"] == "category"
        assert resp.json()["n_classes"] == 2


def test_class_imbalance_validations_and_errors(client: TestClient, db_session: Session, make_csv_file):
    """Tests error responses for missing files, invalid targets, and single-class distributions."""
    df = pd.DataFrame({"feat": [1, 2, 3], "single_val": ["A", "A", "A"]})
    upload_folder, disk_name = make_csv_file(df)
    data_file = DataFile(
        file_name="s.csv", file_type="csv", disk_name=disk_name, columns=["feat", "single_val"], row_count=3
    )
    db_session.add(data_file)
    db_session.commit()

    with patch("app.services.class_imbalance_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(upload_folder)
        assert client.get(f"/api/v1/data/process/imbalance/{uuid4()}?target_field=feat").status_code == 404

        resp_missing = client.get(f"/api/v1/data/process/imbalance/{data_file.id}?target_field=ghost")
        assert resp_missing.status_code == 400
        assert "not found in dataset" in (resp_missing.json().get("message") or resp_missing.json().get("detail", ""))

        resp_single = client.get(f"/api/v1/data/process/imbalance/{data_file.id}?target_field=single_val")
        assert resp_single.status_code == 400
        assert "at least 2 distinct classes" in (
            resp_single.json().get("message") or resp_single.json().get("detail", "")
        )
