import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models.data import DataFile, DataProcess
from app.models.ml import ModelBasic


@pytest.fixture
def sample_classification_csv(tmp_path: Path):
    """Creates a synthetic classification dataset with clear informative, weak, and constant features."""
    np.random.seed(42)
    n = 100
    # strong_feat determines target
    strong_feat = np.random.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-2.5 * strong_feat))
    target = (prob > 0.5).astype(int)

    # moderate_feat has weak correlation
    moderate_feat = strong_feat * 0.4 + np.random.normal(0, 1, n) * 0.8
    # noise_feat is pure random noise
    noise_feat = np.random.uniform(10, 20, n)
    # constant_feat has zero variance
    constant_feat = np.full(n, 5.0)

    df = pd.DataFrame(
        {
            "strong_feat": strong_feat,
            "moderate_feat": moderate_feat,
            "noise_feat": noise_feat,
            "constant_feat": constant_feat,
            "category_label": target,
        }
    )

    upload_folder = tmp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)
    file_name = f"dataset_{uuid.uuid4().hex[:8]}.csv"
    file_path = upload_folder / file_name
    df.to_csv(file_path, index=False)

    return {
        "df": df,
        "file_name": file_name,
        "file_path": file_path,
        "upload_folder": upload_folder,
    }


@pytest.fixture
def sample_regression_csv(tmp_path: Path):
    """Creates a synthetic regression dataset."""
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(1.0, 10.0, n)
    x2 = np.random.uniform(5.0, 15.0, n)
    # y strongly depends on x1, mildly on x2
    y = 4.0 * x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)
    const_col = np.zeros(n)

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "const_col": const_col,
            "y": y,
        }
    )

    upload_folder = tmp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)
    file_name = f"regression_{uuid.uuid4().hex[:8]}.csv"
    file_path = upload_folder / file_name
    df.to_csv(file_path, index=False)

    return {
        "df": df,
        "file_name": file_name,
        "file_path": file_path,
        "upload_folder": upload_folder,
    }


def _seed_file(
    db: Session,
    file_name: str,
    disk_name: str,
    columns: list[str],
    file_type: str = "csv",
    row_count: int = 100,
) -> DataFile:
    file = DataFile(
        file_name=file_name,
        disk_name=disk_name,
        file_type=file_type,
        row_count=row_count,
        columns=columns,
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def test_analyze_classification_feature_relevance(client: TestClient, db_session: Session, sample_classification_csv):
    """Tests feature relevance analysis on a binary classification dataset."""
    data_file = _seed_file(
        db_session,
        file_name="classification.csv",
        disk_name=sample_classification_csv["file_name"],
        columns=list(sample_classification_csv["df"].columns),
    )

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(sample_classification_csv["upload_folder"])

        payload = {"target_field": "category_label", "problem_type": "classification"}
        response = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/analyze",
            json=payload,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["target_field"] == "category_label"
        assert body["problem_type"] == "classification"
        assert body["total_samples"] == 100
        assert body["total_features"] == 4

        features = body["features"]
        assert len(features) == 4

        # Verify sorted descending by normalized_score
        scores = [f["normalized_score"] for f in features]
        assert scores == sorted(scores, reverse=True)

        # strong_feat should be ranked first
        top_feature = features[0]
        assert top_feature["feature"] == "strong_feat"
        assert top_feature["mutual_info"] > 0.0
        assert top_feature["tier"] in ("CRITICAL", "INFORMATIVE")
        assert top_feature["suggested_action"] == "keep"

        # constant_feat should be flagged as constant and recommended to prune
        const_item = next(f for f in features if f["feature"] == "constant_feat")
        assert const_item["is_constant"] is True
        assert const_item["normalized_score"] == 0.0
        assert const_item["tier"] == "CONSTANT_OR_REDUNDANT"
        assert const_item["suggested_action"] == "prune"


def test_analyze_regression_feature_relevance(client: TestClient, db_session: Session, sample_regression_csv):
    """Tests feature relevance analysis on a continuous regression dataset."""
    data_file = _seed_file(
        db_session,
        file_name="regression.csv",
        disk_name=sample_regression_csv["file_name"],
        columns=list(sample_regression_csv["df"].columns),
    )

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(sample_regression_csv["upload_folder"])

        payload = {"target_field": "y"}
        response = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/analyze",
            json=payload,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["target_field"] == "y"
        assert body["problem_type"] == "regression"

        features = body["features"]
        x1_item = next(f for f in features if f["feature"] == "x1")
        x2_item = next(f for f in features if f["feature"] == "x2")

        # x1 is far more predictive than x2
        assert x1_item["normalized_score"] > x2_item["normalized_score"]
        assert x1_item["mutual_info"] > 0.0
        assert x1_item["f_statistic"] is not None


def test_analyze_auto_detects_target_from_dataprocess(
    client: TestClient, db_session: Session, sample_classification_csv
):
    """Verifies that if target_field is omitted in request, it is auto-detected from DataProcess."""
    data_file = _seed_file(
        db_session,
        file_name="classification.csv",
        disk_name=sample_classification_csv["file_name"],
        columns=list(sample_classification_csv["df"].columns),
    )

    # Set active target in DataProcess
    dp = DataProcess(file_id=data_file.id, target="category_label")
    db_session.add(dp)
    db_session.commit()

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(sample_classification_csv["upload_folder"])

        # Omit target_field
        response = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/analyze",
            json={},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["target_field"] == "category_label"


def test_analyze_auto_detects_target_from_model_basic(
    client: TestClient, db_session: Session, sample_classification_csv
):
    """Verifies fallback target detection from ModelBasic when DataProcess has no target."""
    data_file = _seed_file(
        db_session,
        file_name="classification.csv",
        disk_name=sample_classification_csv["file_name"],
        columns=list(sample_classification_csv["df"].columns),
    )

    mb = ModelBasic(
        model_name="test_model",
        file_id=data_file.id,
        target_field="category_label",
    )
    db_session.add(mb)
    db_session.commit()

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(sample_classification_csv["upload_folder"])

        response = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/analyze",
            json={},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["target_field"] == "category_label"


def test_analyze_fails_when_no_target_configured(client: TestClient, db_session: Session, sample_classification_csv):
    """Verifies HTTP 400 when no target is provided in request or database."""
    data_file = _seed_file(
        db_session,
        file_name="classification.csv",
        disk_name=sample_classification_csv["file_name"],
        columns=list(sample_classification_csv["df"].columns),
    )

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(sample_classification_csv["upload_folder"])

        response = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/analyze",
            json={},
        )

        assert response.status_code == 400
        data = response.json()
        err_msg = data.get("message") or data.get("detail", "")
        assert "No target field specified" in err_msg


def test_analyze_rejects_non_existent_or_image_dataset(client: TestClient, db_session: Session):
    """Verifies HTTP 404 for missing file and HTTP 400 for image datasets."""
    # 404 for unknown file_id
    unknown_id = uuid.uuid4()
    resp_404 = client.post(f"/api/v1/data/process/feature-selection/{unknown_id}/analyze", json={})
    assert resp_404.status_code == 404

    # 400 for image dataset
    img_file = _seed_file(
        db_session,
        file_name="images.zip",
        disk_name="images.zip",
        columns=[],
        file_type="zip",
    )
    resp_img = client.post(f"/api/v1/data/process/feature-selection/{img_file.id}/analyze", json={})
    assert resp_img.status_code == 400
    data = resp_img.json()
    err_msg = data.get("message") or data.get("detail", "")
    assert "tabular CSV" in err_msg


def test_analyze_rejects_single_class_target(client: TestClient, db_session: Session, tmp_path: Path):
    """Verifies that classification targets with only 1 class are rejected with HTTP 400."""
    df = pd.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0, 0, 0, 0, 0],
        }
    )
    csv_path = tmp_path / "single_class.csv"
    df.to_csv(csv_path, index=False)

    data_file = _seed_file(
        db_session,
        file_name="single_class.csv",
        disk_name="single_class.csv",
        columns=["feat1", "target"],
    )

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(tmp_path)

        response = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/analyze",
            json={"target_field": "target", "problem_type": "classification"},
        )
        assert response.status_code == 400
        data = response.json()
        err_msg = data.get("message") or data.get("detail", "")
        assert "must contain at least 2 distinct classes" in err_msg


def test_apply_feature_selection_explicit_list(client: TestClient, db_session: Session, sample_classification_csv):
    """Tests pruning a dataset with explicit selected_features list."""
    data_file = _seed_file(
        db_session,
        file_name="classification.csv",
        disk_name=sample_classification_csv["file_name"],
        columns=list(sample_classification_csv["df"].columns),
    )

    dp = DataProcess(file_id=data_file.id, target="category_label")
    db_session.add(dp)
    db_session.commit()

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(sample_classification_csv["upload_folder"])

        payload = {"selected_features": ["strong_feat", "moderate_feat"]}
        response = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/apply",
            json=payload,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["retained_features"] == ["strong_feat", "moderate_feat"]
        assert set(body["dropped_features"]) == {"noise_feat", "constant_feat"}
        assert body["total_columns_remaining"] == 3  # 2 features + 1 target
        assert body["backup_created"] is True

        # Verify CSV content on disk
        df_on_disk = pd.read_csv(sample_classification_csv["file_path"])
        assert list(df_on_disk.columns) == ["strong_feat", "moderate_feat", "category_label"]

        # Verify backup was created
        backup_path = sample_classification_csv["file_path"].with_suffix(".csv.bak")
        assert backup_path.exists()

        # Verify DB DataFile updated
        db_session.refresh(data_file)
        assert data_file.columns == ["strong_feat", "moderate_feat", "category_label"]
        assert len(data_file.columns) == 3


def test_apply_feature_selection_top_k(client: TestClient, db_session: Session, sample_classification_csv):
    """Tests pruning a dataset using automatic top_k ranking."""
    data_file = _seed_file(
        db_session,
        file_name="classification.csv",
        disk_name=sample_classification_csv["file_name"],
        columns=list(sample_classification_csv["df"].columns),
    )

    dp = DataProcess(file_id=data_file.id, target="category_label")
    db_session.add(dp)
    db_session.commit()

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(sample_classification_csv["upload_folder"])

        payload = {"top_k": 2}
        response = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/apply",
            json=payload,
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["retained_features"]) == 2
        assert body["retained_features"][0] == "strong_feat"
        assert body["total_columns_remaining"] == 3


def test_apply_feature_selection_min_score(client: TestClient, db_session: Session, sample_classification_csv):
    """Tests pruning a dataset by threshold score."""
    data_file = _seed_file(
        db_session,
        file_name="classification.csv",
        disk_name=sample_classification_csv["file_name"],
        columns=list(sample_classification_csv["df"].columns),
    )

    dp = DataProcess(file_id=data_file.id, target="category_label")
    db_session.add(dp)
    db_session.commit()

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(sample_classification_csv["upload_folder"])

        payload = {"min_score": 40.0}
        response = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/apply",
            json=payload,
        )

        assert response.status_code == 200
        body = response.json()
        assert "strong_feat" in body["retained_features"]
        assert "constant_feat" in body["dropped_features"]


def test_apply_feature_selection_validations(client: TestClient, db_session: Session, sample_classification_csv):
    """Tests validation errors during feature selection application."""
    data_file = _seed_file(
        db_session,
        file_name="classification.csv",
        disk_name=sample_classification_csv["file_name"],
        columns=list(sample_classification_csv["df"].columns),
    )

    dp = DataProcess(file_id=data_file.id, target="category_label")
    db_session.add(dp)
    db_session.commit()

    with patch("app.services.feature_selection_service.get_settings") as mock_settings:
        mock_settings.return_value.upload_folder = str(sample_classification_csv["upload_folder"])

        # 1. Unknown feature specified
        resp_unknown = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/apply",
            json={"selected_features": ["ghost_column"]},
        )
        assert resp_unknown.status_code == 400
        data = resp_unknown.json()
        err_msg = data.get("message") or data.get("detail", "")
        assert "Unknown feature" in err_msg

        # 2. Empty selection specified
        resp_empty = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/apply",
            json={"selected_features": []},
        )
        assert resp_empty.status_code == 400
        data = resp_empty.json()
        err_msg = data.get("message") or data.get("detail", "")
        assert "At least 1 candidate" in err_msg

        # 3. No selection criteria provided
        resp_none = client.post(
            f"/api/v1/data/process/feature-selection/{data_file.id}/apply",
            json={},
        )
        assert resp_none.status_code == 400
        data = resp_none.json()
        err_msg = data.get("message") or data.get("detail", "")
        assert "Must specify" in err_msg
