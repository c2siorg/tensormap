import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.models.data import DataFile, DataProcess

BASE = "/api/v1/data/upload/file"


def _add_dataset(db_session: Session, *, file_id: uuid.UUID, disk_name: str, file_type: str, file_name: str):
    record = DataFile(
        id=file_id,
        file_name=file_name,
        file_type=file_type,
        disk_name=disk_name,
    )
    db_session.add(record)
    db_session.commit()
    return record


def test_download_returns_current_csv_with_safe_filename_without_modifying_file(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch
):
    file_id = uuid.uuid4()
    disk_name = "stored_12345678.csv"
    saved_csv = b"feature,label\npreprocessed,1\n"
    stored_path = tmp_path / disk_name
    stored_path.write_bytes(saved_csv)
    _add_dataset(
        db_session,
        file_id=file_id,
        disk_name=disk_name,
        file_type="csv",
        file_name="preprocessed_report",
    )
    monkeypatch.setattr(
        "app.routers.data_upload.get_settings",
        lambda: SimpleNamespace(upload_folder=str(tmp_path)),
    )

    response = client.get(f"{BASE}/{file_id}/download")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="preprocessed_report.csv"' in response.headers["content-disposition"]
    assert response.content == saved_csv
    assert stored_path.read_bytes() == saved_csv


def test_download_returns_404_for_unknown_dataset(client: TestClient):
    response = client.get(f"{BASE}/{uuid.uuid4()}/download")

    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "message": "Dataset was not found.",
        "data": None,
    }


def test_download_returns_404_when_csv_file_is_missing(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch
):
    file_id = uuid.uuid4()
    _add_dataset(
        db_session,
        file_id=file_id,
        disk_name="missing_12345678.csv",
        file_type="csv",
        file_name="missing",
    )
    monkeypatch.setattr(
        "app.routers.data_upload.get_settings",
        lambda: SimpleNamespace(upload_folder=str(tmp_path)),
    )

    response = client.get(f"{BASE}/{file_id}/download")

    assert response.status_code == 404
    assert response.json()["success"] is False
    assert "unavailable" in response.json()["message"]
    assert str(tmp_path) not in response.text


def test_download_rejects_zip_dataset_as_csv(client: TestClient, db_session: Session, tmp_path: Path, monkeypatch):
    file_id = uuid.uuid4()
    _add_dataset(
        db_session,
        file_id=file_id,
        disk_name="images_12345678.zip",
        file_type="zip",
        file_name="images",
    )
    monkeypatch.setattr(
        "app.routers.data_upload.get_settings",
        lambda: SimpleNamespace(upload_folder=str(tmp_path)),
    )

    response = client.get(f"{BASE}/{file_id}/download")

    assert response.status_code == 400
    assert response.json()["success"] is False
    assert "Only CSV" in response.json()["message"]


def test_download_does_not_allow_database_disk_name_to_escape_upload_folder(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch
):
    file_id = uuid.uuid4()
    outside_path = tmp_path.parent / "outside.csv"
    outside_path.write_text("feature,label\nsecret,1\n", encoding="utf-8")
    _add_dataset(
        db_session,
        file_id=file_id,
        disk_name="../outside.csv",
        file_type="csv",
        file_name="outside",
    )
    monkeypatch.setattr(
        "app.routers.data_upload.get_settings",
        lambda: SimpleNamespace(upload_folder=str(tmp_path)),
    )

    response = client.get(f"{BASE}/{file_id}/download")

    assert response.status_code == 404
    assert outside_path.read_text(encoding="utf-8") == "feature,label\nsecret,1\n"


def test_delete_file_endpoint_cascades_data_process(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch
):
    """Deleting a file via DELETE /api/v1/data/upload/file/{id} removes DataProcess target record without FK error."""
    file_id = uuid.uuid4()
    disk_name = f"{file_id}.csv"
    file_path = tmp_path / disk_name
    file_path.write_text("feature,target\n1,0\n", encoding="utf-8")

    data_file = DataFile(
        id=file_id,
        file_name="dataset_with_target",
        file_type="csv",
        disk_name=disk_name,
        columns=["feature", "target"],
    )
    data_process = DataProcess(
        file_id=file_id,
        target="target",
    )
    db_session.add(data_file)
    db_session.add(data_process)
    db_session.commit()

    monkeypatch.setattr(
        "app.services.data_upload.get_settings",
        lambda: SimpleNamespace(upload_folder=str(tmp_path)),
    )

    response = client.delete(f"{BASE}/{file_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True

    assert db_session.exec(select(DataFile).where(DataFile.id == file_id)).first() is None
    assert db_session.exec(select(DataProcess).where(DataProcess.file_id == file_id)).first() is None
    assert not file_path.exists()
