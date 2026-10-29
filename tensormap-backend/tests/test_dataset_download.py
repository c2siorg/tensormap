import uuid
from types import SimpleNamespace

from app.models import DataFile


def _create_dataset(db_session, *, file_type: str = "csv", disk_name: str = "iris.csv") -> DataFile:
    data_file = DataFile(
        id=uuid.uuid4(),
        file_name="iris",
        file_type=file_type,
        disk_name=disk_name,
    )
    db_session.add(data_file)
    db_session.commit()
    return data_file


def test_download_dataset_returns_stored_csv_and_filename(client, db_session, monkeypatch, tmp_path):
    import app.routers.data_process as data_process

    original_content = b"sepal_length,species\n5.1,setosa\n"
    data_file = _create_dataset(db_session)
    source_path = tmp_path / data_file.disk_name
    source_path.write_bytes(original_content)
    monkeypatch.setattr(data_process, "get_settings", lambda: SimpleNamespace(upload_folder=str(tmp_path)))

    response = client.get(f"/api/v1/data/process/download/{data_file.id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == 'attachment; filename="iris.csv"'
    assert response.content == original_content
    assert source_path.read_bytes() == original_content


def test_download_dataset_returns_404_for_missing_dataset(client):
    response = client.get(f"/api/v1/data/process/download/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"


def test_download_dataset_returns_404_when_file_is_unavailable(client, db_session, monkeypatch, tmp_path):
    import app.routers.data_process as data_process

    data_file = _create_dataset(db_session)
    monkeypatch.setattr(data_process, "get_settings", lambda: SimpleNamespace(upload_folder=str(tmp_path)))

    response = client.get(f"/api/v1/data/process/download/{data_file.id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset file not found"


def test_download_dataset_rejects_non_csv_files(client, db_session):
    data_file = _create_dataset(db_session, file_type="zip", disk_name="images.zip")

    response = client.get(f"/api/v1/data/process/download/{data_file.id}")

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV datasets can be downloaded as CSV"


def test_download_dataset_rejects_paths_outside_upload_folder(client, db_session, monkeypatch, tmp_path):
    import app.routers.data_process as data_process

    data_file = _create_dataset(db_session, disk_name="../outside.csv")
    monkeypatch.setattr(data_process, "get_settings", lambda: SimpleNamespace(upload_folder=str(tmp_path)))

    response = client.get(f"/api/v1/data/process/download/{data_file.id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset file not found"
