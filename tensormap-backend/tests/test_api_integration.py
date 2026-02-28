import io
import uuid
from unittest.mock import MagicMock, mock_open, patch

from fastapi.testclient import TestClient

CSV_BYTES = b"col1,col2\nval1,val2\nval3,val4\n"


def _create_project(client: TestClient, name: str = "TestProject") -> dict:
    resp = client.post("/api/v1/project", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_create_project_valid(client: TestClient):
    resp = client.post("/api/v1/project", json={"name": "MyProj"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "MyProj"
    assert "id" in body["data"]


def test_create_project_missing_name(client: TestClient):
    resp = client.post("/api/v1/project", json={})
    assert resp.status_code == 422


def test_list_projects_empty(client: TestClient):
    resp = client.get("/api/v1/project")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"] == []
    assert "pagination" in body
    assert body["pagination"]["total"] == 0


def test_get_project_by_id(client: TestClient):
    created = _create_project(client, name="FetchMe")
    project_id = created["data"]["id"]

    resp = client.get(f"/api/v1/project/{project_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == project_id
    assert body["data"]["name"] == "FetchMe"


def test_get_project_not_found(client: TestClient):
    random_id = str(uuid.uuid4())
    resp = client.get(f"/api/v1/project/{random_id}")
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False


def test_delete_project(client: TestClient):
    created = _create_project(client, name="DeleteMe")
    project_id = created["data"]["id"]

    resp = client.delete(f"/api/v1/project/{project_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True

    resp2 = client.get(f"/api/v1/project/{project_id}")
    assert resp2.status_code == 404


def test_upload_valid_csv(client: TestClient):
    resp = client.post(
        "/api/v1/data/upload/file",
        files={"data": ("test.csv", io.BytesIO(CSV_BYTES), "text/csv")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True


def test_upload_non_csv_extension(client: TestClient):
    resp = client.post(
        "/api/v1/data/upload/file",
        files={"data": ("test.txt", io.BytesIO(b"some text"), "text/plain")},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False


def test_upload_no_filename(client: TestClient):
    resp = client.post(
        "/api/v1/data/upload/file",
        files={"data": ("", io.BytesIO(b""), "text/csv")},
    )
    assert resp.status_code in (400, 422)


def test_list_files_empty(client: TestClient):
    resp = client.get("/api/v1/data/upload/file")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"] == []
    assert "pagination" in body


def test_model_list_empty(client: TestClient):
    resp = client.get("/api/v1/model/model-list")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"] == []
    assert "pagination" in body


def test_validate_model_missing_body(client: TestClient):
    resp = client.post("/api/v1/model/validate", json={})
    assert resp.status_code == 422


def test_validate_model_valid(client: TestClient):
    payload = {
        "model": {
            "nodes": [{"id": "input-1", "type": "custominput", "data": {"params": {"dim-1": 4}}}],
            "edges": [],
            "model_name": "ValidModel",
        },
        "code": {
            "dataset": {
                "file_id": str(uuid.uuid4()),
                "target_field": "col1",
                "training_split": 80,
            },
            "dl_model": {
                "model_name": "ValidModel",
                "optimizer": "adam",
                "metric": "mse",
                "epochs": 10,
            },
            "problem_type_id": 1,
        },
    }
    with (
        patch("app.services.deep_learning.model_generation", return_value={}),
        patch("app.services.deep_learning.tf") as mock_tf,
        patch("app.services.deep_learning.open", mock_open()),
    ):
        mock_tf.keras.models.model_from_json.return_value = MagicMock()
        resp = client.post("/api/v1/model/validate", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
