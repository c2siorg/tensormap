"""Integration tests for the dataset upload pipeline."""

import io


def test_upload_rejects_non_csv(client):
    """Uploading a .txt file must be rejected with a 400 error."""
    data = {"data": ("test.txt", io.BytesIO(b"not a csv"), "text/plain")}
    response = client.post("/api/v1/data/upload/file", files=data)
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert "message" in body


def test_upload_rejects_missing_filename(client):
    """Upload with no filename should be rejected."""
    data = {"data": ("", io.BytesIO(b""), "text/csv")}
    response = client.post("/api/v1/data/upload/file", files=data)
    assert response.status_code in (400, 422)


def test_file_list_returns_success(client):
    """File list endpoint should return a successful response."""
    response = client.get("/api/v1/data/upload/file")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "data" in body
