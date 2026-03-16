import os

def test_dataset_upload(client):
    data = {
        "file": (
            open("tests/fixtures/sample_dataset.csv", "rb"),
            "sample_dataset.csv"
        )
    }

    response = client.post("/upload_dataset", data=data)

    assert response.status_code == 200
