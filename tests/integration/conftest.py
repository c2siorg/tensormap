import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app import create_app, db  # adjust import to match actual app factory

@pytest.fixture(scope="session")
def app():
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "UPLOAD_FOLDER": "/tmp/tensormap_test_uploads"
    })
    os.makedirs("/tmp/tensormap_test_uploads", exist_ok=True)
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture()
def client(app):
    return app.test_client()

@pytest.fixture()
def sample_csv():
    return open("tests/fixtures/sample_dataset.csv", "rb")

@pytest.fixture()
def simple_graph():
    import json
    with open("tests/fixtures/simple_dense_graph.json") as f:
        return json.load(f)
