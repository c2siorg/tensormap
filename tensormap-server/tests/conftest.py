import os
from dotenv import load_dotenv
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate, upgrade
from setup.test_settings import SettingUp
from setup.urls import MainURLRegister
from werkzeug.utils import secure_filename
from shared.constants import *
import shutil
from shared.services.config import get_configs
from endpoints.DataUpload.models import DataFile

load_dotenv()
db = SQLAlchemy()
configs = get_configs()
db_name = "tensormap_test_db"


@pytest.fixture(scope="session")
def app():
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    MainURLRegister(app=flask_app)
    SettingUp(app=flask_app)

    db.init_app(flask_app)
    with flask_app.app_context():
        db.create_all()  # Create tables in the in-memory db
        yield flask_app
        db.session.remove()
        db.drop_all()  # Drop tables after tests


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def db_session(app):
    with app.app_context():
        yield db.session
        db.session.rollback()


@pytest.fixture(scope="function")  # Changed to function scope for better isolation
def add_sample_file(db_session):  # Inject the db_session
    filename = "test.csv"
    file_name = secure_filename(filename.lower())
    upload_folder = configs["api"]["upload"][
        "folder"
    ]  # Ensure this path exists or mock it
    destination_path = os.path.join(upload_folder, file_name)

    # Mock shutil.copy to avoid actual file system interaction
    with patch("shutil.copy") as mock_copy:
        mock_copy.return_value = None  # Make sure the mock behaves as expected

        file_name_db = secure_filename(filename.rsplit(".", 1)[0].lower())
        file_type_db = filename.rsplit(".", 1)[1].lower()
        data = DataFile(file_name=file_name_db, file_type=file_type_db)
        db_session.add(data)
        db_session.commit()

        yield data  # Yield the created DataFile object

    # Clean up after the test
    db_session.delete(data)
    db_session.commit()

    # Mock os.remove to avoid actual file system interaction
    with patch("os.remove") as mock_remove:
        mock_remove.return_value = None


# Example test using the fixtures:
def test_data_upload(client, add_sample_file, db_session):
    # Access the DataFile object yielded by the fixture
    data_file = add_sample_file

    # Assertions or other test logic using the database and client
    assert data_file.file_name == "test"
    assert data_file.file_type == "csv"

    # Example: Querying the database (using the in-memory SQLite)
    retrieved_file = db_session.query(DataFile).filter_by(file_name="test").first()
    assert retrieved_file is not None

    # ... Your test logic ...
