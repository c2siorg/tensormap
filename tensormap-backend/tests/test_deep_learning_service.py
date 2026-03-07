"""Unit tests for the deep_learning service — delete_model_service.

All database interactions use MagicMock — no running DB required.
File-system interactions are patched via unittest.mock.patch.
TensorFlow is stubbed out via sys.modules so the tests run without it installed.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy third-party modules that deep_learning.py imports at module level
# so the tests can run without TensorFlow / flatten_json installed.
_tf_stub = MagicMock()
sys.modules.setdefault("tensorflow", _tf_stub)
sys.modules.setdefault("flatten_json", MagicMock())

from app.models.ml import ModelBasic  # noqa: E402
from app.services.deep_learning import delete_model_service  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db():
    return MagicMock()


@pytest.fixture()
def sample_model():
    m = MagicMock(spec=ModelBasic)
    m.id = 1
    m.model_name = "my_model"
    return m


# ---------------------------------------------------------------------------
# delete_model_service
# ---------------------------------------------------------------------------


class TestDeleteModelService:
    def test_success_deletes_model_and_file(self, mock_db, sample_model, tmp_path):
        """Deleting an existing model removes the DB row and the JSON file."""
        mock_db.get.return_value = sample_model

        model_file = tmp_path / "my_model.json"
        model_file.write_text("{}")

        with (
            patch("app.services.deep_learning.MODEL_GENERATION_LOCATION", str(tmp_path) + "/"),
            patch("app.services.deep_learning.MODEL_GENERATION_TYPE", ".json"),
        ):
            body, status_code = delete_model_service(mock_db, model_id=1)

        assert status_code == 200
        assert body["success"] is True
        mock_db.delete.assert_called_once_with(sample_model)
        mock_db.commit.assert_called_once()
        assert not model_file.exists()

    def test_model_not_found_returns_404(self, mock_db):
        """Returns 404 when no model with the given ID exists."""
        mock_db.get.return_value = None

        body, status_code = delete_model_service(mock_db, model_id=999)

        assert status_code == 404
        assert body["success"] is False
        assert "not found" in body["message"].lower()
        mock_db.delete.assert_not_called()

    def test_success_when_json_file_missing(self, mock_db, sample_model, tmp_path):
        """Deletion succeeds even when the model JSON file is already absent."""
        mock_db.get.return_value = sample_model

        # Do NOT create the file — it is intentionally absent

        with (
            patch("app.services.deep_learning.MODEL_GENERATION_LOCATION", str(tmp_path) + "/"),
            patch("app.services.deep_learning.MODEL_GENERATION_TYPE", ".json"),
        ):
            body, status_code = delete_model_service(mock_db, model_id=1)

        assert status_code == 200
        assert body["success"] is True
        mock_db.delete.assert_called_once_with(sample_model)

    def test_db_error_rolls_back(self, mock_db, sample_model):
        """A database exception triggers a rollback and returns 500."""
        mock_db.get.return_value = sample_model
        mock_db.commit.side_effect = Exception("DB failure")

        body, status_code = delete_model_service(mock_db, model_id=1)

        assert status_code == 500
        assert body["success"] is False
        mock_db.rollback.assert_called_once()

    def test_get_called_with_correct_id(self, mock_db, sample_model, tmp_path):
        """db.get is called with ModelBasic and the provided model_id."""
        mock_db.get.return_value = sample_model

        with (
            patch("app.services.deep_learning.MODEL_GENERATION_LOCATION", str(tmp_path) + "/"),
            patch("app.services.deep_learning.MODEL_GENERATION_TYPE", ".json"),
        ):
            delete_model_service(mock_db, model_id=42)

        mock_db.get.assert_called_once_with(ModelBasic, 42)
