import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.services.deep_learning import get_code_service, run_code_service
from app.services.model_run import model_run
from app.shared.enums import ProblemType


def _make_model_config(problem_type: int = ProblemType.CLASSIFICATION) -> MagicMock:
    cfg = MagicMock()
    cfg.model_type = problem_type
    cfg.target_field = "label"
    cfg.file_id = "fake-file-uuid"
    cfg.training_split = 80
    cfg.optimizer = "adam"
    cfg.metric = "accuracy"
    cfg.epochs = 1
    cfg.loss = "sparse_categorical_crossentropy"
    return cfg


def _make_db(model_config: MagicMock) -> MagicMock:
    db = MagicMock()
    db.exec.return_value.first.return_value = model_config
    return db


class TestModelRunEmitsErrorOnFailure:
    def test_emits_socketio_error_when_csv_read_fails(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        db = _make_db(cfg)

        with (
            patch("app.services.model_run._model_result") as mock_emit,
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"),
            patch("app.services.model_run.pd.read_csv", side_effect=FileNotFoundError("data file missing")),
            pytest.raises(FileNotFoundError),
        ):
            model_run("my_model", db)

        error_calls = [c for c in mock_emit.call_args_list if c.args[1] == -1]
        assert len(error_calls) == 1

    def test_error_message_contains_exception_text(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        db = _make_db(cfg)

        with (
            patch("app.services.model_run._model_result") as mock_emit,
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"),
            patch("app.services.model_run.pd.read_csv", side_effect=ValueError("shape mismatch")),
            pytest.raises(ValueError),
        ):
            model_run("my_model", db)

        error_calls = [c for c in mock_emit.call_args_list if c.args[1] == -1]
        assert "shape mismatch" in error_calls[0].args[0]

    def test_exception_is_reraised_after_emit(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        db = _make_db(cfg)

        with (
            patch("app.services.model_run._model_result"),
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"),
            patch("app.services.model_run.pd.read_csv", side_effect=RuntimeError("OOM")),
            pytest.raises(RuntimeError, match="OOM"),
        ):
            model_run("my_model", db)


class TestRunCodeServiceOnFailure:
    def test_returns_http_400_when_model_run_raises(self):
        db = MagicMock()
        with patch("app.services.deep_learning.model_run", side_effect=RuntimeError("bad config")):
            _, status_code = run_code_service(db, "my_model")

        assert status_code == 400

    def test_response_body_has_success_false(self):
        db = MagicMock()
        with patch("app.services.deep_learning.model_run", side_effect=RuntimeError("bad config")):
            result, _ = run_code_service(db, "my_model")

        assert result["success"] is False


class TestGetCodeServiceOnFailure:
    def test_returns_http_400_when_generate_code_raises_value_error(self):
        db = MagicMock()
        with patch("app.services.deep_learning.generate_code", side_effect=ValueError("file missing")):
            _, status_code = get_code_service(db, "my_model")

        assert status_code == 400

    def test_response_body_has_error_message_from_value_error(self):
        db = MagicMock()
        with patch("app.services.deep_learning.generate_code", side_effect=ValueError("file missing")):
            result, _ = get_code_service(db, "my_model")

        assert result["success"] is False
        assert result["message"] == "file missing"


class TestModelNameScoping:
    def test_get_code_requires_project_id_when_name_is_ambiguous(self):
        m1 = MagicMock()
        m1.id = 1
        m2 = MagicMock()
        m2.id = 2
        db = MagicMock()
        db.exec.return_value.all.return_value = [m1, m2]

        result, status_code = get_code_service(db, "shared_name")

        assert status_code == 409
        assert result["success"] is False
        assert "project_id" in result["message"]

    def test_get_code_uses_project_scoped_model_id(self):
        project_id = uuid.uuid4()
        model = MagicMock()
        model.id = 42
        db = MagicMock()
        db.exec.return_value.first.return_value = model

        with patch("app.services.deep_learning.generate_code", return_value="# code") as mock_generate:
            result, status_code = get_code_service(db, "shared_name", project_id=project_id)

        assert status_code == 200
        assert result["file_name"] == "shared_name.py"
        mock_generate.assert_called_once_with("shared_name", db, model_id=42)

    def test_get_code_returns_400_when_generation_raises_value_error(self):
        model = MagicMock()
        model.id = 9
        db = MagicMock()
        db.exec.return_value.all.return_value = [model]

        with patch("app.services.deep_learning.generate_code", side_effect=ValueError("file missing")):
            result, status_code = get_code_service(db, "shared_name")

        assert status_code == 400
        assert result["success"] is False
        assert result["message"] == "file missing"

    def test_run_code_requires_project_id_when_name_is_ambiguous(self):
        m1 = MagicMock()
        m1.id = 1
        m2 = MagicMock()
        m2.id = 2
        db = MagicMock()
        db.exec.return_value.all.return_value = [m1, m2]

        result, status_code = run_code_service(db, "shared_name")

        assert status_code == 409
        assert result["success"] is False
        assert "project_id" in result["message"]
