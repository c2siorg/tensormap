from unittest.mock import MagicMock, patch

import pytest

from app.services.deep_learning import run_code_service
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

        with patch("app.services.model_run._model_result") as mock_emit, \
             patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"), \
             patch("app.services.model_run.pd.read_csv", side_effect=FileNotFoundError("data file missing")):
            with pytest.raises(FileNotFoundError):
                model_run("my_model", db)

        error_calls = [c for c in mock_emit.call_args_list if c.args[1] == -1]
        assert len(error_calls) == 1

    def test_error_message_contains_exception_text(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        db = _make_db(cfg)

        with patch("app.services.model_run._model_result") as mock_emit, \
             patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"), \
             patch("app.services.model_run.pd.read_csv", side_effect=ValueError("shape mismatch")):
            with pytest.raises(ValueError):
                model_run("my_model", db)

        error_calls = [c for c in mock_emit.call_args_list if c.args[1] == -1]
        assert "shape mismatch" in error_calls[0].args[0]

    def test_exception_is_reraised_after_emit(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        db = _make_db(cfg)

        with patch("app.services.model_run._model_result"), \
             patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"), \
             patch("app.services.model_run.pd.read_csv", side_effect=RuntimeError("OOM")):
            with pytest.raises(RuntimeError, match="OOM"):
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
