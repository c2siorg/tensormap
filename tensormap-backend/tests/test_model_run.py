from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.deep_learning import run_code_service
from app.services.model_run import _run, model_run
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


def _error_result_calls(mock_emit: MagicMock) -> list:
    return [
        c
        for c in mock_emit.call_args_list
        if (len(c.args) > 1 and c.args[1] == -1)
        or c.kwargs.get("test") == -1
        or c.kwargs.get("status") == -1
    ]


class TestModelRunEmitsErrorOnFailure:
    # model_run contract: emit one error result via _model_result and then re-raise.
    def test_emits_socketio_error_when_target_field_is_missing(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        cfg.target_field = None
        db = _make_db(cfg)

        with (
            patch("app.services.model_run._model_result") as mock_emit,
            pytest.raises(
                ValueError,
                match="Training configuration incomplete: target field is required for tabular models",
            ),
        ):
            model_run("my_model", db)

        error_calls = _error_result_calls(mock_emit)
        assert len(error_calls) == 1
        assert "target field is required for tabular models" in error_calls[0].args[0]

    def test_emits_socketio_error_when_target_field_is_blank(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        cfg.target_field = "   "
        db = _make_db(cfg)

        with (
            patch("app.services.model_run._model_result") as mock_emit,
            pytest.raises(
                ValueError,
                match="Training configuration incomplete: target field is required for tabular models",
            ),
        ):
            model_run("my_model", db)

        error_calls = _error_result_calls(mock_emit)
        assert len(error_calls) == 1
        assert "target field is required for tabular models" in error_calls[0].args[0]

    def test_emits_socketio_error_when_target_field_not_in_csv_columns(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        cfg.target_field = "missing_label"
        db = _make_db(cfg)
        features = pd.DataFrame({"feature": [1, 2], "label": [0, 1]})

        with (
            patch("app.services.model_run._model_result") as mock_emit,
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"),
            patch("app.services.model_run.pd.read_csv", return_value=features),
            pytest.raises(
                ValueError,
                match="Training configuration error: target field 'missing_label' not found in data file columns",
            ),
        ):
            model_run("my_model", db)

        error_calls = _error_result_calls(mock_emit)
        assert len(error_calls) == 1
        assert "missing_label" in error_calls[0].args[0]
        assert "Please check the configured target field name." in error_calls[0].args[0]
        assert "Available columns" not in error_calls[0].args[0]

    def test_target_field_whitespace_is_normalized_before_column_lookup(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        cfg.target_field = " label "
        db = _make_db(cfg)
        features = pd.DataFrame({"feature": [1, 2], "label": [0, 1]})

        with (
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"),
            patch("app.services.model_run.pd.read_csv", return_value=features),
            patch(
                "app.services.model_run._helper_generate_json_model_file_location",
                side_effect=RuntimeError("stop after target field validation"),
            ),
            pytest.raises(RuntimeError, match="stop after target field validation"),
        ):
            _run("my_model", db)

    def test_non_string_target_field_is_preserved_for_column_lookup(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        cfg.target_field = 1
        db = _make_db(cfg)
        features = pd.DataFrame({0: [10, 20], 1: [0, 1]})

        with (
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"),
            patch("app.services.model_run.pd.read_csv", return_value=features),
            patch(
                "app.services.model_run._helper_generate_json_model_file_location",
                side_effect=RuntimeError("stop after target field validation"),
            ),
            pytest.raises(RuntimeError, match="stop after target field validation"),
        ):
            _run("my_model", db)

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

        error_calls = _error_result_calls(mock_emit)
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

        error_calls = _error_result_calls(mock_emit)
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
