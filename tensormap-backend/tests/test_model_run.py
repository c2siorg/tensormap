from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
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
    def test_emits_socketio_error_when_target_field_is_missing(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        cfg.target_field = None
        db = _make_db(cfg)

        with patch("app.services.model_run._model_result") as mock_emit:
            with pytest.raises(
                ValueError,
                match="Training configuration incomplete: target field is required for tabular models",
            ):
                model_run("my_model", db)

        error_calls = [c for c in mock_emit.call_args_list if c.args[1] == -1]
        assert len(error_calls) == 1
        assert "target field is required for tabular models" in error_calls[0].args[0]

    def test_emits_socketio_error_when_target_field_is_blank(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        cfg.target_field = "   "
        db = _make_db(cfg)

        with patch("app.services.model_run._model_result") as mock_emit:
            with pytest.raises(
                ValueError,
                match="Training configuration incomplete: target field is required for tabular models",
            ):
                model_run("my_model", db)

        error_calls = [c for c in mock_emit.call_args_list if c.args[1] == -1]
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
        ):
            with pytest.raises(
                ValueError,
                match="Training configuration error: target field 'missing_label' not found in data file columns",
            ):
                model_run("my_model", db)

        error_calls = [c for c in mock_emit.call_args_list if c.args[1] == -1]
        assert len(error_calls) == 1
        assert "missing_label" in error_calls[0].args[0]
        assert "Available columns (2 total): ['feature', 'label']" in error_calls[0].args[0]

    def test_target_field_whitespace_is_normalized_before_column_lookup(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        cfg.target_field = " label "
        db = _make_db(cfg)
        features = pd.DataFrame({"feature": [1, 2], "label": [0, 1]})

        mock_model = MagicMock()

        with (
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/path"),
            patch("app.services.model_run.pd.read_csv", return_value=features),
            patch.object(pd.DataFrame, "sample", return_value=features),
            patch("app.services.model_run._helper_generate_json_model_file_location", return_value="/fake/model.json"),
            patch("builtins.open", mock_open(read_data="{}")),
            patch("app.services.model_run.tf.keras.models.model_from_json", return_value=mock_model),
        ):
            model_run("my_model", db)

        fit_args = mock_model.fit.call_args.args
        evaluate_args = mock_model.evaluate.call_args.args
        assert list(fit_args[0].columns) == ["feature"]
        assert fit_args[1].tolist() == [0]
        assert list(evaluate_args[0].columns) == ["feature"]
        assert evaluate_args[1].tolist() == [1]

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
