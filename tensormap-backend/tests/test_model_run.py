from unittest.mock import MagicMock, patch

import pytest

from app.models.training_job import TrainingJob
from app.services.deep_learning import get_code_service, run_code_service
from app.services.model_run import (
    ModelRunError,
    _build_optimizer,
    _helper_generate_file_location,
    _run,
    _validate_training_params,
    model_run,
)
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
    cfg.batch_size = 32  # Must be a real int so _validate_training_params accepts it.
    cfg.loss = "sparse_categorical_crossentropy"
    return cfg


def _make_db(model_config: MagicMock) -> MagicMock:
    db = MagicMock()
    db.exec.return_value.first.return_value = model_config
    return db


class TestModelRunEmitsErrorOnFailure:
    def test_job_failure_emits_structured_status_envelope_to_job_room(self):
        session_context = MagicMock()
        session_context.__enter__.return_value = MagicMock()
        with (
            patch("app.services.model_run._run", side_effect=RuntimeError("training failed")),
            patch("app.services.training_service.make_session", return_value=session_context),
            patch("app.services.training_service.update_job_status") as update_status,
            patch("app.callbacks.metrics_callback.schedule_room_emit") as emit,
            pytest.raises(RuntimeError, match="training failed"),
        ):
            model_run("my_model", MagicMock(), loop=None, job_id="job-123")

        update_status.assert_called_once()
        args = emit.call_args.args
        event = args[3]
        assert args[2] == "job-123"
        assert event["version"] == 1
        assert event["type"] == "status"
        assert event["job_id"] == "job-123"
        assert event["status"] == "failed"
        assert event["error"] == "training failed"
        assert event["phase"] is None
        assert event["epoch"] is None
        assert event["batch"] is None
        assert event["steps"] is None
        assert event["timestamp"]

    def test_emits_socketio_error_when_csv_read_fails(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        db = _make_db(cfg)
        with (
            patch("app.services.model_run._model_result") as mock_emit,
            patch(
                "app.services.model_run._helper_generate_file_location",
                return_value="/fake/path",
            ),
            patch(
                "app.services.model_run.pd.read_csv",
                side_effect=FileNotFoundError("data file missing"),
            ),
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
            patch(
                "app.services.model_run._helper_generate_file_location",
                return_value="/fake/path",
            ),
            patch(
                "app.services.model_run.pd.read_csv",
                side_effect=ValueError("shape mismatch"),
            ),
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
            patch(
                "app.services.model_run._helper_generate_file_location",
                return_value="/fake/path",
            ),
            patch("app.services.model_run.pd.read_csv", side_effect=RuntimeError("OOM")),
            pytest.raises(RuntimeError, match="OOM"),
        ):
            model_run("my_model", db)


class TestRunCodeServiceOnFailure:
    def test_returns_http_400_when_model_run_raises(self):
        db = MagicMock()
        with patch(
            "app.services.deep_learning.model_run",
            side_effect=RuntimeError("bad config"),
        ):
            _, status_code = run_code_service(db, "my_model")
        assert status_code == 400

    def test_response_body_has_success_false(self):
        db = MagicMock()
        with patch(
            "app.services.deep_learning.model_run",
            side_effect=RuntimeError("bad config"),
        ):
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


class TestHelperGenerateFileLocation:
    def test_returns_path_with_disk_name(self):
        mock_file = MagicMock()
        mock_file.disk_name = "my_data_a1b2c3d4.csv"
        mock_file.file_type = "csv"
        db = MagicMock()
        db.exec.return_value.first.return_value = mock_file

        with patch("app.services.model_run.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = "/uploads"
            result = _helper_generate_file_location(db, file_id="some-uuid")

        assert result == "/uploads/my_data_a1b2c3d4.csv"

    def test_returns_path_for_zip_without_extension(self):
        mock_file = MagicMock()
        mock_file.disk_name = "images_a1b2c3d4.zip"
        mock_file.file_type = "zip"
        db = MagicMock()
        db.exec.return_value.first.return_value = mock_file

        with patch("app.services.model_run.get_settings") as mock_settings:
            mock_settings.return_value.upload_folder = "/uploads"
            result = _helper_generate_file_location(db, file_id="some-uuid")

        assert result == "/uploads/images_a1b2c3d4"

    def test_raises_error_when_file_not_found(self):
        db = MagicMock()
        db.exec.return_value.first.return_value = None

        with pytest.raises(ModelRunError, match="Dataset file not found"):
            _helper_generate_file_location(db, file_id="missing-uuid")


class TestValidateTrainingParams:
    def test_valid_parameters_pass(self):
        # Should not raise
        _validate_training_params(batch_size=32, epochs=10, training_split=80)
        _validate_training_params(batch_size=None, epochs=5, training_split=75.5)

    def test_missing_training_split_raises(self):
        with pytest.raises(ValueError, match="training_split must be specified"):
            _validate_training_params(batch_size=32, epochs=10, training_split=None)

    def test_missing_epochs_raises(self):
        with pytest.raises(ValueError, match="epochs must be a positive integer, got None"):
            _validate_training_params(batch_size=32, epochs=None, training_split=80)

    def test_invalid_epochs_raises(self):
        with pytest.raises(ValueError, match="epochs must be a positive integer, got 0"):
            _validate_training_params(batch_size=32, epochs=0, training_split=80)
        with pytest.raises(ValueError, match="epochs must be a positive integer, got -2"):
            _validate_training_params(batch_size=32, epochs=-2, training_split=80)

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size must be a positive integer, got 0"):
            _validate_training_params(batch_size=0, epochs=5, training_split=80)
        with pytest.raises(ValueError, match="batch_size must be a positive integer, got 'large'"):
            _validate_training_params(batch_size="large", epochs=5, training_split=80)

    def test_invalid_training_split_range_raises(self):
        with pytest.raises(ValueError, match="training_split must be between 0 and 100"):
            _validate_training_params(batch_size=32, epochs=5, training_split=0)
        with pytest.raises(ValueError, match="training_split must be between 0 and 100"):
            _validate_training_params(batch_size=32, epochs=5, training_split=100)
        with pytest.raises(ValueError, match="training_split must be a number"):
            _validate_training_params(batch_size=32, epochs=5, training_split="invalid")


class TestBuildOptimizer:
    def test_default_returns_string_when_no_lr(self):
        assert _build_optimizer("adam", lr=None) == "adam"
        assert _build_optimizer("SGD", lr=None) == "sgd"
        assert _build_optimizer(None, lr=None) == "adam"

    def test_non_string_optimizer_fallback(self):
        mock_opt = MagicMock()
        assert _build_optimizer(mock_opt, lr=None) is mock_opt

    def test_build_with_custom_lr(self):
        import tensorflow as tf

        opt = _build_optimizer("adam", lr=0.005)
        assert isinstance(opt, tf.keras.optimizers.Optimizer)
        assert abs(float(opt.learning_rate.numpy()) - 0.005) < 1e-6


class TestRunHyperparameterOverrides:
    def test_run_uses_job_hyperparams_overrides(self):
        cfg = _make_model_config(ProblemType.CLASSIFICATION)
        cfg.epochs = 1
        cfg.batch_size = 32
        cfg.optimizer = "adam"

        job = TrainingJob(
            id="job-override-1",
            model_id=1,
            hyperparams={
                "epochs": 12,
                "batch_size": 64,
                "optimizer": "sgd",
                "lr": 0.02,
            },
        )

        db = MagicMock()
        db.exec.return_value.first.return_value = cfg
        db.get.return_value = job

        mock_model = MagicMock()
        with (
            patch("app.services.model_run._build_training_callbacks", return_value=[]),
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/data.csv"),
            patch("app.services.model_run.pd.read_csv", return_value=MagicMock()),
            patch(
                "app.services.model_run._prepare_training_data",
                return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()),
            ),
            patch("builtins.open", MagicMock()),
            patch("app.services.model_run.tf.keras.models.model_from_json", return_value=mock_model),
            patch("app.services.model_run._emits_logits", return_value=False),
        ):
            _run("my_model", db, job_id="job-override-1")

        # Verify model.compile used sgd with lr
        mock_model.compile.assert_called_once()
        compile_kwargs = mock_model.compile.call_args.kwargs
        compiled_optimizer = compile_kwargs["optimizer"]
        assert hasattr(compiled_optimizer, "learning_rate")
        assert abs(float(compiled_optimizer.learning_rate.numpy()) - 0.02) < 1e-6

        # Verify model.fit received overridden epochs and batch_size
        mock_model.fit.assert_called_once()
        fit_kwargs = mock_model.fit.call_args.kwargs
        assert fit_kwargs["epochs"] == 12
        assert fit_kwargs["batch_size"] == 64
