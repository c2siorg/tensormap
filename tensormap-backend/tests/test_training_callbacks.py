"""Unit tests for adaptive training callbacks and hyperparameter integration."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.callbacks.cancellation_callback import CancellationCheckCallback
from app.callbacks.early_stopping import (
    EarlyStoppingNotificationCallback,
    ReduceLROnPlateauNotificationCallback,
)
from app.callbacks.metrics_callback import MetricsCallback
from app.schemas.deep_learning import TrainingConfigRequest
from app.schemas.training import TrainingStartRequest
from app.services.model_run import (
    CustomProgressBar,
    _build_training_callbacks,
    _should_skip_evaluation,
)
from app.shared.enums import LossFunction, Metric, Optimizer


class TestEarlyStoppingNotificationCallback:
    def test_emits_early_stopping_event_when_stopped(self):
        sio = MagicMock()
        loop = MagicMock()
        cb = EarlyStoppingNotificationCallback("job-1", sio, loop, monitor="val_loss", patience=3)
        cb.stopped_epoch = 7

        with patch("app.callbacks.early_stopping.schedule_room_emit") as mock_emit:
            cb.on_train_end()
            mock_emit.assert_called_once()
            args = mock_emit.call_args[0]
            assert args[0] == sio
            assert args[1] == loop
            assert args[2] == "job-1"
            event = args[3]
            assert event["type"] == "early_stopping"
            assert event["job_id"] == "job-1"
            assert event["stopped_epoch"] == 8
            assert event["best_epoch"] == 5
            assert event["monitor"] == "val_loss"

    def test_does_not_emit_when_not_stopped_early(self):
        sio = MagicMock()
        loop = MagicMock()
        cb = EarlyStoppingNotificationCallback("job-1", sio, loop, monitor="val_loss", patience=3)
        cb.stopped_epoch = 0

        with patch("app.callbacks.early_stopping.schedule_room_emit") as mock_emit:
            cb.on_train_end()
            mock_emit.assert_not_called()


class TestReduceLROnPlateauNotificationCallback:
    def test_emits_learning_rate_reduced_event_when_lr_drops(self):
        sio = MagicMock()
        loop = MagicMock()
        cb = ReduceLROnPlateauNotificationCallback("job-1", sio, loop, monitor="val_loss", factor=0.5, patience=2)

        # Mock model optimizer learning rate transition from 0.01 to 0.005
        model = MagicMock()
        opt = MagicMock()
        lr_mock = MagicMock()
        lr_mock.numpy.side_effect = [0.01, 0.005]
        opt.learning_rate = lr_mock
        model.optimizer = opt
        cb.set_model(model)

        with (
            patch("tensorflow.keras.callbacks.ReduceLROnPlateau.on_epoch_end"),
            patch("app.callbacks.early_stopping.schedule_room_emit") as mock_emit,
        ):
            cb.on_epoch_end(2, logs={"val_loss": 0.5})
            mock_emit.assert_called_once()
            args = mock_emit.call_args[0]
            event = args[3]
            assert event["type"] == "learning_rate_reduced"
            assert event["job_id"] == "job-1"
            assert event["epoch"] == 3
            assert event["old_lr"] == 0.01
            assert event["new_lr"] == 0.005

    def test_does_not_emit_when_lr_unchanged(self):
        sio = MagicMock()
        loop = MagicMock()
        cb = ReduceLROnPlateauNotificationCallback("job-1", sio, loop, monitor="val_loss", factor=0.5, patience=2)

        model = MagicMock()
        opt = MagicMock()
        lr_mock = MagicMock()
        lr_mock.numpy.side_effect = [0.01, 0.01]
        opt.learning_rate = lr_mock
        model.optimizer = opt
        cb.set_model(model)

        with (
            patch("tensorflow.keras.callbacks.ReduceLROnPlateau.on_epoch_end"),
            patch("app.callbacks.early_stopping.schedule_room_emit") as mock_emit,
        ):
            cb.on_epoch_end(0, logs={"val_loss": 0.5})
            mock_emit.assert_not_called()


class TestBuildTrainingCallbacks:
    def test_legacy_mode_without_job_id_returns_custom_progress_bar(self):
        callbacks = _build_training_callbacks(job_id=None)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], CustomProgressBar)

    def test_default_callbacks_with_job_id(self):
        with patch("app.services.training_service.make_session"):
            callbacks = _build_training_callbacks(job_id="job-123")
            assert len(callbacks) == 2
            assert any(isinstance(c, MetricsCallback) for c in callbacks)
            assert any(isinstance(c, CancellationCheckCallback) for c in callbacks)

    def test_early_stopping_callback_inclusion_and_monitoring(self):
        hyperparams = {
            "early_stopping": True,
            "early_stopping_patience": 7,
            "early_stopping_min_delta": 0.005,
            "early_stopping_restore_best_weights": True,
        }
        with patch("app.services.training_service.make_session"):
            callbacks_with_val = _build_training_callbacks("job-123", hyperparams=hyperparams, has_validation=True)
            es_val = next(c for c in callbacks_with_val if isinstance(c, EarlyStoppingNotificationCallback))
            assert es_val.monitor == "val_loss"
            assert es_val.patience == 7
            assert es_val.min_delta == 0.005
            assert es_val.restore_best_weights is True

            callbacks_no_val = _build_training_callbacks("job-123", hyperparams=hyperparams, has_validation=False)
            es_no_val = next(c for c in callbacks_no_val if isinstance(c, EarlyStoppingNotificationCallback))
            assert es_no_val.monitor == "loss"

    def test_reduce_lr_callback_inclusion_and_monitoring(self):
        hyperparams = {
            "reduce_lr_on_plateau": True,
            "reduce_lr_patience": 4,
            "reduce_lr_factor": 0.3,
            "reduce_lr_min_lr": 1e-5,
        }
        with patch("app.services.training_service.make_session"):
            callbacks = _build_training_callbacks("job-123", hyperparams=hyperparams, has_validation=True)
            lr_cb = next(c for c in callbacks if isinstance(c, ReduceLROnPlateauNotificationCallback))
            assert lr_cb.monitor == "val_loss"
            assert lr_cb.patience == 4
            assert pytest.approx(lr_cb.factor) == 0.3
            assert pytest.approx(lr_cb.min_lr) == 1e-5

    def test_both_adaptive_callbacks_included(self):
        hyperparams = {
            "early_stopping": True,
            "reduce_lr_on_plateau": True,
        }
        with patch("app.services.training_service.make_session"):
            callbacks = _build_training_callbacks("job-123", hyperparams=hyperparams)
            assert len(callbacks) == 4
            assert any(isinstance(c, EarlyStoppingNotificationCallback) for c in callbacks)
            assert any(isinstance(c, ReduceLROnPlateauNotificationCallback) for c in callbacks)


class TestShouldSkipEvaluation:
    def test_skips_when_cancellation_callback_is_cancelled(self):
        cb = CancellationCheckCallback("job-123", MagicMock())
        cb.cancelled = True
        model = MagicMock()
        model.stop_training = True

        assert _should_skip_evaluation([cb], model) is True

    def test_does_not_skip_when_early_stopping_stopped_training(self):
        cb = CancellationCheckCallback("job-123", MagicMock())
        cb.cancelled = False
        cb.check_cancelled = MagicMock(return_value=False)

        es_cb = EarlyStoppingNotificationCallback("job-123", MagicMock(), MagicMock(), monitor="val_loss", patience=3)
        es_cb.stopped_epoch = 3

        model = MagicMock()
        # EarlyStopping sets model.stop_training = True, but cancellation did not occur
        model.stop_training = True

        assert _should_skip_evaluation([cb, es_cb], model) is False

    def test_does_not_skip_on_normal_completion(self):
        cb = CancellationCheckCallback("job-123", MagicMock())
        cb.cancelled = False
        cb.check_cancelled = MagicMock(return_value=False)

        model = MagicMock()
        model.stop_training = False

        assert _should_skip_evaluation([cb], model) is False


class TestCallbackSchemaValidation:
    def test_training_start_request_defaults(self):
        req = TrainingStartRequest(model_name="test_model")
        assert req.early_stopping is False
        assert req.early_stopping_patience == 5
        assert req.early_stopping_min_delta == 0.001
        assert req.early_stopping_restore_best_weights is True
        assert req.reduce_lr_on_plateau is False
        assert req.reduce_lr_patience == 3
        assert req.reduce_lr_factor == 0.2
        assert req.reduce_lr_min_lr == 1e-6

    def test_training_start_request_rejects_invalid_values(self):
        with pytest.raises(ValidationError):
            TrainingStartRequest(model_name="test_model", early_stopping_patience=0)

        with pytest.raises(ValidationError):
            TrainingStartRequest(model_name="test_model", reduce_lr_factor=1.5)

        with pytest.raises(ValidationError):
            TrainingStartRequest(model_name="test_model", reduce_lr_factor=0.0)

    def test_training_config_request_callback_support(self):
        import uuid

        req = TrainingConfigRequest(
            model_name="test_model",
            file_id=uuid.uuid4(),
            training_split=80.0,
            problem_type_id=1,
            optimizer=Optimizer.ADAM,
            metric=Metric.ACCURACY,
            epochs=10,
            loss=LossFunction.CATEGORICAL_CROSSENTROPY,
            early_stopping=True,
            early_stopping_patience=10,
            reduce_lr_on_plateau=True,
        )
        assert req.early_stopping is True
        assert req.early_stopping_patience == 10
        assert req.reduce_lr_on_plateau is True
