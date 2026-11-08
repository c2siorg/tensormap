from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.models.data import DataFile
from app.models.training_job import TrainingJob
from app.services.deep_learning import get_code_service, run_code_service
from app.services.model_run import (
    ModelRunError,
    _build_optimizer,
    _fit_model,
    _helper_generate_file_location,
    _persist_reproducibility_snapshot,
    _run,
    _validate_training_params,
    model_run,
)
from app.services.reproducibility import TrainingCancelledError, image_dataset_snapshot
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
        dataset_file = MagicMock(spec=DataFile)
        dataset_file.id = "fake-file-uuid"
        dataset_file.file_name = "training"
        dataset_file.file_type = "csv"
        cfg.file_id = "fake-file-uuid"

        db = MagicMock()
        db.exec.side_effect = [
            MagicMock(first=MagicMock(return_value=cfg)),
            MagicMock(first=MagicMock(return_value=dataset_file)),
        ]
        db.get.return_value = job

        mock_model = MagicMock()
        mock_model.to_json.return_value = '{"class_name":"Sequential","config":{}}'
        with (
            patch("app.services.model_run._build_training_callbacks", return_value=[]),
            patch("app.services.model_run._raise_if_job_cancelled"),
            patch("app.services.model_run._helper_generate_file_location", return_value="/fake/data.csv"),
            patch(
                "app.services.model_run.read_csv_with_fingerprint",
                return_value=(
                    pd.DataFrame({"feature": [1, 2, 3], "label": [0, 1, 0]}),
                    "a" * 64,
                    30,
                ),
            ),
            patch("builtins.open", MagicMock()),
            patch("app.services.model_run.tf.keras.models.model_from_json", return_value=mock_model),
            patch("app.services.model_run._emits_logits", return_value=False),
            patch(
                "app.services.model_run.tf.keras.optimizers.serialize",
                return_value={"class_name": "SGD", "config": {"learning_rate": 0.02}},
            ),
            patch("app.services.model_run.tf.keras.losses.serialize", return_value="sparse_categorical_crossentropy"),
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
        assert len(mock_model.fit.call_args.args[0]) == 2
        snapshot = job.reproducibility_metadata
        assert snapshot["job_id"] == "job-override-1"
        assert snapshot["training_configuration"]["model_architecture"] is not None
        assert len(snapshot["training_configuration"]["architecture_sha256"]) == 64
        assert snapshot["training_configuration"]["epochs"] == 12
        assert snapshot["training_configuration"]["batch_size"] == 64
        assert snapshot["training_configuration"]["feature_columns"] == ["feature"]
        assert snapshot["schema_version"] == 2
        assert snapshot["training_configuration"]["requested_training_split_percent"] == 80
        assert snapshot["training_configuration"]["requested_holdout_split_percent"] == 20
        assert snapshot["training_configuration"]["actual_training_rows"] == 2
        assert snapshot["training_configuration"]["actual_holdout_rows"] == 1
        assert snapshot["training_configuration"]["actual_training_split_percent"] == pytest.approx(200 / 3)
        assert snapshot["training_configuration"]["actual_holdout_split_percent"] == pytest.approx(100 / 3)
        assert snapshot["training_configuration"]["configured_hyperparameters"]["runtime_overrides"]["epochs"] == 12
        assert snapshot["randomness"]["configured_seed"] is None
        assert snapshot["randomness"]["effective_seeds"]["tabular_dataframe_sample_random_state"] == 42
        assert snapshot["randomness"]["applied_seeds"] == ["tabular_dataframe_sample_random_state"]
        assert snapshot["randomness"]["global_framework_seed_applied"] is False
        assert snapshot["runtime_versions"]["python"]


def test_image_snapshot_records_applied_directory_seed_and_holdout_role():
    import uuid

    from app.models.ml import ModelBasic
    from app.models.training_job import TrainingStatus

    dataset_id = uuid.uuid4()
    job = TrainingJob(id="image-job", model_id=1, status=TrainingStatus.PENDING, hyperparams={})
    dataset_file = DataFile(
        id=dataset_id,
        file_name="images",
        file_type="zip",
        disk_name="images-123.zip",
    )
    model_configs = ModelBasic(
        model_name="image-model",
        file_id=dataset_id,
        model_type=ProblemType.IMAGE_CLASSIFICATION,
        target_field=None,
        training_split=80,
        optimizer="adam",
        metric="accuracy",
        epochs=2,
        batch_size=8,
        loss="sparse_categorical_crossentropy",
    )
    model = MagicMock()
    model.to_json.return_value = '{"class_name":"Sequential","config":{}}'
    model.optimizer = MagicMock()
    model.loss = MagicMock()
    db = MagicMock()

    with (
        patch("app.services.model_run.tf.keras.optimizers.serialize", return_value={"class_name": "Adam"}),
        patch("app.services.model_run.tf.keras.losses.serialize", return_value="sparse_categorical_crossentropy"),
        patch("app.services.model_run.runtime_versions", return_value={"python": "3.12.0"}),
    ):
        _persist_reproducibility_snapshot(
            db=db,
            job=job,
            dataset_file=dataset_file,
            fingerprint="b" * 64,
            fingerprint_scope="image_tree_manifest_v1",
            size_bytes=20,
            model_configs=model_configs,
            model=model,
            hyperparams={},
            optimizer_name="adam",
            learning_rate=None,
            epochs=2,
            batch_size=8,
            training_split=80,
            feature_columns=None,
            class_names=["cat", "dog"],
            dataset_parameters={"image_size": 224, "color_mode": "rgb", "label_mode": "int"},
        )

    snapshot = job.reproducibility_metadata
    assert snapshot["dataset"]["fingerprint_scope"] == "image_tree_manifest_v1"
    assert snapshot["training_configuration"]["holdout_role"] == "validation"
    assert snapshot["training_configuration"]["class_names"] == ["cat", "dog"]
    assert snapshot["randomness"]["configured_seed"] is None
    assert snapshot["randomness"]["effective_seeds"]["keras_image_dataset_seed"] == 123
    assert snapshot["randomness"]["applied_seeds"] == ["keras_image_dataset_seed"]
    db.commit.assert_called_once()


@pytest.mark.parametrize(
    "exit_path", ["success", "fit_failure", "snapshot_failure", "cooperative_cancel", "pre_fit_cancel"]
)
def test_image_staging_is_cleaned_on_fit_exit_paths(tmp_path, monkeypatch, exit_path):
    source = tmp_path / "images"
    (source / "cats").mkdir(parents=True)
    (source / "cats" / "one.png").write_bytes(b"image bytes")
    tracked_staging_dirs = []
    real_snapshot = image_dataset_snapshot

    @contextmanager
    def tracked_snapshot(*args, **kwargs):
        with real_snapshot(*args, **kwargs) as staged:
            tracked_staging_dirs.append(Path(staged["path"]).parent)
            yield staged

    monkeypatch.setattr("app.services.model_run.image_dataset_snapshot", tracked_snapshot)
    job = TrainingJob(id="cleanup-job", model_id=1, hyperparams={})
    if exit_path == "pre_fit_cancel":
        from app.models.training_job import TrainingStatus

        job.status = TrainingStatus.CANCELLED
    dataset_file = MagicMock(spec=DataFile)
    cfg = _make_model_config(ProblemType.IMAGE_CLASSIFICATION)
    cfg.loss = "sparse_categorical_crossentropy"
    model = MagicMock()
    model.to_json.return_value = '{"class_name":"Sequential","config":{}}'
    callbacks = []
    if exit_path == "cooperative_cancel":
        from app.callbacks.cancellation_callback import CancellationCheckCallback

        cancel_callback = CancellationCheckCallback("cleanup-job", MagicMock())
        cancel_callback.cancelled = True
        callbacks = [cancel_callback]
    elif exit_path == "fit_failure":
        model.fit.side_effect = RuntimeError("synthetic fit failure")

    snapshot_side_effect = RuntimeError("synthetic snapshot failure") if exit_path == "snapshot_failure" else None
    db = MagicMock()

    with (
        patch("app.services.model_run._helper_generate_json_model_file_location", return_value="model.json"),
        patch("builtins.open", MagicMock()),
        patch("app.services.model_run.tf.keras.models.model_from_json", return_value=model),
        patch("app.services.model_run._emits_logits", return_value=False),
        patch(
            "app.services.model_run._raise_if_job_cancelled",
            side_effect=(
                [TrainingCancelledError("Training job was cancelled")] if exit_path == "pre_fit_cancel" else None
            ),
        ),
        patch("app.services.model_run._persist_reproducibility_snapshot", side_effect=snapshot_side_effect),
        tracked_snapshot(str(source)) as training_snapshot,
    ):
        if exit_path == "pre_fit_cancel":
            with pytest.raises(TrainingCancelledError):
                _fit_model(
                    "model",
                    db,
                    job,
                    cfg,
                    {},
                    "adam",
                    None,
                    1,
                    32,
                    80,
                    dataset_file,
                    "a" * 64,
                    "image_tree_manifest_v1",
                    11,
                    None,
                    ["cats"],
                    {},
                    None,
                    None,
                    callbacks,
                    image_data=(object(), object()),
                )
        elif exit_path in ("fit_failure", "snapshot_failure"):
            with pytest.raises(RuntimeError):
                _fit_model(
                    "model",
                    db,
                    job,
                    cfg,
                    {},
                    "adam",
                    None,
                    1,
                    32,
                    80,
                    dataset_file,
                    "a" * 64,
                    "image_tree_manifest_v1",
                    11,
                    None,
                    ["cats"],
                    {},
                    None,
                    None,
                    callbacks,
                    image_data=(object(), object()),
                )
        else:
            _fit_model(
                "model",
                db,
                job,
                cfg,
                {},
                "adam",
                None,
                1,
                32,
                80,
                dataset_file,
                "a" * 64,
                "image_tree_manifest_v1",
                11,
                None,
                ["cats"],
                {},
                None,
                None,
                callbacks,
                image_data=(object(), object()),
            )
        assert Path(training_snapshot["path"]).is_dir()

    assert tracked_staging_dirs
    assert all(not path.exists() for path in tracked_staging_dirs)
    if exit_path == "pre_fit_cancel":
        model.fit.assert_not_called()
        assert job.status == TrainingStatus.CANCELLED
    elif exit_path == "cooperative_cancel":
        model.fit.assert_called_once()
        model.evaluate.assert_not_called()


def test_cancellation_during_image_staging_skips_fit_and_preserves_cancelled_status(tmp_path, monkeypatch):
    from app.models.training_job import TrainingStatus

    source = tmp_path / "images"
    (source / "cats").mkdir(parents=True)
    (source / "cats" / "large.png").write_bytes(b"x" * (2 * 1024 * 1024))
    job = TrainingJob(id="cancel-during-stage", model_id=1, status=TrainingStatus.PENDING, hyperparams={})
    cfg = _make_model_config(ProblemType.IMAGE_CLASSIFICATION)
    cfg.file_id = "image-file-id"
    dataset_file = MagicMock(spec=DataFile)
    db = MagicMock()
    db.exec.side_effect = [
        MagicMock(first=MagicMock(return_value=cfg)),
        MagicMock(first=MagicMock(return_value=dataset_file)),
        MagicMock(
            first=MagicMock(return_value=MagicMock(image_size=16, batch_size=2, color_mode="rgb", label_mode="int"))
        ),
    ]
    db.get.return_value = job
    fit_model = MagicMock()
    cancellation_checks = 0

    def cancel_while_copying(_job_id):
        nonlocal cancellation_checks
        cancellation_checks += 1
        if cancellation_checks >= 4:
            job.status = TrainingStatus.CANCELLED
            return True
        return False

    with (
        patch("app.services.model_run._build_training_callbacks", return_value=[]),
        patch("app.services.model_run._raise_if_job_cancelled"),
        patch("app.services.model_run._helper_generate_file_location", return_value=str(source)),
        patch("app.services.model_run._fit_model", fit_model),
        patch("app.services.model_run._job_is_cancelled", side_effect=cancel_while_copying),
        pytest.raises(TrainingCancelledError),
    ):
        _run("image-model", db, job_id=job.id)

    assert cancellation_checks >= 4
    assert job.status == TrainingStatus.CANCELLED
    fit_model.assert_not_called()


def test_cancellation_during_staging_preserves_cancelled_status_and_prevents_fit(tmp_path):
    """Verify that cancellation during image staging raises TrainingCancelledError,
    preserves CANCELLED status, and prevents model.fit() from being called.
    """
    from app.models.training_job import TrainingStatus

    source = tmp_path / "images"
    (source / "cats").mkdir(parents=True)
    (source / "dogs").mkdir()
    # Create files that will trigger multiple cancel_check iterations
    for i in range(5):
        (source / "cats" / f"cat{i}.png").write_bytes(b"cat image bytes")
        (source / "dogs" / f"dog{i}.png").write_bytes(b"dog image bytes")

    job = TrainingJob(id="cancel-staging-test", model_id=1, status=TrainingStatus.PENDING, hyperparams={})
    cfg = _make_model_config(ProblemType.IMAGE_CLASSIFICATION)
    cfg.file_id = "staging-cancel-file-id"
    dataset_file = MagicMock(spec=DataFile)
    db = MagicMock()
    db.exec.side_effect = [
        MagicMock(first=MagicMock(return_value=cfg)),
        MagicMock(first=MagicMock(return_value=dataset_file)),
        MagicMock(
            first=MagicMock(return_value=MagicMock(image_size=32, batch_size=4, color_mode="rgb", label_mode="int"))
        ),
    ]
    db.get.return_value = job

    cancel_check_count = 0

    def cancel_during_staging(_job_id):
        nonlocal cancel_check_count
        cancel_check_count += 1
        # Cancel after a few checks to simulate cancellation during copy loop
        if cancel_check_count >= 3:
            job.status = TrainingStatus.CANCELLED
            return True
        return False

    fit_model_mock = MagicMock()

    with (
        patch("app.services.model_run._build_training_callbacks", return_value=[]),
        patch("app.services.model_run._raise_if_job_cancelled"),
        patch("app.services.model_run._helper_generate_file_location", return_value=str(source)),
        patch("app.services.model_run._fit_model", fit_model_mock),
        patch("app.services.model_run._job_is_cancelled", side_effect=cancel_during_staging),
        pytest.raises(TrainingCancelledError, match="cancelled during image dataset staging"),
    ):
        _run("staging-cancel-model", db, job_id=job.id)

    # Verify cancellation was detected
    assert cancel_check_count >= 3
    # Verify status remains CANCELLED
    assert job.status == TrainingStatus.CANCELLED
    # Verify fit was never called
    fit_model_mock.assert_not_called()


def test_cancellation_before_fit_after_staging_preserves_cancelled_status(tmp_path):
    """Verify that cancellation after staging completes but before model.fit()
    is caught by the pre-fit _raise_if_job_cancelled() check.
    """
    from app.models.training_job import TrainingStatus

    source = tmp_path / "images"
    (source / "cats").mkdir(parents=True)
    # Create at least 2 images per class for TensorFlow validation_split to work
    (source / "cats" / "image1.png").write_bytes(b"image data 1")
    (source / "cats" / "image2.png").write_bytes(b"image data 2")

    job = TrainingJob(id="cancel-before-fit", model_id=1, status=TrainingStatus.PENDING, hyperparams={})
    cfg = _make_model_config(ProblemType.IMAGE_CLASSIFICATION)
    cfg.file_id = "pre-fit-cancel-file-id"
    cfg.loss = "sparse_categorical_crossentropy"
    dataset_file = MagicMock(spec=DataFile)
    dataset_file.id = "pre-fit-cancel-file-id"
    dataset_file.file_name = "images"
    dataset_file.file_type = "zip"

    db = MagicMock()
    db.exec.side_effect = [
        MagicMock(first=MagicMock(return_value=cfg)),
        MagicMock(first=MagicMock(return_value=dataset_file)),
        MagicMock(
            first=MagicMock(return_value=MagicMock(image_size=32, batch_size=4, color_mode="rgb", label_mode="int"))
        ),
    ]
    db.get.return_value = job

    # Track the flow: staging -> pre-fit cancellation
    raise_call_count = 0

    def mock_job_is_cancelled(_job_id):
        """Allow staging to complete."""
        return False

    def cancel_before_fit_check(job_id):
        nonlocal raise_call_count
        raise_call_count += 1
        # First call (before staging) passes
        if raise_call_count == 1:
            return
        # Second call (inside _fit_model before model.fit) cancels
        job.status = TrainingStatus.CANCELLED
        raise TrainingCancelledError("Training job was cancelled")

    model_mock = MagicMock()
    model_mock.to_json.return_value = '{"class_name":"Sequential","config":{}}'

    # Mock image dataset creation - it's called after staging but before _fit_model
    mock_dataset = MagicMock()

    with (
        patch("app.services.model_run._build_training_callbacks", return_value=[]),
        patch("app.services.model_run._helper_generate_file_location", return_value=str(source)),
        patch("app.services.model_run._job_is_cancelled", side_effect=mock_job_is_cancelled),
        patch("app.services.model_run._raise_if_job_cancelled", side_effect=cancel_before_fit_check),
        patch("app.services.model_run.tf.keras.utils.image_dataset_from_directory", return_value=mock_dataset),
        patch("app.services.model_run._helper_generate_json_model_file_location", return_value="model.json"),
        patch("builtins.open", MagicMock()),
        patch("app.services.model_run.tf.keras.models.model_from_json", return_value=model_mock),
        patch("app.services.model_run._emits_logits", return_value=False),
        patch("app.services.model_run._persist_reproducibility_snapshot"),
        pytest.raises(TrainingCancelledError, match="Training job was cancelled"),
    ):
        _run("pre-fit-cancel-model", db, job_id=job.id)

    # Verify the pre-fit check was called twice (once before staging, once before fit)
    assert raise_call_count == 2, f"Expected 2 cancellation checks, got {raise_call_count}"
    # Verify status is CANCELLED
    assert job.status == TrainingStatus.CANCELLED
    # Verify model.fit was never called (cancellation happened before fit)
    model_mock.fit.assert_not_called()


def test_snapshot_backward_compatible_split_fields_populated():
    """Verify that both legacy (v1) and explicit (v2) split fields are populated
    for backward compatibility with existing consumers.
    """
    import uuid

    from app.models.ml import ModelBasic
    from app.models.training_job import TrainingStatus

    dataset_id = uuid.uuid4()
    job = TrainingJob(id="split-compat-test", model_id=1, status=TrainingStatus.PENDING, hyperparams={})
    dataset_file = DataFile(
        id=dataset_id,
        file_name="training",
        file_type="csv",
        disk_name="training-abc.csv",
    )
    model_configs = ModelBasic(
        model_name="split-test-model",
        file_id=dataset_id,
        model_type=ProblemType.CLASSIFICATION,
        target_field="label",
        training_split=75,  # Requested 75% training
        optimizer="adam",
        metric="accuracy",
        epochs=5,
        batch_size=16,
        loss="sparse_categorical_crossentropy",
    )
    model = MagicMock()
    model.to_json.return_value = '{"class_name":"Sequential","config":{}}'
    model.optimizer = MagicMock()
    model.loss = MagicMock()
    db = MagicMock()

    with (
        patch("app.services.model_run.tf.keras.optimizers.serialize", return_value={"class_name": "Adam"}),
        patch("app.services.model_run.tf.keras.losses.serialize", return_value="sparse_categorical_crossentropy"),
        patch("app.services.model_run.runtime_versions", return_value={"python": "3.12.0"}),
    ):
        _persist_reproducibility_snapshot(
            db=db,
            job=job,
            dataset_file=dataset_file,
            fingerprint="a" * 64,
            fingerprint_scope="file_bytes",
            size_bytes=1000,
            model_configs=model_configs,
            model=model,
            hyperparams={},
            optimizer_name="adam",
            learning_rate=None,
            epochs=5,
            batch_size=16,
            training_split=75,
            feature_columns=["feature1", "feature2"],
            class_names=None,
            actual_training_rows=75,  # Actual split: 75 training, 25 holdout
            actual_holdout_rows=25,
            dataset_parameters=None,
        )

    snapshot = job.reproducibility_metadata
    config = snapshot["training_configuration"]

    # Verify legacy fields (schema v1) exist and match requested values
    assert config["training_split_percent"] == 75.0
    assert config["holdout_split_percent"] == 25.0

    # Verify explicit requested fields (schema v2) exist and match
    assert config["requested_training_split_percent"] == 75.0
    assert config["requested_holdout_split_percent"] == 25.0

    # Verify actual observed fields (schema v2) exist and are computed correctly
    assert config["actual_training_rows"] == 75
    assert config["actual_holdout_rows"] == 25
    assert config["actual_training_split_percent"] == 75.0  # 75/100 = 75%
    assert config["actual_holdout_split_percent"] == 25.0  # 25/100 = 25%

    # Verify schema version
    assert snapshot["schema_version"] == 2

    db.commit.assert_called_once()
