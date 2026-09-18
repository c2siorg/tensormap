"""Tests for model export service (SavedModel, TFLite, ONNX)."""

import importlib.util
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from sqlmodel import Session

from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.model_export import (
    ONNXUnsupportedError,
    cleanup_exports,
    delete_job_exports,
    delete_model_exports,
    export_onnx,
    export_savedmodel,
    export_tflite,
    get_export_formats,
    validate_onnx_compatible,
)


def _check_tf2onnx_available():
    """Check if tf2onnx can be imported without errors."""
    try:
        import tf2onnx  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


@pytest.fixture
def mock_keras_model():
    """Create a mock Keras model."""
    mock_model = Mock()
    mock_model.input_shape = (None, 28, 28, 1)
    mock_model.save = Mock()
    return mock_model


@pytest.fixture
def setup_export_dir(tmp_path):
    """Create a temporary export directory with model.keras."""
    job_id = str(uuid4())
    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)

    # Create a dummy model.keras file
    model_file = export_dir / "model.keras"
    model_file.write_text("dummy model content")

    return job_id, export_dir


def test_savedmodel_export_creates_zip(tmp_path, mock_keras_model):
    """SavedModel export creates a zip file."""
    job_id = str(uuid4())
    model_name = "test_model"

    # Setup
    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    # Patch EXPORTS_BASE and tensorflow
    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        patch("tensorflow.keras.models.save_model"),
    ):
        # Create savedmodel directory for zipping
        savedmodel_dir = export_dir / "savedmodel"
        savedmodel_dir.mkdir()
        (savedmodel_dir / "saved_model.pb").write_text("dummy pb")

        zip_path = export_savedmodel(job_id, model_name)

        assert zip_path.exists()
        assert zip_path.suffix == ".zip"
        assert model_name in zip_path.name


def test_tflite_export_uses_savedmodel_path(tmp_path, mock_keras_model):
    """TFLite export converts from a SavedModel, never via from_keras_model().

    Regression guard for the TF 2.16 / Keras 3 breakage: TFLiteConverter
    .from_keras_model() always raises "TypeError: 'NoneType' object is not
    callable" for Keras 3 models.
    """
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    # Mock TFLite converter
    mock_converter = Mock()
    mock_converter.convert.return_value = b"tflite model bytes"

    # Import tensorflow to make the patch work
    import tensorflow as tf

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch.object(tf.keras.models, "load_model", return_value=mock_keras_model),
        patch.object(tf.lite.TFLiteConverter, "from_keras_model") as mock_from_keras,
        patch.object(tf.lite.TFLiteConverter, "from_saved_model", return_value=mock_converter) as mock_from_saved,
    ):
        tflite_path = export_tflite(job_id, model_name)

        # The Keras-3-incompatible entry point must not come back.
        mock_from_keras.assert_not_called()
        # The model is exported to the job's savedmodel directory first.
        mock_keras_model.export.assert_called_once_with(str(export_dir / "savedmodel"))
        mock_from_saved.assert_called_once_with(str(export_dir / "savedmodel"))

        assert tflite_path.exists()
        assert tflite_path.suffix == ".tflite"
        assert tflite_path.read_bytes() == b"tflite model bytes"


def _tflite_output(interpreter):
    """Run an already-fed TFLite interpreter and return its output tensor."""
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


@pytest.mark.slow
def test_tflite_export_produces_working_tflite(tmp_path):
    """TFLite export produces a file the TFLite interpreter can actually run.

    Previously the conversion always failed on the pinned TensorFlow 2.16 /
    Keras 3 stack ("TypeError: 'NoneType' object is not callable"), so every
    TFLite download returned HTTP 500. Exercises the real save → export →
    convert path without mocks.
    """
    import numpy as np
    import tensorflow as tf

    job_id = str(uuid4())
    model_name = "real_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    model = tf.keras.Sequential([tf.keras.layers.Input((8,)), tf.keras.layers.Dense(4), tf.keras.layers.Dense(2)])
    model.save(export_dir / "model.keras")

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        tflite_path = export_tflite(job_id, model_name)

    assert tflite_path.exists()
    assert tflite_path.suffix == ".tflite"

    interpreter = tf.lite.Interpreter(model_content=tflite_path.read_bytes())
    interpreter.allocate_tensors()
    details = interpreter.get_input_details()[0]
    assert tuple(details["shape"]) == (1, 8)

    sample = np.zeros((1, 8), dtype=np.float32)
    interpreter.set_tensor(details["index"], sample)
    got = _tflite_output(interpreter)

    reference = model(sample).numpy()
    assert got.shape == reference.shape
    assert np.allclose(reference, got, atol=1e-5)


@pytest.mark.slow
def test_tflite_export_supports_multi_input_models(tmp_path):
    """TFLite export also works for multi-input (functional) models.

    Converting from a SavedModel keeps working where the concrete-function
    recipes do not, and TFLite input ordering is not guaranteed, so the test
    matches inputs by shape.
    """
    import numpy as np
    import tensorflow as tf

    job_id = str(uuid4())
    model_name = "multi_input_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    wide = tf.keras.layers.Input((8,))
    narrow = tf.keras.layers.Input((4,))
    merged = tf.keras.layers.Concatenate()([wide, narrow])
    model = tf.keras.Model(inputs=[wide, narrow], outputs=[tf.keras.layers.Dense(2)(merged)])
    model.save(export_dir / "model.keras")

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        tflite_path = export_tflite(job_id, model_name)

    interpreter = tf.lite.Interpreter(model_content=tflite_path.read_bytes())
    interpreter.allocate_tensors()
    by_shape = {tuple(detail["shape"]): detail for detail in interpreter.get_input_details()}
    assert set(by_shape) == {(1, 8), (1, 4)}

    wide_sample = np.zeros((1, 8), dtype=np.float32)
    narrow_sample = np.zeros((1, 4), dtype=np.float32)
    interpreter.set_tensor(by_shape[(1, 8)]["index"], wide_sample)
    interpreter.set_tensor(by_shape[(1, 4)]["index"], narrow_sample)
    got = _tflite_output(interpreter)

    reference = model([wide_sample, narrow_sample]).numpy()
    assert got.shape == reference.shape
    assert np.allclose(reference, got, atol=1e-5)


def test_savedmodel_cached_on_second_call(tmp_path, mock_keras_model):
    """Second call to export_savedmodel returns existing file without regenerating."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    # Create existing zip
    zip_path = export_dir / f"{model_name}.savedmodel.zip"
    zip_path.write_text("existing zip")

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        result = export_savedmodel(job_id, model_name)

        assert result == zip_path
        assert result.read_text() == "existing zip"  # Not regenerated


def test_export_fails_404_if_no_keras_file(tmp_path):
    """Export fails if model.keras doesn't exist (training not completed)."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    # No model.keras file

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        pytest.raises(FileNotFoundError, match="model.keras not found"),
    ):
        export_savedmodel(job_id, model_name)


def test_validate_onnx_compatible_passes_for_simple_model():
    """validate_onnx_compatible returns empty list for simple Dense model."""
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [28, 28, 1]}},
            {"node_params": {"layer_type": "dense", "units": 10}},
        ]
    }

    mock_model = Mock()
    issues = validate_onnx_compatible(mock_model, graph_ir)

    assert issues == []


def test_validate_onnx_compatible_fails_for_rnn_var_length():
    """validate_onnx_compatible detects LSTM with variable-length input."""
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [None, 128]}},  # Variable length
            {"node_params": {"layer_type": "lstm", "units": 64, "return_sequences": True}},
        ]
    }

    mock_model = Mock()
    issues = validate_onnx_compatible(mock_model, graph_ir)

    assert len(issues) > 0
    assert "LSTM/GRU" in issues[0]
    assert "fixed input sequence length" in issues[0]


@pytest.mark.skipif(
    not importlib.util.find_spec("tf2onnx") or not _check_tf2onnx_available(),
    reason="tf2onnx not available or dependencies incompatible",
)
def test_onnx_export_simple_dense(tmp_path, mock_keras_model):
    """ONNX export succeeds for simple Dense model."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [28, 28]}},
            {"node_params": {"layer_type": "dense", "units": 10}},
        ]
    }

    # Mock ONNX model
    mock_onnx_model = Mock()
    mock_onnx_model.SerializeToString.return_value = b"onnx model bytes"

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        patch("tensorflow.TensorSpec", return_value="tensor_spec"),
        patch("tf2onnx.convert.from_keras", return_value=(mock_onnx_model, None)),
    ):
        onnx_path = export_onnx(job_id, model_name, graph_ir)

        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"


def test_onnx_export_raises_on_unsupported(tmp_path, mock_keras_model):
    """ONNX export raises ONNXUnsupportedError for incompatible architecture."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [None, 128]}},
            {"node_params": {"layer_type": "lstm", "units": 64, "return_sequences": True}},
        ]
    }

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        pytest.raises(ONNXUnsupportedError) as exc_info,
    ):
        export_onnx(job_id, model_name, graph_ir)

    assert len(exc_info.value.issues) > 0


def test_onnx_export_degrades_when_toolchain_unavailable(tmp_path, mock_keras_model):
    """A broken onnx/tf2onnx toolchain raises ONNXUnsupportedError, not a raw exception.

    onnx 1.19 needs ml_dtypes.float4_e2m1fn, which the ml_dtypes 0.3.2 that
    TensorFlow 2.16 pins does not provide, so ``import tf2onnx`` raises
    AttributeError at import time. That must surface as the documented
    400 "onnx_unsupported" response, not a raw 500.
    """
    import builtins

    job_id = str(uuid4())
    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "tf2onnx":
            raise AttributeError("module 'ml_dtypes' has no attribute 'float4_e2m1fn'")
        return real_import(name, *args, **kwargs)

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        patch.object(builtins, "__import__", side_effect=broken_import),
        pytest.raises(ONNXUnsupportedError) as exc_info,
    ):
        export_onnx(job_id, "test_model")

    joined = "; ".join(exc_info.value.issues)
    assert "tf2onnx/onnx toolchain unavailable" in joined
    assert "ml_dtypes" in joined


@pytest.mark.skipif(
    not importlib.util.find_spec("tf2onnx") or not _check_tf2onnx_available(),
    reason="tf2onnx not available or dependencies incompatible",
)
def test_onnx_export_succeeds_with_fixed_shape(tmp_path, mock_keras_model):
    """ONNX export succeeds for LSTM with fixed input shape."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [10, 128]}},  # Fixed length
            {"node_params": {"layer_type": "lstm", "units": 64, "return_sequences": True}},
        ]
    }

    # Mock ONNX model
    mock_onnx_model = Mock()
    mock_onnx_model.SerializeToString.return_value = b"onnx model bytes"

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        patch("tensorflow.TensorSpec", return_value="tensor_spec"),
        patch("tf2onnx.convert.from_keras", return_value=(mock_onnx_model, None)),
    ):
        onnx_path = export_onnx(job_id, model_name, graph_ir)

        assert onnx_path.exists()


def test_get_export_formats_structure(tmp_path):
    """get_export_formats returns correct structure."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    graph_ir = {"nodes": [{"node_params": {"layer_type": "dense", "units": 10}}]}

    mock_model = Mock()
    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_model),
    ):
        formats = get_export_formats(job_id, model_name, graph_ir)

        assert "savedmodel" in formats
        assert "tflite" in formats
        assert "onnx" in formats

        # Check structure
        assert "available" in formats["savedmodel"]
        assert "size_bytes" in formats["savedmodel"]
        assert "expires_at" in formats["savedmodel"]

        # ONNX should have additional fields
        assert "onnx_supported" in formats["onnx"]


def test_cleanup_deletes_old_directories(tmp_path, db_session: Session):
    """cleanup_exports deletes export directories older than retention_days."""
    # Create old export directory
    old_job_id = str(uuid4())
    old_dir = tmp_path / "exports" / old_job_id
    old_dir.mkdir(parents=True)
    (old_dir / "model.keras").write_text("old")

    # Create recent export directory
    recent_job_id = str(uuid4())
    recent_dir = tmp_path / "exports" / recent_job_id
    recent_dir.mkdir(parents=True)
    (recent_dir / "model.keras").write_text("recent")

    # Parent model row so the training_job FK constraint is satisfied (Postgres/CI).
    db_session.add(ModelBasic(id=1, model_name="export-cleanup-model"))
    db_session.commit()

    # Create training jobs
    old_job = TrainingJob(
        id=old_job_id,
        model_id=1,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        completed_at=datetime.now(UTC) - timedelta(days=10),
    )
    recent_job = TrainingJob(
        id=recent_job_id,
        model_id=1,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        completed_at=datetime.now(UTC),
    )

    db_session.add(old_job)
    db_session.add(recent_job)
    db_session.commit()

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        count = cleanup_exports(retention_days=7, session=db_session)

        assert count == 1
        assert not old_dir.exists()
        assert recent_dir.exists()


def test_delete_job_exports(tmp_path):
    """delete_job_exports removes export directory for a specific job."""
    job_id = str(uuid4())
    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        result = delete_job_exports(job_id)

        assert result is True
        assert not export_dir.exists()


def test_delete_model_exports_cascades(tmp_path, db_session: Session):
    """delete_model_exports removes all export directories for model's jobs."""
    model_id = 1

    # Parent model row so the training_job FK constraint is satisfied (Postgres/CI).
    db_session.add(ModelBasic(id=model_id, model_name="export-cascade-model"))
    db_session.commit()

    # Create multiple jobs for the model
    job1_id = str(uuid4())
    job2_id = str(uuid4())

    job1 = TrainingJob(id=job1_id, model_id=model_id, status=TrainingStatus.COMPLETED, hyperparams={})
    job2 = TrainingJob(id=job2_id, model_id=model_id, status=TrainingStatus.COMPLETED, hyperparams={})

    db_session.add(job1)
    db_session.add(job2)
    db_session.commit()

    # Create export directories
    job1_dir = tmp_path / "exports" / job1_id
    job1_dir.mkdir(parents=True)
    (job1_dir / "model.keras").write_text("job1")

    job2_dir = tmp_path / "exports" / job2_id
    job2_dir.mkdir(parents=True)
    (job2_dir / "model.keras").write_text("job2")

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        count = delete_model_exports(model_id, db_session)

        assert count == 2
        assert not job1_dir.exists()
        assert not job2_dir.exists()


def test_path_traversal_prevention(tmp_path, mock_keras_model):
    """Model names with path traversal attempts are sanitized."""
    job_id = str(uuid4())
    # Attempt path traversal with malicious model name
    malicious_name = "../../../tmp/pwned"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        patch("tensorflow.keras.models.save_model"),
    ):
        # Create savedmodel directory for zipping
        savedmodel_dir = export_dir / "savedmodel"
        savedmodel_dir.mkdir()
        (savedmodel_dir / "saved_model.pb").write_text("dummy pb")

        zip_path = export_savedmodel(job_id, malicious_name)

        # Verify the file is created inside the export_dir, not outside
        assert zip_path.is_relative_to(export_dir)
        assert ".." not in str(zip_path)
        # Verify sanitized name doesn't contain path separators
        assert "/" not in zip_path.name
        assert "\\" not in zip_path.name
        # Verify the actual path traversal was prevented
        assert zip_path.parent == export_dir


def test_export_directory_not_found(tmp_path):
    """get_export_formats returns 404-like structure when export directory doesn't exist."""
    job_id = str(uuid4())
    model_name = "nonexistent_model"

    # No export directory created - job hasn't run yet
    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        formats = get_export_formats(job_id, model_name, None)

        # All formats should be unavailable
        assert formats["savedmodel"]["available"] is False
        assert formats["tflite"]["available"] is False
        assert formats["onnx"]["available"] is False

        # ONNX should indicate training not completed
        assert formats["onnx"]["onnx_supported"] is False
        assert "Training not completed" in formats["onnx"]["onnx_issues"]


def test_concurrent_export_requests(tmp_path, mock_keras_model):
    """Two simultaneous requests for same format should both succeed (no duplicate generation)."""
    import threading

    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    results = []
    errors = []

    def export_worker():
        try:
            # Mock TFLite converter
            mock_converter = Mock()
            mock_converter.convert.return_value = b"tflite model bytes"

            import tensorflow as tf

            with (
                patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
                patch.object(tf.keras.models, "load_model", return_value=mock_keras_model),
                patch.object(tf.lite.TFLiteConverter, "from_saved_model", return_value=mock_converter),
            ):
                result = export_tflite(job_id, model_name)
                results.append(result)
        except Exception as e:
            errors.append(e)

    # Start two threads simultaneously
    thread1 = threading.Thread(target=export_worker)
    thread2 = threading.Thread(target=export_worker)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # Both should succeed (no errors)
    assert len(errors) == 0
    assert len(results) == 2

    # Both should return the same path
    assert results[0] == results[1]
    assert results[0].exists()


def test_storage_limit_triggers_cleanup(tmp_path, db_session: Session):
    """Verify that cleanup_exports is called with aggressive retention when storage exceeds limit."""
    # This test verifies the cleanup logic, not the main.py background task itself
    # Create multiple export directories
    from datetime import timedelta

    model_id = 1
    db_session.add(ModelBasic(id=model_id, model_name="storage-test-model"))
    db_session.commit()

    # Create old jobs that should be cleaned up
    old_jobs = []
    for _ in range(3):
        job_id = str(uuid4())
        export_dir = tmp_path / "exports" / job_id
        export_dir.mkdir(parents=True)
        (export_dir / "model.keras").write_text("old" * 1000)  # Some content

        job = TrainingJob(
            id=job_id,
            model_id=model_id,
            status=TrainingStatus.COMPLETED,
            hyperparams={},
            completed_at=datetime.now(UTC) - timedelta(days=2),  # 2 days old
        )
        db_session.add(job)
        old_jobs.append(job_id)

    db_session.commit()

    # Verify cleanup with 1 day retention (aggressive)
    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        count = cleanup_exports(retention_days=1, session=db_session)

        # Should delete all 3 directories (all are > 1 day old)
        assert count == 3

        for job_id in old_jobs:
            export_dir = tmp_path / "exports" / job_id
            assert not export_dir.exists()


def test_export_size_in_formats_response(tmp_path, mock_keras_model):
    """Verify that get_export_formats returns correct size_bytes for existing exports."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    # Create a TFLite export with known size
    tflite_path = export_dir / f"{model_name}.tflite"
    test_content = b"tflite test content" * 100
    tflite_path.write_bytes(test_content)
    expected_size = len(test_content)

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        formats = get_export_formats(job_id, model_name, None)

        # TFLite should be available with correct size
        assert formats["tflite"]["available"] is True
        assert formats["tflite"]["size_bytes"] == expected_size

        # SavedModel and ONNX should not be available
        assert formats["savedmodel"]["available"] is False
        assert formats["onnx"]["available"] is False


def test_expires_at_in_formats_response(tmp_path):
    """Verify that expires_at is correctly calculated and formatted."""
    import os
    from datetime import timedelta

    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    # Create a TFLite export
    tflite_path = export_dir / f"{model_name}.tflite"
    tflite_path.write_bytes(b"test")

    # Record the creation time
    creation_time = datetime.fromtimestamp(tflite_path.stat().st_mtime, UTC)

    # Mock retention days
    retention_days = 7
    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch.dict(os.environ, {"EXPORT_RETENTION_DAYS": str(retention_days)}),
    ):
        formats = get_export_formats(job_id, model_name, None)

        # Verify expires_at exists and is correctly formatted
        assert formats["tflite"]["expires_at"] is not None
        expires_at_str = formats["tflite"]["expires_at"]

        # Parse ISO format
        expires_at = datetime.fromisoformat(expires_at_str)
        expected_expires = creation_time + timedelta(days=retention_days)

        # Allow 1 second tolerance for file system time precision
        time_diff = abs((expires_at - expected_expires).total_seconds())
        assert time_diff < 1, f"Expected {expected_expires}, got {expires_at}"
