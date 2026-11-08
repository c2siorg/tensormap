"""Tests for dataset fingerprinting and the reproducibility snapshot schema."""

import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

import app.services.reproducibility as reproducibility
from app.schemas.training import (
    DatasetProvenance,
    RandomnessSnapshot,
    ReproducibilitySnapshot,
    RuntimeVersions,
    TrainingConfigurationSnapshot,
)
from app.services.reproducibility import image_dataset_snapshot, read_csv_with_fingerprint, runtime_versions


def test_csv_fingerprint_is_stable_and_changes_with_content(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_bytes(b"feature,target\n1,a\n")
    _, first, _ = read_csv_with_fingerprint(str(dataset))
    _, same, _ = read_csv_with_fingerprint(str(dataset))

    dataset.write_bytes(b"feature,target\n2,a\n")
    _, changed, _ = read_csv_with_fingerprint(str(dataset))

    assert first == same
    assert first != changed


def test_csv_fingerprint_streams_large_files_in_chunks(tmp_path, monkeypatch):
    content = b"feature\n" + b"x" * (reproducibility.CHUNK_SIZE * 2 + 17) + b"\n"
    dataset = tmp_path / "large.csv"
    dataset.write_bytes(content)
    real_sha256 = hashlib.sha256
    update_sizes = []

    class TrackingHash:
        def __init__(self):
            self._digest = real_sha256()

        def update(self, value):
            update_sizes.append(len(value))
            self._digest.update(value)

        def hexdigest(self):
            return self._digest.hexdigest()

    monkeypatch.setattr(reproducibility.hashlib, "sha256", TrackingHash)
    _, fingerprint, size_bytes = read_csv_with_fingerprint(str(dataset))

    assert fingerprint == real_sha256(content).hexdigest()
    assert size_bytes == len(content)
    assert max(update_sizes) <= reproducibility.CHUNK_SIZE
    assert len(update_sizes) > 2


def test_csv_path_replacement_after_staging_keeps_hash_and_data_on_staged_copy(tmp_path, monkeypatch):
    dataset = tmp_path / "data.csv"
    replacement = tmp_path / "replacement.csv"
    original_content = b"feature,target\n1,old\n"
    replacement_content = b"feature,target\n2,new\n"
    dataset.write_bytes(original_content)
    replacement.write_bytes(replacement_content)
    real_read_csv = pd.read_csv

    def replace_path_then_parse(source, *args, **kwargs):
        assert Path(source).parent.name.startswith("tensormap-training-csv-")
        os.replace(replacement, dataset)
        return real_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(reproducibility.pd, "read_csv", replace_path_then_parse)
    frame, fingerprint, size_bytes = read_csv_with_fingerprint(str(dataset))

    assert frame.iloc[0].to_dict() == {"feature": 1, "target": "old"}
    assert fingerprint == hashlib.sha256(original_content).hexdigest()
    assert size_bytes == len(original_content)
    assert dataset.read_bytes() == replacement_content


def test_csv_copy_detects_same_size_mutation_even_when_mtime_is_restored(tmp_path, monkeypatch):
    dataset = tmp_path / "data.csv"
    original_content = b"feature,target\n1,old\n"
    changed_content = b"feature,target\n2,new\n"
    dataset.write_bytes(original_content)
    original_stat = dataset.stat()
    real_copy_and_hash = reproducibility._copy_and_hash

    def mutate_after_copy(source, destination, digest, cancel_check=None):
        size = real_copy_and_hash(source, destination, digest, cancel_check)
        dataset.write_bytes(changed_content)
        os.utime(dataset, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return size

    monkeypatch.setattr(reproducibility, "_copy_and_hash", mutate_after_copy)
    with pytest.raises(OSError, match="changed while its fingerprint was being calculated"):
        read_csv_with_fingerprint(str(dataset))


def test_csv_restored_mtime_mutation_is_detected(tmp_path, monkeypatch):
    """Verify that source mutation during staging is detected even when mtime is restored.

    Attack scenario: An adversary modifies the source file after the copy begins,
    then restores mtime/atime to mask the change. On Unix systems including macOS,
    st_ctime_ns changes on content modification and cannot be restored via os.utime,
    so the stat signature check should detect the mutation.

    This test asserts that mutation detection MUST succeed on this filesystem.
    """
    dataset = tmp_path / "data.csv"
    original_content = b"feature,target\n1,old\n"
    changed_content = b"feature,target\n2,new\n"  # Same size, different content
    assert len(original_content) == len(changed_content)

    dataset.write_bytes(original_content)
    original_stat = dataset.stat()
    real_copy_and_hash = reproducibility._copy_and_hash

    def mutate_and_restore_mtime(source, destination, digest, cancel_check=None):
        # Copy and hash the original content
        size = real_copy_and_hash(source, destination, digest, cancel_check)
        # Attacker changes source file content after staging
        dataset.write_bytes(changed_content)
        # Attacker restores mtime and atime
        os.utime(dataset, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        # Note: ctime_ns cannot be directly set via os.utime on Unix
        return size

    monkeypatch.setattr(reproducibility, "_copy_and_hash", mutate_and_restore_mtime)

    # On macOS and Unix systems, the mutation MUST be detected via st_ctime_ns change
    with pytest.raises(OSError, match="changed while its fingerprint was being calculated"):
        read_csv_with_fingerprint(str(dataset))


def test_csv_staged_data_integrity_after_source_mutation(tmp_path, monkeypatch):
    """Verify that staged data integrity is preserved when source is mutated after staging.

    This tests the defensive property: even if mutation detection were to fail,
    the training path consumes the staged copy, not the mutated source. The recorded
    fingerprint corresponds to the staged bytes that training actually uses.
    """
    dataset = tmp_path / "data.csv"
    original_content = b"feature,target\n1,old\n"
    changed_content = b"feature,target\n2,new\n"

    dataset.write_bytes(original_content)
    staged_path_captured = None

    real_read_csv = pd.read_csv

    def capture_staged_path_and_mutate_source(path, *args, **kwargs):
        nonlocal staged_path_captured
        staged_path_captured = Path(path)
        # At this point, the source has been staged and hashed
        # Now mutate the original source file
        dataset.write_bytes(changed_content)
        # Return the data from the staged copy
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(reproducibility.pd, "read_csv", capture_staged_path_and_mutate_source)

    frame, fingerprint, size_bytes = read_csv_with_fingerprint(str(dataset))

    # The fingerprint MUST match the staged (original) content, not the mutated source
    assert fingerprint == hashlib.sha256(original_content).hexdigest(), (
        "Fingerprint must match staged bytes, not mutated source"
    )
    assert size_bytes == len(original_content)

    # Training uses the original data from the staged copy
    assert frame.iloc[0].to_dict() == {"feature": 1, "target": "old"}

    # The staged copy was in a temporary directory and has been cleaned up
    assert staged_path_captured is not None
    assert not staged_path_captured.exists(), "Staged copy should be cleaned up"

    # The source file now contains the mutated content
    assert dataset.read_bytes() == changed_content


def test_csv_temporary_copy_is_removed_after_parse_success_and_failure(tmp_path, monkeypatch):
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,target\n1,old\n")
    real_read_csv = pd.read_csv
    parsed_path = None

    def record_path(source, *args, **kwargs):
        nonlocal parsed_path
        parsed_path = Path(source)
        return real_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(reproducibility.pd, "read_csv", record_path)
    read_csv_with_fingerprint(str(dataset))
    assert parsed_path is not None and not parsed_path.exists()

    def fail_parse(source, *args, **kwargs):
        nonlocal parsed_path
        parsed_path = Path(source)
        raise pd.errors.ParserError("bad csv")

    monkeypatch.setattr(reproducibility.pd, "read_csv", fail_parse)
    with pytest.raises(pd.errors.ParserError):
        read_csv_with_fingerprint(str(dataset))
    assert parsed_path is not None and not parsed_path.exists()


def test_missing_csv_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_csv_with_fingerprint(str(tmp_path / "missing.csv"))


def test_image_manifest_is_stable_and_uses_staged_tree(tmp_path):
    source = tmp_path / "images"
    (source / "zebra").mkdir(parents=True)
    (source / "ant").mkdir()
    (source / "zebra" / "b.png").write_bytes(b"zebra-image")
    (source / "ant" / "a.png").write_bytes(b"ant-image")
    source_copy = tmp_path / "images-copy"
    (source_copy / "ant").mkdir(parents=True)
    (source_copy / "zebra").mkdir()
    (source_copy / "ant" / "a.png").write_bytes(b"ant-image")
    (source_copy / "zebra" / "b.png").write_bytes(b"zebra-image")

    with image_dataset_snapshot(str(source)) as first:
        assert Path(first["path"], "zebra", "b.png").read_bytes() == b"zebra-image"
        assert first["class_names"] == ["ant", "zebra"]
        with image_dataset_snapshot(str(source_copy)) as same:
            assert first["fingerprint"] == same["fingerprint"]

    (source_copy / "zebra" / "b.png").write_bytes(b"changed")
    with image_dataset_snapshot(str(source_copy)) as changed:
        assert first["fingerprint"] != changed["fingerprint"]


def test_missing_image_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError), image_dataset_snapshot(str(tmp_path / "missing-images")):
        pytest.fail("A missing image directory must not produce a snapshot")


def test_image_manifest_rejects_source_mutation_during_copy(tmp_path, monkeypatch):
    source = tmp_path / "images"
    (source / "cats").mkdir(parents=True)
    image = source / "cats" / "one.png"
    image.write_bytes(b"image")
    real_copy_and_hash = reproducibility._copy_and_hash

    def mutate_after_copy(src, destination, digest, cancel_check=None):
        size = real_copy_and_hash(src, destination, digest, cancel_check)
        image.write_bytes(b"changed image")
        return size

    monkeypatch.setattr(reproducibility, "_copy_and_hash", mutate_after_copy)
    with (
        pytest.raises(OSError, match="changed while its fingerprint was being calculated"),
        image_dataset_snapshot(str(source)),
    ):
        pytest.fail("A changed image dataset must not produce a snapshot")


def test_image_staging_checks_cancellation_and_cleans_temporary_directory(tmp_path, monkeypatch):
    source = tmp_path / "images"
    (source / "cats").mkdir(parents=True)
    (source / "cats" / "one.png").write_bytes(b"one")
    (source / "cats" / "two.png").write_bytes(b"two")
    real_temporary_directory = reproducibility.tempfile.TemporaryDirectory
    staged_dirs = []

    class TrackingTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self._inner = real_temporary_directory(*args, **kwargs)
            staged_dirs.append(Path(self._inner.name))
            self.name = self._inner.name

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

    monkeypatch.setattr(reproducibility.tempfile, "TemporaryDirectory", TrackingTemporaryDirectory)
    checks = 0

    def cancel_during_staging():
        nonlocal checks
        checks += 1
        return checks >= 3

    with (
        pytest.raises(reproducibility.TrainingCancelledError),
        image_dataset_snapshot(str(source), cancel_check=cancel_during_staging),
    ):
        pytest.fail("Cancelled staging must not yield a dataset")

    assert checks >= 3
    assert len(staged_dirs) == 1
    assert not staged_dirs[0].exists()


def test_runtime_versions_and_snapshot_schema_are_typed():
    versions = runtime_versions()
    snapshot = ReproducibilitySnapshot(
        job_id="job-1",
        captured_at="2026-11-02T12:00:00Z",
        dataset=DatasetProvenance(
            file_id="file-1",
            filename="training.csv",
            dataset_type="csv",
            fingerprint_algorithm="sha256",
            fingerprint="a" * 64,
            fingerprint_scope="file_bytes",
            size_bytes=12,
        ),
        training_configuration=TrainingConfigurationSnapshot(
            problem_type_id=1,
            target_column="label",
            feature_columns=["feature"],
            optimizer={"class_name": "Adam", "config": {"learning_rate": 0.01}},
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            epochs=5,
            batch_size=16,
            training_split_percent=80,
            validation_split_percent=20,
            configured_hyperparameters={"saved_epochs": 3, "job_overrides": {"epochs": 5}},
            effective_hyperparameters={"epochs": 5, "batch_size": 16},
        ),
        randomness=RandomnessSnapshot(
            configured_seed=None,
            effective_seeds={"tabular_shuffle": 42, "image_dataset_split": None},
            applied_seeds=["tabular_shuffle"],
        ),
        runtime_versions=RuntimeVersions(**versions),
    )

    assert snapshot.schema_version == 2
    assert snapshot.randomness.configured_seed is None
    assert not snapshot.randomness.global_framework_seed_applied
    assert not snapshot.randomness.deterministic_execution_guaranteed
    assert snapshot.model_dump(mode="json")["runtime_versions"]["python"] == versions["python"]


def test_runtime_versions_return_none_for_missing_distribution(monkeypatch):
    import importlib.metadata

    real_version = importlib.metadata.version

    def missing_tensorflow(package):
        if package == "tensorflow":
            raise importlib.metadata.PackageNotFoundError(package)
        return real_version(package)

    monkeypatch.setattr(importlib.metadata, "version", missing_tensorflow)
    assert runtime_versions()["tensorflow"] is None


def test_legacy_snapshot_schema_version_one_still_validates():
    snapshot = ReproducibilitySnapshot(
        schema_version=1,
        job_id="old-job",
        captured_at="2026-11-02T12:00:00Z",
        dataset=DatasetProvenance(
            file_id="file-1",
            filename="training.csv",
            dataset_type="csv",
            fingerprint_algorithm="sha256",
            fingerprint="a" * 64,
            fingerprint_scope="file_bytes",
            size_bytes=12,
        ),
        training_configuration=TrainingConfigurationSnapshot(),
        randomness=RandomnessSnapshot(),
        runtime_versions=RuntimeVersions(python="3.12.0"),
    )
    assert snapshot.schema_version == 1
    assert snapshot.training_configuration.actual_training_rows is None
