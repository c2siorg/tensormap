"""Reproducibility snapshot service for training runs.

Captures execution context, deterministic dataset fingerprints, training
configuration, random seeds, and runtime ML library versions to allow
accurate reproduction and auditability of training experiments (Issue #520).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import uuid as uuid_pkg
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, select

from app.config import get_settings
from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def compute_file_fingerprint(path: str | None) -> str | None:
    """Compute a deterministic SHA-256 content fingerprint for a file or directory.

    For regular files, hashes the raw byte content in 64KB chunks.
    For directories (e.g. unzipped image datasets), hashes relative file paths
    and contents in strictly sorted order.
    Returns None if the path does not exist or cannot be read.
    """
    if not path or not os.path.exists(path):
        return None

    try:
        hasher = hashlib.sha256()
        if os.path.isfile(path):
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return f"sha256:{hasher.hexdigest()}"

        if os.path.isdir(path):
            # Sort paths recursively for cross-platform deterministic order
            for root, _, files in sorted(os.walk(path)):
                for filename in sorted(files):
                    file_full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_full_path, path).replace("\\", "/")
                    hasher.update(rel_path.encode("utf-8"))
                    with open(file_full_path, "rb") as f:
                        while chunk := f.read(65536):
                            hasher.update(chunk)
            return f"sha256:{hasher.hexdigest()}"
    except OSError as exc:
        logger.warning("Could not compute fingerprint for %s: %s", path, exc)
        return None

    return None


def get_runtime_environment() -> dict[str, Any]:
    """Capture runtime platform, Python version, and core ML library versions.

    Explicitly excludes credentials, environment variables, or arbitrary system tokens.
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            "tensorflow": getattr(tf, "__version__", None),
            "numpy": getattr(np, "__version__", None),
            "pandas": getattr(pd, "__version__", None),
        },
    }


def resolve_dataset_snapshot(session: Session, file_id: uuid_pkg.UUID | str | None) -> dict[str, Any] | None:
    """Resolve dataset metadata and compute its deterministic fingerprint."""
    if file_id is None:
        return None

    file = session.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if file is None:
        return None

    upload_folder = get_settings().upload_folder
    raw_file_path = os.path.join(upload_folder, file.disk_name)

    # For zip datasets (images), check both extracted dir and original archive
    fingerprint_target = raw_file_path
    if file.file_type == "zip":
        extracted_dir = os.path.join(upload_folder, file.disk_name.rsplit(".", 1)[0])
        if os.path.isdir(extracted_dir):
            fingerprint_target = extracted_dir

    fingerprint = compute_file_fingerprint(fingerprint_target)
    file_size_bytes = None
    if os.path.exists(raw_file_path):
        with contextlib.suppress(OSError):
            file_size_bytes = os.path.getsize(raw_file_path)

    return {
        "file_id": str(file.id),
        "file_name": file.file_name,
        "file_type": file.file_type,
        "disk_name": file.disk_name,
        "fingerprint": fingerprint,
        "file_size_bytes": file_size_bytes,
        "row_count": file.row_count,
    }


def _problem_type_name(problem_type: int | None) -> str | None:
    if problem_type is None:
        return None
    mapping = {
        ProblemType.CLASSIFICATION: "classification",
        ProblemType.REGRESSION: "regression",
        ProblemType.IMAGE_CLASSIFICATION: "image_classification",
    }
    return mapping.get(problem_type, str(problem_type))


def capture_reproducibility_snapshot(
    session: Session,
    model: ModelBasic,
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture a complete reproducibility snapshot for a training job."""
    hp = hyperparams or {}

    optimizer = hp.get("optimizer") or model.optimizer
    learning_rate = hp.get("lr")
    epochs = hp.get("epochs") or model.epochs
    batch_size = hp.get("batch_size") or model.batch_size
    training_split = model.training_split

    # Determine random seed: explicit override > model-type defaults
    random_seed = hp.get("seed") or hp.get("random_seed")
    if random_seed is None:
        random_seed = 123 if model.model_type == ProblemType.IMAGE_CLASSIFICATION else 42

    dataset_snapshot = resolve_dataset_snapshot(session, model.file_id)

    training_config = {
        "model_name": model.model_name,
        "problem_type": _problem_type_name(model.model_type),
        "problem_type_id": model.model_type,
        "target_field": model.target_field,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "loss": model.loss,
        "metric": model.metric,
        "epochs": epochs,
        "batch_size": batch_size,
        "training_split": training_split,
        "hyperparameters": hp,
    }

    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_snapshot,
        "training_config": training_config,
        "random_seed": random_seed,
        "runtime_environment": get_runtime_environment(),
    }
