"""Helpers for capturing versioned training-run provenance."""

import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import pandas as pd

CHUNK_SIZE = 1024 * 1024


def _stat_signature(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (file_stat.st_dev, file_stat.st_ino, file_stat.st_size, file_stat.st_mtime_ns, file_stat.st_ctime_ns)


def _copy_and_hash(
    source: BinaryIO, destination: BinaryIO, digest, cancel_check: Callable[[], bool] | None = None
) -> int:
    size = 0
    while chunk := source.read(CHUNK_SIZE):
        if cancel_check is not None and cancel_check():
            raise TrainingCancelledError("Training job was cancelled during image dataset staging")
        destination.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    return size


def read_csv_with_fingerprint(file_path: str) -> tuple[pd.DataFrame, str, int]:
    """Hash a staged CSV copy, then parse exactly that copy with bounded copying memory.

    Defense layers against source mutation during fingerprinting:
    1. Stage a copy to temporary storage before hashing or parsing
    2. Check source file stat signature (dev, ino, size, mtime_ns, ctime_ns) before and after copying
    3. Verify copied byte count matches source file size
    4. Parse only the staged copy (source mutations after staging don't affect training)

    Limitations: This approach detects most concurrent modification patterns but cannot
    guarantee absolute immunity to adversarial filesystem tampering. A sophisticated
    attacker with the ability to restore mtime_ns/ctime_ns after same-size content
    mutation could evade detection during the copy operation itself.
    """
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="tensormap-training-csv-") as temp_dir:
        staged_path = Path(temp_dir) / "dataset.csv"
        with open(file_path, "rb") as source, staged_path.open("wb") as staged:
            before = os.fstat(source.fileno())
            size = _copy_and_hash(source, staged, digest)
            after_copy = os.fstat(source.fileno())
            if _stat_signature(before) != _stat_signature(after_copy) or size != before.st_size:
                raise OSError("Dataset changed while its fingerprint was being calculated")

        frame = pd.read_csv(staged_path)

    return frame, digest.hexdigest(), size


class TrainingCancelledError(Exception):
    """Raised when a training job is cancelled before or during model fitting."""


def _walk_dataset(root: Path) -> list[tuple[str, Path, bool]]:
    """Return a stable tree listing, including directories that define classes."""
    entries: list[tuple[str, Path, bool]] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirs.sort()
        files.sort()
        for dirname in dirs:
            path = current_path / dirname
            if path.is_symlink():
                raise OSError("Image dataset contains a symbolic link")
            relpath = path.relative_to(root).as_posix()
            entries.append((relpath, path, True))
        for filename in files:
            path = current_path / filename
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise OSError("Image dataset contains a non-regular file")
            relpath = path.relative_to(root).as_posix()
            entries.append((relpath, path, False))
    entries.sort(key=lambda entry: entry[0])
    return entries


@contextmanager
def image_dataset_snapshot(source_dir: str, cancel_check: Callable[[], bool] | None = None) -> Iterator[dict]:
    """Stage an immutable image tree and fingerprint its paths and file bytes.

    Manifest encoding is SHA-256 over a version marker followed by sorted entries.
    Each entry contains a one-byte type marker (directory/file), an unsigned
    64-bit big-endian UTF-8 path length, the relative POSIX path, and for files
    an unsigned 64-bit byte length followed by the exact file bytes.
    """
    source_root = Path(source_dir)
    if not source_root.is_dir():
        raise FileNotFoundError("Image dataset directory is unavailable")

    if cancel_check is not None and cancel_check():
        raise TrainingCancelledError("Training job was cancelled before image dataset staging")

    digest = hashlib.sha256(b"TensorMap image dataset manifest\x00v1")
    total_bytes = 0
    with tempfile.TemporaryDirectory(prefix="tensormap-training-images-") as temp_dir:
        staged_root = Path(temp_dir) / "dataset"
        staged_root.mkdir()
        entries = _walk_dataset(source_root)
        class_names = sorted(path for path, _, is_directory in entries if is_directory and "/" not in path)
        for relative_path, source_path, is_directory in entries:
            if cancel_check is not None and cancel_check():
                raise TrainingCancelledError("Training job was cancelled during image dataset staging")
            path_bytes = relative_path.encode("utf-8")
            digest.update(b"D" if is_directory else b"F")
            digest.update(len(path_bytes).to_bytes(8, "big"))
            digest.update(path_bytes)
            destination = staged_root / relative_path
            if is_directory:
                destination.mkdir(parents=True, exist_ok=True)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            before = source_path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise OSError("Image dataset contains a non-regular file")
            with source_path.open("rb") as source, destination.open("wb") as output:
                opened = os.fstat(source.fileno())
                if _stat_signature(before) != _stat_signature(opened):
                    raise OSError("Image dataset changed while its fingerprint was being calculated")
                digest.update(opened.st_size.to_bytes(8, "big"))
                size = _copy_and_hash(source, output, digest, cancel_check)
                after = os.fstat(source.fileno())
                if _stat_signature(opened) != _stat_signature(after) or size != opened.st_size:
                    raise OSError("Image dataset changed while its fingerprint was being calculated")
            total_bytes += size

        if _walk_dataset(source_root) != entries:
            raise OSError("Image dataset changed while its fingerprint was being calculated")
        if cancel_check is not None and cancel_check():
            raise TrainingCancelledError("Training job was cancelled after image dataset staging")

        yield {
            "path": str(staged_root),
            "fingerprint": digest.hexdigest(),
            "size_bytes": total_bytes,
            "class_names": class_names,
        }


def runtime_versions() -> dict[str, str | None]:
    """Return only the runtime versions relevant to supported training paths."""
    import importlib.metadata
    import platform

    def version(package: str) -> str | None:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "tensorflow": version("tensorflow"),
        "keras": version("keras"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
    }


_UNIX_ABSOLUTE_PATH = re.compile(r"(?<![\w:])/(?:[^\s:]+)")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)\b[A-Z]:\\[^\s:]+")


def sanitize_error_message(message: str) -> str:
    """Remove absolute filesystem paths while retaining the error category and detail."""
    sanitized = _UNIX_ABSOLUTE_PATH.sub("<path>", message)
    return _WINDOWS_ABSOLUTE_PATH.sub("<path>", sanitized)[:2000]
