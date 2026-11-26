"""Service layer for inspecting and configuring image dataset preprocessing properties."""

import uuid as uuid_pkg
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from app.config import get_settings
from app.models.data import DataFile, ImageProperties
from app.schemas.data_process import ImagePropertiesUpdateRequest
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

ALLOWED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif"})
_COLOR_MODE_CHANNELS: dict[str, int] = {"grayscale": 1, "rgb": 3, "rgba": 4}
_DEFAULT_IMAGE_SIZE = 224
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_COLOR_MODE = "rgb"
_DEFAULT_LABEL_MODE = "int"


def _resp(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    """Build a standard API response tuple of (body_dict, status_code)."""
    return {"success": success, "message": message, "data": data}, status_code


def _resolve_safe_image_dir(upload_folder: str, disk_name: str) -> Path:
    """Resolve extracted image directory while ensuring containment within upload_folder."""
    base_dir = Path(upload_folder).resolve()
    extract_name = disk_name.rsplit(".", 1)[0]
    candidate = (base_dir / extract_name).resolve()
    if not candidate.is_relative_to(base_dir) or candidate == base_dir:
        raise ValueError(f"Unsafe extracted directory path for dataset '{disk_name}'")
    return candidate


def _scan_image_classes(image_dir: Path) -> tuple[list[dict[str, Any]], int, bool]:
    """Scan class subdirectories and count valid image files."""
    if not image_dir.is_dir():
        return [], 0, False

    classes: list[dict[str, Any]] = []
    total_images = 0
    for child in sorted(image_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        count = sum(
            1 for item in child.rglob("*") if item.is_file() and item.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        )
        classes.append({"name": child.name, "count": count})
        total_images += count
    return classes, total_images, True


def _serialize_image_properties(data_file: DataFile, props: ImageProperties | None) -> dict[str, Any]:
    """Build API payload combining ImageProperties with extracted directory summary."""
    image_size = props.image_size if props is not None else _DEFAULT_IMAGE_SIZE
    batch_size = props.batch_size if props is not None else _DEFAULT_BATCH_SIZE
    color_mode = props.color_mode if props is not None else _DEFAULT_COLOR_MODE
    label_mode = props.label_mode if props is not None else _DEFAULT_LABEL_MODE
    channels = _COLOR_MODE_CHANNELS.get(color_mode, 3)

    upload_folder = get_settings().upload_folder
    classes: list[dict[str, Any]] = []
    total_images = 0
    directory_exists = False
    try:
        image_dir = _resolve_safe_image_dir(upload_folder, data_file.disk_name)
        classes, total_images, directory_exists = _scan_image_classes(image_dir)
    except (OSError, ValueError) as e:
        logger.warning("Failed to scan image directory for file_id=%s: %s", data_file.id, e)

    return {
        "file_id": str(data_file.id),
        "file_name": data_file.file_name,
        "image_size": image_size,
        "batch_size": batch_size,
        "color_mode": color_mode,
        "label_mode": label_mode,
        "channels": channels,
        "suggested_input_shape": f"{image_size}, {image_size}, {channels}",
        "total_images": total_images,
        "num_classes": len(classes),
        "classes": classes,
        "directory_exists": directory_exists,
    }


def _load_zip_file(db: Session, file_id: uuid_pkg.UUID) -> tuple[DataFile | None, tuple | None]:
    """Fetch a DataFile record and validate it is a ZIP image dataset archive."""
    data_file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not data_file:
        return None, _resp(404, False, "Dataset file not found")
    if data_file.file_type != "zip":
        return None, _resp(
            400,
            False,
            f"Image properties are only available for ZIP image datasets, got '{data_file.file_type}'",
        )
    return data_file, None


def get_image_properties_service(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    """Return image preprocessing properties and class summary for a ZIP dataset."""
    data_file, err = _load_zip_file(db, file_id)
    if err is not None:
        return err

    assert data_file is not None
    props = db.exec(select(ImageProperties).where(ImageProperties.id == file_id)).first()
    payload = _serialize_image_properties(data_file, props)
    return _resp(200, True, "Image dataset properties fetched successfully", payload)


def update_image_properties_service(
    db: Session,
    file_id: uuid_pkg.UUID,
    updates: ImagePropertiesUpdateRequest,
) -> tuple:
    """Update persisted ImageProperties for a ZIP image dataset."""
    data_file, err = _load_zip_file(db, file_id)
    if err is not None:
        return err

    assert data_file is not None
    props = db.exec(select(ImageProperties).where(ImageProperties.id == file_id)).first()
    if props is None:
        props = ImageProperties(
            id=file_id,
            image_size=updates.image_size if updates.image_size is not None else _DEFAULT_IMAGE_SIZE,
            batch_size=updates.batch_size if updates.batch_size is not None else _DEFAULT_BATCH_SIZE,
            color_mode=updates.color_mode if updates.color_mode is not None else _DEFAULT_COLOR_MODE,
            label_mode=updates.label_mode if updates.label_mode is not None else _DEFAULT_LABEL_MODE,
        )
        db.add(props)
    else:
        if updates.image_size is not None:
            props.image_size = updates.image_size
        if updates.batch_size is not None:
            props.batch_size = updates.batch_size
        if updates.color_mode is not None:
            props.color_mode = updates.color_mode
        if updates.label_mode is not None:
            props.label_mode = updates.label_mode

    try:
        db.commit()
        db.refresh(props)
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Failed to update image properties for file_id=%s", file_id)
        return _resp(500, False, "Failed to update image dataset properties")

    payload = _serialize_image_properties(data_file, props)
    return _resp(200, True, "Image dataset properties updated successfully", payload)
