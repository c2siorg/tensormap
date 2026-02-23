import uuid as uuid_pkg
from typing import Any

import pandas as pd
from sqlmodel import Session, select
from werkzeug.utils import secure_filename

from app.config import get_settings
from app.models import DataFile, DataProcess, ImageProperties
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _resp(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    return {"success": success, "message": message, "data": data}, status_code


def _get_file_path(file: DataFile) -> str:
    settings = get_settings()
    if file.file_type == "zip":
        return f"{settings.upload_folder}/{file.file_name}"
    return f"{settings.upload_folder}/{file.file_name}.{file.file_type}"


def add_target_service(db: Session, file_id: uuid_pkg.UUID, target: str) -> tuple:
    try:
        file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
        if not file:
            return _resp(400, False, "File doesn't exist in DB")

        data_process = DataProcess(file_id=file_id, target=target)
        db.add(data_process)
        db.commit()
        return _resp(201, True, "Target field added successfully")
    except Exception as e:
        logger.exception("Error storing record: %s", str(e))
        return _resp(500, False, f"Error storing record: {e}")


def get_all_targets_service(db: Session) -> tuple:
    results = db.exec(select(DataProcess)).all()
    data = []
    for proc in results:
        file = db.exec(select(DataFile).where(DataFile.id == proc.file_id)).first()
        if file:
            data.append(
                {
                    "file_id": str(proc.file_id),
                    "file_name": file.file_name,
                    "file_type": file.file_type,
                    "target_field": proc.target,
                }
            )
    return _resp(200, True, "Target fields of all files received successfully", data)


def delete_one_target_by_id_service(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "File doesn't exist in DB")

    target_record = db.exec(select(DataProcess).where(DataProcess.file_id == file_id)).first()
    if not target_record:
        return _resp(400, False, "Target field doesn't exist")

    db.delete(target_record)
    db.commit()
    return _resp(200, True, "Target field deleted successfully")


def get_one_target_by_id_service(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "File doesn't exist in DB")

    target_record = db.exec(select(DataProcess).where(DataProcess.file_id == file_id)).first()
    if not target_record:
        return _resp(400, False, "Target field doesn't exist")

    data = {
        "file_name": file.file_name,
        "file_type": file.file_type,
        "target_field": target_record.target,
    }
    return _resp(200, True, "Target fields of all files received successfully", data)


def get_data_metrics(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "File doesn't exist in DB")

    file_path = _get_file_path(file)
    try:
        # Read a smaller sample to prevent DoS via massive CSVs
        df = pd.read_csv(file_path, nrows=5000)
        metrics = {
            "data_types": df.dtypes.apply(str).to_dict(),
            "correlation_matrix": df.corr(numeric_only=True).map(str).to_dict(),
            "metric": df.describe().map(str).to_dict(),
        }
        return _resp(200, True, "Dataset metrics generated successfully", metrics)
    except Exception as e:
        logger.exception("Error generating data metrics: %s", str(e))
        return _resp(500, False, "Error generating data metrics")


def get_file_data(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "Unable to open file")

    file_path = _get_file_path(file)
    try:
        # Only read top 1000 records to prevent memory explosion sending JSON
        df = pd.read_csv(file_path, nrows=1000)
        data_json = df.to_json(orient="records")
        return _resp(200, True, "Data sent successfully", data_json)
    except Exception as e:
        logger.exception("Error extracting file data: %s", str(e))
        return _resp(500, False, "Error extracting file data")


def preprocess_data(db: Session, file_id: uuid_pkg.UUID, transformations: list) -> tuple:
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if not file:
        return _resp(400, False, "File doesn't exist in DB")

    settings = get_settings()
    file_path = _get_file_path(file)

    try:
        df = pd.read_csv(file_path)
        for t in transformations:
            if t.transformation == "One Hot Encoding":
                df = pd.get_dummies(df, columns=[t.feature])
            if t.transformation == "Categorical to Numerical":
                df[t.feature] = pd.Categorical(df[t.feature]).codes
            if t.transformation == "Drop Column":
                df = df.drop(columns=[t.feature])

        file_name = file.file_name + "_preprocessed.csv"
        file_name_db = secure_filename(file_name.rsplit(".", 1)[0].lower())
        file_type_db = file_name.rsplit(".", 1)[1].lower()
        record = DataFile(file_name=file_name_db, file_type=file_type_db)
        db.add(record)
        db.commit()
        df.to_csv(f"{settings.upload_folder}/{file_name}", index=False)
        return _resp(200, True, f"Preprocessed Dataset created with name: {file_name}")
    except pd.errors.ParserError as e:
        logger.exception("CSV parsing error: %s", str(e))
        return _resp(500, False, f"Error parsing CSV data: {e}")
    except Exception as e:
        logger.exception("Error preprocessing data: %s", str(e))
        return _resp(500, False, f"Error preprocessing data: {e}")


def update_image_properties(
    db: Session, file_id: uuid_pkg.UUID, image_size: int, batch_size: int, color_mode: str, label_mode: str
) -> tuple:
    try:
        image_properties = db.exec(select(ImageProperties).where(ImageProperties.id == file_id)).first()

        if image_properties:
            image_properties.image_size = image_size
            image_properties.batch_size = batch_size
            image_properties.color_mode = color_mode
            image_properties.label_mode = label_mode
            db.add(image_properties)
            db.commit()
            return _resp(200, True, "Image properties updated successfully")
        else:
            new_props = ImageProperties(
                id=file_id,
                image_size=image_size,
                batch_size=batch_size,
                color_mode=color_mode,
                label_mode=label_mode,
            )
            db.add(new_props)
            db.commit()
            return _resp(201, True, "Image properties added successfully")
    except Exception:
        logger.exception("Error upserting image properties")
        return _resp(500, False, "An error occurred while upserting the image properties")
