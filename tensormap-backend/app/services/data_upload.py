import os
import shutil
import uuid as uuid_pkg
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import tensorflow as tf
from sqlmodel import Session, select
from werkzeug.utils import secure_filename

from app.config import get_settings
from app.models import DataFile
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _resp(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    return {"success": success, "message": message, "data": data}, status_code


def _load_dataset(file_path: str, filename: str, upload_folder: str) -> bool:
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            unzip_folder = os.path.join(upload_folder, filename.rsplit(".", 1)[0])
            zip_ref.extractall(unzip_folder)
        with tf.device("/CPU:0"):
            tf.keras.preprocessing.image_dataset_from_directory(unzip_folder)
            return True
    except zipfile.BadZipFile:
        logger.exception("Invalid zip file: %s", file_path)
        return False
    except Exception:
        logger.exception("An error occurred loading dataset from: %s", file_path)
        return False


def add_file_service(db: Session, file_wrapper: Any, project_id: uuid_pkg.UUID | None = None) -> tuple:
    settings = get_settings()
    upload_folder = settings.upload_folder
    os.makedirs(upload_folder, exist_ok=True)

    filename = secure_filename(file_wrapper.filename.lower())
    file_path = os.path.join(upload_folder, filename)
    file_wrapper.save(file_path)

    if filename.endswith(".zip"):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_load_dataset, file_path, filename, upload_folder)
                result = future.result(timeout=10)
            if result is False:
                raise Exception("The ZIP file doesnt have images in the proper format.")

            file_name_db = secure_filename(file_wrapper.filename.rsplit(".", 1)[0].lower())
            file_type_db = file_wrapper.filename.rsplit(".", 1)[1].lower()
            logger.info("Saving file: %s (type: %s)", file_name_db, file_type_db)
            record = DataFile(file_name=file_name_db, file_type=file_type_db, project_id=project_id)
            db.add(record)
            db.commit()
            return _resp(201, True, "File processed successfully")
        except Exception:
            logger.exception("Error processing zip file")
            unzip_folder = os.path.join(upload_folder, filename.rsplit(".", 1)[0])
            if os.path.exists(unzip_folder):
                shutil.rmtree(unzip_folder)
            if os.path.exists(file_path):
                os.remove(file_path)
            return _resp(500, False, "An error occurred while processing the file")

    file_name_db = secure_filename(file_wrapper.filename.rsplit(".", 1)[0].lower())
    file_type_db = file_wrapper.filename.rsplit(".", 1)[1].lower()
    logger.info("Saving file: %s", file_name_db)
    record = DataFile(file_name=file_name_db, file_type=file_type_db, project_id=project_id)
    db.add(record)
    db.commit()
    return _resp(201, True, "File saved successfully")


def get_all_files_service(db: Session, project_id: uuid_pkg.UUID | None = None) -> tuple:
    settings = get_settings()
    upload_folder = settings.upload_folder

    try:
        stmt = select(DataFile)
        if project_id is not None:
            stmt = stmt.where(DataFile.project_id == project_id)
        files = db.exec(stmt).all()
        data = []
        for file in files:
            if file.file_type == "zip":
                data.append(
                    {
                        "file_name": file.file_name,
                        "file_type": file.file_type,
                        "file_id": str(file.id),
                        "fields": [],
                    }
                )
            else:
                df = pd.read_csv(f"{upload_folder}/{file.file_name}.{file.file_type}")
                fields = list(df.columns)
                data.append(
                    {
                        "file_name": file.file_name,
                        "file_type": file.file_type,
                        "file_id": str(file.id),
                        "fields": fields,
                    }
                )
        return _resp(200, True, "Saved files found successfully", data)
    except pd.errors.ParserError:
        logger.exception("CSV parsing error")
        return _resp(500, False, "An error occurred while parsing a CSV file")
    except Exception:
        logger.exception("Error fetching files")
        return _resp(500, False, "An error occurred while fetching the files")


def delete_one_file_by_id_service(db: Session, file_id: uuid_pkg.UUID) -> tuple:
    settings = get_settings()
    upload_folder = settings.upload_folder

    try:
        file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
        if not file:
            return _resp(400, False, "File not in the DB")

        file_path = os.path.join(upload_folder, f"{file.file_name}.{file.file_type}")

        if file.file_type == "zip":
            directory_path = os.path.join(upload_folder, file.file_name)
            if os.path.isdir(directory_path):
                shutil.rmtree(directory_path)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                db.delete(file)
                db.commit()
                return _resp(200, True, "Zip file and directory deleted successfully")
            else:
                return _resp(400, False, "Zip file not found")
        else:
            if os.path.isfile(file_path):
                os.remove(file_path)
                db.delete(file)
                db.commit()
                return _resp(200, True, "File deleted successfully")
            else:
                return _resp(400, False, "File not found")
    except Exception:
        logger.exception("Error deleting file")
        return _resp(500, False, "An error occurred while deleting the file")
