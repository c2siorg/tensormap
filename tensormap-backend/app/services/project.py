import uuid as uuid_pkg
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlmodel import Session, select

from app.models.data import DataFile
from app.models.ml import ModelBasic
from app.models.project import Project
from app.schemas.project import ProjectCreateRequest, ProjectUpdateRequest
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def _resp(status_code: int, success: bool, message: str, data: Any = None) -> tuple:
    return {"success": success, "message": message, "data": data}, status_code


def create_project_service(db: Session, data: ProjectCreateRequest) -> tuple:
    try:
        project = Project(name=data.name, description=data.description)
        db.add(project)
        db.commit()
        db.refresh(project)
        return _resp(
            201,
            True,
            "Project created successfully",
            {"id": str(project.id), "name": project.name, "description": project.description},
        )
    except Exception:
        db.rollback()
        logger.exception("Error creating project")
        return _resp(500, False, "An error occurred while creating the project")


def get_all_projects_service(db: Session) -> tuple:
    try:
        stmt = (
            select(
                Project.id,
                Project.name,
                Project.description,
                Project.created_on,
                Project.updated_on,
                func.count(func.distinct(DataFile.id)).label("file_count"),
                func.count(func.distinct(ModelBasic.id)).label("model_count"),
            )
            .outerjoin(DataFile, DataFile.project_id == Project.id)
            .outerjoin(ModelBasic, ModelBasic.project_id == Project.id)
            .group_by(Project.id)
            .order_by(Project.updated_on.desc())
        )
        rows = db.exec(stmt).all()
        data = [
            {
                "id": str(row.id),
                "name": row.name,
                "description": row.description,
                "created_on": row.created_on.isoformat() if row.created_on else None,
                "updated_on": row.updated_on.isoformat() if row.updated_on else None,
                "file_count": row.file_count,
                "model_count": row.model_count,
            }
            for row in rows
        ]
        return _resp(200, True, "Projects retrieved successfully", data)
    except Exception:
        logger.exception("Error fetching projects")
        return _resp(500, False, "An error occurred while fetching projects")


def get_project_by_id_service(db: Session, project_id: uuid_pkg.UUID) -> tuple:
    try:
        project = db.get(Project, project_id)
        if not project:
            return _resp(404, False, "Project not found")
        return _resp(
            200,
            True,
            "Project retrieved successfully",
            {
                "id": str(project.id),
                "name": project.name,
                "description": project.description,
                "created_on": project.created_on.isoformat() if project.created_on else None,
                "updated_on": project.updated_on.isoformat() if project.updated_on else None,
            },
        )
    except Exception:
        logger.exception("Error fetching project")
        return _resp(500, False, "An error occurred while fetching the project")


def update_project_service(db: Session, project_id: uuid_pkg.UUID, data: ProjectUpdateRequest) -> tuple:
    try:
        project = db.get(Project, project_id)
        if not project:
            return _resp(404, False, "Project not found")

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(project, key, value)
        project.updated_on = datetime.utcnow()

        db.add(project)
        db.commit()
        db.refresh(project)
        return _resp(
            200,
            True,
            "Project updated successfully",
            {"id": str(project.id), "name": project.name, "description": project.description},
        )
    except Exception:
        db.rollback()
        logger.exception("Error updating project")
        return _resp(500, False, "An error occurred while updating the project")


def delete_project_service(db: Session, project_id: uuid_pkg.UUID) -> tuple:
    try:
        project = db.get(Project, project_id)
        if not project:
            return _resp(404, False, "Project not found")

        db.delete(project)
        db.commit()
        return _resp(200, True, "Project deleted successfully")
    except Exception:
        db.rollback()
        logger.exception("Error deleting project")
        return _resp(500, False, "An error occurred while deleting the project")
