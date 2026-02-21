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
    """Build a standard API response tuple of (body_dict, status_code)."""
    return {"success": success, "message": message, "data": data}, status_code


def create_project_service(db: Session, data: ProjectCreateRequest) -> tuple:
    """Create a new project and return its details."""
    try:
        project = Project(name=data.name, description=data.description)
        db.add(project)
        db.commit()
        db.refresh(project)
        logger.info("Project created: id=%s, name=%s", project.id, project.name)
        return _resp(
            201,
            True,
            "Project created successfully",
            {
                "id": str(project.id),
                "name": project.name,
                "description": project.description,
                "created_on": project.created_on.isoformat() if project.created_on else None,
                "updated_on": project.updated_on.isoformat() if project.updated_on else None,
            },
        )
    except Exception:
        db.rollback()
        logger.exception("Error creating project")
        return _resp(500, False, "An error occurred while creating the project")


def get_all_projects_service(db: Session, offset: int = 0, limit: int = 50) -> tuple:
    """Return a paginated list of projects with aggregated file and model counts."""
    try:
        total = db.exec(select(func.count()).select_from(Project)).one()

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
            .offset(offset)
            .limit(limit)
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
        body = {"success": True, "message": "Projects retrieved successfully", "data": data}
        body["pagination"] = {"total": total, "offset": offset, "limit": limit}
        return body, 200
    except Exception:
        logger.exception("Error fetching projects")
        return _resp(500, False, "An error occurred while fetching projects")


def get_project_by_id_service(db: Session, project_id: uuid_pkg.UUID) -> tuple:
    """Fetch a single project by its UUID."""
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
    """Partially update a project's fields."""
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
        logger.info("Project updated: id=%s", project_id)
        return _resp(
            200,
            True,
            "Project updated successfully",
            {
                "id": str(project.id),
                "name": project.name,
                "description": project.description,
                "created_on": project.created_on.isoformat() if project.created_on else None,
                "updated_on": project.updated_on.isoformat() if project.updated_on else None,
            },
        )
    except Exception:
        db.rollback()
        logger.exception("Error updating project")
        return _resp(500, False, "An error occurred while updating the project")


def delete_project_service(db: Session, project_id: uuid_pkg.UUID) -> tuple:
    """Delete a project and cascade-remove associated files and models."""
    try:
        project = db.get(Project, project_id)
        if not project:
            return _resp(404, False, "Project not found")

        db.delete(project)
        db.commit()
        logger.info("Project deleted: id=%s", project_id)
        return _resp(200, True, "Project deleted successfully")
    except Exception:
        db.rollback()
        logger.exception("Error deleting project")
        return _resp(500, False, "An error occurred while deleting the project")
