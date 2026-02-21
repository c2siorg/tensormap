import uuid as uuid_pkg

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.database import get_db
from app.schemas.project import ProjectCreateRequest, ProjectUpdateRequest
from app.services.project import (
    create_project_service,
    delete_project_service,
    get_all_projects_service,
    get_project_by_id_service,
    update_project_service,
)

router = APIRouter(tags=["project"])


@router.post("/project")
async def create_project(request: ProjectCreateRequest, db: Session = Depends(get_db)):
    body, status_code = create_project_service(db, data=request)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/project")
async def get_all_projects(db: Session = Depends(get_db)):
    body, status_code = get_all_projects_service(db)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/project/{project_id}")
async def get_project(project_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    body, status_code = get_project_by_id_service(db, project_id=project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.patch("/project/{project_id}")
async def update_project(project_id: uuid_pkg.UUID, request: ProjectUpdateRequest, db: Session = Depends(get_db)):
    body, status_code = update_project_service(db, project_id=project_id, data=request)
    return JSONResponse(status_code=status_code, content=body)


@router.delete("/project/{project_id}")
async def delete_project(project_id: uuid_pkg.UUID, db: Session = Depends(get_db)):
    body, status_code = delete_project_service(db, project_id=project_id)
    return JSONResponse(status_code=status_code, content=body)
