"""Feature relevance profiling and selection endpoints."""

import uuid as uuid_pkg

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.database import get_db
from app.schemas.feature_selection import FeatureAnalysisRequest, FeatureSelectionApplyRequest
from app.services.feature_selection import analyze_features, apply_feature_selection

router = APIRouter(tags=["feature-selection"])


@router.post("/data/process/feature-selection/{file_id}/analyze")
def analyze_feature_selection(
    file_id: uuid_pkg.UUID,
    request: FeatureAnalysisRequest,
    db: Session = Depends(get_db),
):
    """Return target-relative relevance scores for the dataset's predictor columns."""
    body, status_code = analyze_features(db, file_id, request.target, request.task)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/data/process/feature-selection/{file_id}/apply")
def apply_feature_selection_route(
    file_id: uuid_pkg.UUID,
    request: FeatureSelectionApplyRequest,
    db: Session = Depends(get_db),
):
    """Remove selected predictor columns after creating a recoverable backup."""
    body, status_code = apply_feature_selection(db, file_id, request.target, request.columns_to_drop)
    return JSONResponse(status_code=status_code, content=body)
