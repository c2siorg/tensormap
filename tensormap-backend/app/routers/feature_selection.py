import uuid as uuid_pkg

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.database import get_db
from app.schemas.feature_selection import (
    ApplyFeatureSelectionRequest,
    ApplyFeatureSelectionResponse,
    FeatureSelectionAnalyzeRequest,
    FeatureSelectionAnalyzeResponse,
)
from app.services.feature_selection_service import feature_selection_service
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["feature-selection"])


@router.post(
    "/data/process/feature-selection/{file_id}/analyze",
    response_model=FeatureSelectionAnalyzeResponse,
    status_code=200,
)
def analyze_feature_selection(
    file_id: uuid_pkg.UUID,
    request: FeatureSelectionAnalyzeRequest = FeatureSelectionAnalyzeRequest(),
    db: Session = Depends(get_db),
) -> FeatureSelectionAnalyzeResponse:
    """Analyze feature relevance, mutual information, ANOVA F-scores, and variance

    for all candidate predictive features in a dataset.
    """
    logger.info("Analyzing feature relevance for file_id=%s, target=%s", file_id, request.target_field)
    return feature_selection_service.analyze_feature_relevance(db, file_id, request)


@router.post(
    "/data/process/feature-selection/{file_id}/apply",
    response_model=ApplyFeatureSelectionResponse,
    status_code=200,
)
def apply_feature_selection(
    file_id: uuid_pkg.UUID,
    request: ApplyFeatureSelectionRequest,
    db: Session = Depends(get_db),
) -> ApplyFeatureSelectionResponse:
    """Prunes a dataset CSV to keep only the chosen features plus the target column,

    updating the database record and creating a restorable .bak backup.
    """
    logger.info("Applying feature selection pruning to file_id=%s", file_id)
    return feature_selection_service.apply_feature_selection(db, file_id, request)
