from fastapi import APIRouter

from ..schemas.prediction import HealthResponse
from ..services.model_service import get_model_service

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_model_service()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(status="ok" if loaded else "degraded", service="EFTA API", model_loaded=loaded)
