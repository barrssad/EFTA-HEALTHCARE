from fastapi import APIRouter, HTTPException

from ..schemas.experiments import ExperimentResponse
from ..services.experiment_service import get_experiment_results

router = APIRouter(prefix="/api", tags=["experiments"])


@router.get("/experiments", response_model=ExperimentResponse)
def experiments() -> ExperimentResponse:
    try:
        return ExperimentResponse(**get_experiment_results())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Experiment results unavailable: {exc}") from exc
