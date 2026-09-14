from fastapi import APIRouter, HTTPException

from ..schemas.prediction import PredictionRequest, PredictionResponse
from ..services.prediction_service import predict

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def create_prediction(request: PredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**predict(request))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
