from fastapi import APIRouter, HTTPException

from ..schemas.prediction import GateInfo, ModelInfo
from ..services.model_service import get_model_service

router = APIRouter(prefix="/api", tags=["model"])


@router.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    try:
        return ModelInfo(**get_model_service().metadata())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@router.get("/gates", response_model=list[GateInfo])
def gates() -> list[GateInfo]:
    return [
        GateInfo(name="G1", description="Local Uncertainty Check", pass_condition="Highest predicted class probability >= confidence threshold."),
        GateInfo(name="G2", description="Explanation Faithfulness Check", pass_condition="Masking top-k SHAP features lowers the class-1 (benign) probability by >= drop threshold."),
        GateInfo(name="G3", description="Explanation Stability Check", pass_condition="Mean pairwise Jaccard overlap across 20 noisy explanations >= stability threshold."),
    ]
