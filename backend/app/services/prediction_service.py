from __future__ import annotations

from .efta_service import run_efta
from .model_service import get_model_service


def predict(request) -> dict:
    service = get_model_service(request.model_name)
    scaled = service.transform_raw(request.features)
    result = run_efta(
        service,
        scaled,
        request.confidence_threshold,
        request.faithfulness_drop_threshold,
        request.stability_threshold,
    )
    result["model_name"] = request.model_name
    return result
