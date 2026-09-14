from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 raw Wisconsin Diagnostic Breast Cancer features in dataset order.",
    )
    model_name: str = Field(default="logistic_regression")
    confidence_threshold: float | None = Field(default=None, ge=0.5, le=1.0)
    faithfulness_drop_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    stability_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("features")
    @classmethod
    def finite_features(cls, values: list[float]) -> list[float]:
        if any(not isinstance(value, (int, float)) or value != value or abs(value) == float("inf") for value in values):
            raise ValueError("All feature values must be finite numbers.")
        return [float(value) for value in values]


class GateResult(BaseModel):
    passed: bool
    score: float | None = None
    threshold: float | None = None
    reason: str


class PredictionResponse(BaseModel):
    model_name: str
    prediction: int
    prediction_label: str
    probability: float
    probability_class: int
    probability_class_label: str
    malignant_probability: float
    benign_probability: float
    confidence: float
    gates: dict[str, GateResult]
    overall_decision: str
    explanation: dict[str, object]


class ModelInfo(BaseModel):
    model_name: str
    dataset: str
    feature_count: int
    feature_names: list[str]
    class_names: list[str]
    training_seed: int
    preprocessing: str


class GateInfo(BaseModel):
    name: str
    description: str
    pass_condition: str


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
