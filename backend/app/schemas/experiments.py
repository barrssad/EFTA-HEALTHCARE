from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ExperimentResponse(BaseModel):
    available: bool
    source: str
    generated_by: str
    methodology: dict[str, Any]
    summary: list[dict[str, Any]]
    raw_results: list[dict[str, Any]]
    raw_result_count: int
    message: str | None = None
