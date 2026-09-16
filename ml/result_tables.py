"""Provenance-preserving result tables and figure data for EFTA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class ResultContext:
    dataset: str
    model: str
    seed: int
    split: str
    policy: str
    shift_condition: str
    coverage: float | None
    denominator: int | None
    threshold_id: str
    config_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_CONTEXT = tuple(field.name for field in fields(ResultContext))


def _with_context(rows: Iterable[dict[str, Any]], context: ResultContext) -> pd.DataFrame:
    records = []
    for row in rows:
        record = context.as_dict()
        record.update(row)
        records.append(record)
    columns = list(REQUIRED_CONTEXT)
    for record in records:
        columns.extend(key for key in record if key not in columns)
    return pd.DataFrame(records, columns=columns)


def table1_design(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Build Table 1 from declarative dataset/model/explainer design rows."""
    return pd.DataFrame(list(rows))


def table2_metrics(metrics: dict[str, Any], context: ResultContext) -> pd.DataFrame:
    """Build a long metric table while retaining complete result context."""
    excluded = {"ece_detail", "risk_coverage_curve"}
    rows = [
        {"metric": key, "value": value}
        for key, value in metrics.items()
        if key not in excluded and not isinstance(value, (list, dict, tuple))
    ]
    return _with_context(rows, context)


def table3_gate_ablation(rows: Iterable[dict[str, Any]], context: ResultContext) -> pd.DataFrame:
    """Build the gate-ablation and failure-reason panel."""
    return _with_context(rows, context)


def risk_coverage_data(curve: Iterable[dict[str, Any]], context: ResultContext) -> pd.DataFrame:
    """Build Figure 2 data with one row per risk-coverage point."""
    return _with_context(curve, context)


def architecture_decision_data(rows: Iterable[dict[str, Any]], context: ResultContext) -> pd.DataFrame:
    """Build Figure 1 architecture/decision data."""
    return _with_context(rows, context)


def export_table(table: pd.DataFrame, path: str | Path, *, markdown: bool = False) -> Path:
    """Export a result table as CSV or Markdown without changing values."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if markdown or destination.suffix.lower() in {".md", ".markdown"}:
        destination.write_text(table.to_markdown(index=False), encoding="utf-8")
    else:
        table.to_csv(destination, index=False)
    return destination