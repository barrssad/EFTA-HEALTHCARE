from .unified_interface import DatasetBundle, load_dataset
from .split_protocol import PreparedSplits, get_prepared_splits
from .dataset_registry import DATASET_REGISTRY, get_metadata


def __getattr__(name: str):
    if name in {"audit_dataset", "audit_all_datasets"}:
        from .audit import audit_all_datasets, audit_dataset

        return {"audit_dataset": audit_dataset, "audit_all_datasets": audit_all_datasets}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DatasetBundle",
    "load_dataset",
    "PreparedSplits",
    "get_prepared_splits",
    "audit_dataset",
    "audit_all_datasets",
    "DATASET_REGISTRY",
    "get_metadata",
]
