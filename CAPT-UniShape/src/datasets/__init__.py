"""PEMFC fault-classification dataset, transforms, and collation utilities."""

from .pemfc_dataset import PEMFCDataset
from .csv_dataset import CSVFuelCellDataset, build_csv_datasets
from .feature_engineering import engineer_features, normalize_columns
from .transforms import (
    Compose,
    EISFreqMask,
    GaussianNoise,
    ModalityDropout,
    Normalize,
    RandomMissing,
    TimeWarp,
    WindowCrop,
)
from .collate import pemfc_collate_fn

__all__ = [
    "PEMFCDataset",
    "CSVFuelCellDataset",
    "build_csv_datasets",
    "engineer_features",
    "normalize_columns",
    "Compose",
    "EISFreqMask",
    "GaussianNoise",
    "ModalityDropout",
    "Normalize",
    "RandomMissing",
    "TimeWarp",
    "WindowCrop",
    "pemfc_collate_fn",
]
