from typing import Any

from utils.datasets.base import DatasetBase
from utils.datasets.numpy_dataset import NumpyDataset
from utils.datasets.preprocessed import FFTDataset, MorletDataset, PreprocessedDataset, STFTDataset, SuperletDataset
from utils.datasets.schemas import (
    CacheWarmupError,
    CacheWarmupReport,
    GeometricSample,
    LoadedSample,
    RandomSample,
    Sample,
    SpectralSample,
    TorchSample,
    TorchSampleBatch,
    TorchSpectralBatch,
    TorchSpectralSample,
)
from utils.datasets.torch_dataset import TorchDataset, collate_torch_samples
from utils.datasets.torch_preprocessed import TorchPreprocessedDataset, collate_torch_spectral_samples

__all__ = [
    "CacheWarmupError",
    "CacheWarmupReport",
    "DatasetBase",
    "FeatureDataset",
    "FFTDataset",
    "MorletDataset",
    "NumpyDataset",
    "PreprocessedDataset",
    "STFTDataset",
    "SuperletDataset",
    "LoadedSample",
    "SpectralSample",
    "TorchSample",
    "TorchSampleBatch",
    "TorchDataset",
    "TorchPreprocessedDataset",
    "TorchSpectralBatch",
    "TorchSpectralSample",
    "collate_torch_samples",
    "collate_torch_spectral_samples",
    "Sample",
    "GeometricSample",
    "RandomSample",
]


def __getattr__(name: str) -> Any:
    if name == "FeatureDataset":
        from utils.datasets.feature_dataset import FeatureDataset

        return FeatureDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
