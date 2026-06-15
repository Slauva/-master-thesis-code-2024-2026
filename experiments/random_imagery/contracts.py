from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from experiments.logistic_regression.schemas import (
    AlignedFeaturePartition,
    AlignedTrainingFeatures,
    PixelTargetDataset,
)
from experiments.random_imagery.config import RandomImageryExperimentConfigLike
from experiments.random_imagery.registry import ModelSpec


@dataclass(frozen=True, slots=True)
class ScoreDiagnostics:
    score_semantics: str
    clipped_below_zero_fraction: float = 0.0
    clipped_above_one_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.score_semantics not in {
            "native_probability",
            "calibrated_probability",
            "clipped_regression",
        }:
            raise ValueError(f"Unsupported score semantics: {self.score_semantics!r}")
        for name, value in (
            ("clipped_below_zero_fraction", self.clipped_below_zero_fraction),
            ("clipped_above_one_fraction", self.clipped_above_one_fraction),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"`{name}` must be finite and in [0, 1]")
        if (
            self.score_semantics != "clipped_regression"
            and (
                self.clipped_below_zero_fraction != 0.0
                or self.clipped_above_one_fraction != 0.0
            )
        ):
            raise ValueError("Only clipped regression scores may report clipping fractions")


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    test_target_indices: NDArray[np.int64]
    test_sample_keys: tuple[tuple[int, int, int], ...]
    scores: NDArray[np.float64]
    predictions: NDArray[np.int8]
    threshold: float
    diagnostics: ScoreDiagnostics

    def __post_init__(self) -> None:
        n_rows = self.test_target_indices.size
        if (
            self.test_target_indices.shape != (n_rows,)
            or self.test_target_indices.dtype != np.dtype(np.int64)
        ):
            raise TypeError("Test target indices must be a one-dimensional int64 array")
        if len(self.test_sample_keys) != n_rows:
            raise ValueError("Test sample keys must match test rows")
        if self.scores.ndim != 2 or self.scores.shape[0] != n_rows:
            raise ValueError("Model scores must have shape (test row, target)")
        if self.scores.dtype != np.dtype(np.float64):
            raise TypeError("Model scores must use float64")
        if not np.isfinite(self.scores).all() or np.any(
            (self.scores < 0.0) | (self.scores > 1.0)
        ):
            raise ValueError("Model scores must be finite and in [0, 1]")
        if (
            self.predictions.shape != self.scores.shape
            or self.predictions.dtype != np.dtype(np.int8)
        ):
            raise TypeError("Predictions must be an int8 matrix matching scores")
        if not np.isin(self.predictions, (0, 1)).all():
            raise ValueError("Predictions must be binary")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("Prediction threshold must be between zero and one")
        expected = (self.scores >= self.threshold).astype(np.int8)
        if not np.array_equal(self.predictions, expected):
            raise ValueError("Predictions must equal thresholded model scores")


PayloadT = TypeVar("PayloadT")
SelectionT = TypeVar("SelectionT")


@dataclass(frozen=True, slots=True)
class FittedDirectionModel(Generic[PayloadT, SelectionT]):
    spec: ModelSpec
    selected_block_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    training_target_indices: NDArray[np.int64]
    training_sample_keys: tuple[tuple[int, int, int], ...]
    payload: PayloadT
    selection: SelectionT

    def __post_init__(self) -> None:
        if not self.selected_block_names or not self.feature_names:
            raise ValueError("A fitted direction model requires a feature family and features")
        if (
            self.training_target_indices.ndim != 1
            or self.training_target_indices.dtype != np.dtype(np.int64)
        ):
            raise TypeError("Training target indices must be a one-dimensional int64 array")
        if len(self.training_sample_keys) != self.training_target_indices.size:
            raise ValueError("Training sample keys must match training target indices")


class RandomImageryModelBackend(Protocol[PayloadT, SelectionT]):
    spec: ModelSpec

    def fit(
        self,
        training_features: AlignedTrainingFeatures,
        *,
        targets: PixelTargetDataset,
        config: RandomImageryExperimentConfigLike,
    ) -> FittedDirectionModel[PayloadT, SelectionT]: ...

    def predict(
        self,
        fitted: FittedDirectionModel[PayloadT, SelectionT],
        *,
        test_features: AlignedFeaturePartition,
        targets: PixelTargetDataset,
        config: RandomImageryExperimentConfigLike,
    ) -> ModelPrediction: ...


AnyFittedDirectionModel = FittedDirectionModel[Any, Any]
AnyRandomImageryModelBackend = RandomImageryModelBackend[Any, Any]


__all__ = [
    "AnyFittedDirectionModel",
    "AnyRandomImageryModelBackend",
    "FittedDirectionModel",
    "ModelPrediction",
    "RandomImageryModelBackend",
    "ScoreDiagnostics",
]
