from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch
from numpy.typing import NDArray

from experiments.random_imagery.contracts import ModelPrediction, ScoreDiagnostics
from experiments.random_imagery_torch.models import ArchitectureName, SpectralModelShape
from experiments.random_imagery_torch.schemas import SpectralNormalizationState
from preprocessors.config import PreprocessingMethod
from utils.datasets.base import SampleKey


@dataclass(frozen=True, slots=True)
class GroupedTrainingFold:
    fold_index: int
    train_target_indices: NDArray[np.int64]
    validation_target_indices: NDArray[np.int64]
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("Fold index must be non-negative")
        for name, values in (
            ("train_target_indices", self.train_target_indices),
            ("validation_target_indices", self.validation_target_indices),
        ):
            if values.ndim != 1 or values.dtype != np.dtype(np.int64) or values.size < 1:
                raise TypeError(f"`{name}` must be a non-empty int64 vector")
            if not np.array_equal(values, np.unique(values)):
                raise ValueError(f"`{name}` must be sorted and unique")
            values.setflags(write=False)
        if np.intersect1d(
            self.train_target_indices,
            self.validation_target_indices,
        ).size:
            raise ValueError("Grouped fold train and validation rows must be disjoint")
        if set(self.train_subjects) & set(self.validation_subjects):
            raise ValueError("Grouped fold train and validation subjects must be disjoint")


@dataclass(frozen=True, slots=True)
class ValidationEpochRecord:
    epoch: int
    train_bce: float
    validation_bce: float
    validation_balanced_accuracy: float

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError("Epoch numbers are one-based")
        for name, value in (
            ("train_bce", self.train_bce),
            ("validation_bce", self.validation_bce),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"`{name}` must be finite and non-negative")
        if (
            not np.isfinite(self.validation_balanced_accuracy)
            or not 0.0 <= self.validation_balanced_accuracy <= 1.0
        ):
            raise ValueError("Validation balanced accuracy must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class FinalEpochRecord:
    epoch: int
    train_bce: float

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError("Epoch numbers are one-based")
        if not np.isfinite(self.train_bce) or self.train_bce < 0:
            raise ValueError("Training BCE must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ModelCheckpointState:
    epoch: int
    state_dict: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError("Checkpoint epoch must be positive")
        if not self.state_dict:
            raise ValueError("Checkpoint state dict must not be empty")
        validated: dict[str, torch.Tensor] = {}
        for name, tensor in self.state_dict.items():
            if not name or not isinstance(tensor, torch.Tensor):
                raise TypeError("Checkpoint entries require named tensors")
            if tensor.device.type != "cpu" or tensor.requires_grad:
                raise ValueError("Checkpoint tensors must be detached CPU tensors")
            if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                raise ValueError("Checkpoint tensors must be finite")
            validated[name] = tensor.clone()
        object.__setattr__(self, "state_dict", MappingProxyType(validated))


@dataclass(frozen=True, slots=True)
class FoldSelectionResult:
    fold: GroupedTrainingFold
    normalization: SpectralNormalizationState
    positive_weights: NDArray[np.float32]
    history: tuple[ValidationEpochRecord, ...]
    checkpoint: ModelCheckpointState
    stopped_epoch: int

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("Fold selection requires a validation history")
        if self.stopped_epoch != self.history[-1].epoch:
            raise ValueError("Stopped epoch must match the final history record")
        if self.checkpoint.epoch > self.stopped_epoch:
            raise ValueError("Best checkpoint cannot occur after training stopped")
        if self.positive_weights.ndim != 1 or self.positive_weights.dtype != np.dtype(np.float32):
            raise TypeError("Positive weights must be a float32 vector")
        if not np.isfinite(self.positive_weights).all() or np.any(self.positive_weights <= 0):
            raise ValueError("Positive weights must be finite and positive")
        self.positive_weights.setflags(write=False)

    @property
    def best_record(self) -> ValidationEpochRecord:
        return self.history[self.checkpoint.epoch - 1]


@dataclass(frozen=True, slots=True)
class EpochSelectionResult:
    folds: tuple[FoldSelectionResult, FoldSelectionResult, FoldSelectionResult]
    selected_epoch_count: int
    selection_seed: int

    def __post_init__(self) -> None:
        if tuple(result.fold.fold_index for result in self.folds) != (0, 1, 2):
            raise ValueError("Epoch selection requires exactly three ordered folds")
        expected = int(np.median([result.checkpoint.epoch for result in self.folds]))
        if self.selected_epoch_count != expected or self.selected_epoch_count < 1:
            raise ValueError("Selected epoch count must be the median fold-best epoch")


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    seed: int
    history: tuple[FinalEpochRecord, ...]
    checkpoint: ModelCheckpointState

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("Ensemble members require a training history")
        if self.checkpoint.epoch != self.history[-1].epoch:
            raise ValueError("Final checkpoint must match the configured training duration")


@dataclass(frozen=True, slots=True)
class FittedTorchEnsemble:
    architecture: ArchitectureName
    method: PreprocessingMethod
    input_shape: SpectralModelShape
    training_target_indices: NDArray[np.int64]
    training_sample_keys: tuple[SampleKey, ...]
    normalization: SpectralNormalizationState
    positive_weights: NDArray[np.float32]
    selection: EpochSelectionResult
    members: tuple[EnsembleMember, EnsembleMember, EnsembleMember]
    prediction_threshold: float

    def __post_init__(self) -> None:
        if (
            self.training_target_indices.ndim != 1
            or self.training_target_indices.dtype != np.dtype(np.int64)
            or self.training_target_indices.size < 1
        ):
            raise TypeError("Training target indices must be a non-empty int64 vector")
        if not np.array_equal(self.training_target_indices, np.unique(self.training_target_indices)):
            raise ValueError("Training target indices must be sorted and unique")
        if len(self.training_sample_keys) != self.training_target_indices.size:
            raise ValueError("Training sample keys must match training rows")
        if self.normalization.fit_sample_keys != self.training_sample_keys:
            raise ValueError("Final normalization must be fitted on every training sample only")
        if self.positive_weights.shape != (36,) or self.positive_weights.dtype != np.dtype(np.float32):
            raise TypeError("Final positive weights must be a float32 36-pixel vector")
        if len({member.seed for member in self.members}) != 3:
            raise ValueError("Ensemble members require three unique seeds")
        if any(
            member.checkpoint.epoch != self.selection.selected_epoch_count
            for member in self.members
        ):
            raise ValueError("Every ensemble member must use the selected epoch count")
        if not 0.0 < self.prediction_threshold < 1.0:
            raise ValueError("Prediction threshold must be between zero and one")
        self.training_target_indices.setflags(write=False)
        self.positive_weights.setflags(write=False)


@dataclass(frozen=True, slots=True)
class TorchEnsemblePrediction:
    test_target_indices: NDArray[np.int64]
    test_sample_keys: tuple[SampleKey, ...]
    member_seeds: tuple[int, int, int]
    member_scores: NDArray[np.float64]
    scores: NDArray[np.float64]
    predictions: NDArray[np.int8]
    threshold: float

    def __post_init__(self) -> None:
        n_rows = self.test_target_indices.size
        if (
            self.test_target_indices.shape != (n_rows,)
            or self.test_target_indices.dtype != np.dtype(np.int64)
        ):
            raise TypeError("Test target indices must be an int64 vector")
        if len(self.test_sample_keys) != n_rows or len(set(self.member_seeds)) != 3:
            raise ValueError("Prediction metadata does not match rows or ensemble seeds")
        if (
            self.member_scores.shape != (3, n_rows, 36)
            or self.member_scores.dtype != np.dtype(np.float64)
        ):
            raise TypeError("Member scores must have shape (3, sample, 36) and use float64")
        if self.scores.shape != (n_rows, 36) or self.scores.dtype != np.dtype(np.float64):
            raise TypeError("Ensemble scores must have shape (sample, 36) and use float64")
        if not np.isfinite(self.member_scores).all() or np.any(
            (self.member_scores < 0.0) | (self.member_scores > 1.0)
        ):
            raise ValueError("Member scores must be finite probabilities")
        if not np.array_equal(self.scores, self.member_scores.mean(axis=0, dtype=np.float64)):
            raise ValueError("Ensemble scores must equal the mean member probability")
        if (
            self.predictions.shape != self.scores.shape
            or self.predictions.dtype != np.dtype(np.int8)
            or not np.array_equal(
                self.predictions,
                (self.scores >= self.threshold).astype(np.int8),
            )
        ):
            raise TypeError("Predictions must be threshold-derived int8 labels")
        for array in (
            self.test_target_indices,
            self.member_scores,
            self.scores,
            self.predictions,
        ):
            array.setflags(write=False)

    def to_model_prediction(self) -> ModelPrediction:
        return ModelPrediction(
            test_target_indices=self.test_target_indices,
            test_sample_keys=self.test_sample_keys,
            scores=self.scores,
            predictions=self.predictions,
            threshold=self.threshold,
            diagnostics=ScoreDiagnostics(score_semantics="native_probability"),
        )
