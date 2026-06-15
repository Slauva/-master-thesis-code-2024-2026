from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from features.schemas import SampleKey

EvaluationProtocol: TypeAlias = Literal["cross-subject", "within-subject"]
EvaluationDirectionName: TypeAlias = Literal[
    "cross-subject",
    "trial-1-to-trial-2",
    "trial-2-to-trial-1",
]


@dataclass(frozen=True, slots=True)
class PixelTargetDataset:
    y: NDArray[np.int8]
    pixel_names: tuple[str, ...]
    sample_keys: tuple[SampleKey, ...]
    subject_ids: NDArray[np.int64]
    trial_numbers: NDArray[np.int64]
    block_indices: NDArray[np.int64]
    seeds: NDArray[np.int64]
    image_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.y.ndim != 2:
            raise ValueError("Pixel targets must have shape (sample, pixel)")
        if self.y.dtype != np.dtype(np.int8):
            raise TypeError("Pixel targets must use int8")
        if not np.isin(self.y, (0, 1)).all():
            raise ValueError("Pixel targets must be binary")
        if len(self.pixel_names) != self.y.shape[1]:
            raise ValueError("Pixel names must match the target columns")
        if len(set(self.pixel_names)) != len(self.pixel_names):
            raise ValueError("Pixel names must be unique")

        n_samples = self.y.shape[0]
        if len(self.sample_keys) != n_samples or len(set(self.sample_keys)) != n_samples:
            raise ValueError("Sample keys must be unique and match the target rows")
        for name, values in (
            ("subject_ids", self.subject_ids),
            ("trial_numbers", self.trial_numbers),
            ("block_indices", self.block_indices),
            ("seeds", self.seeds),
        ):
            if values.shape != (n_samples,) or values.dtype != np.dtype(np.int64):
                raise TypeError(f"`{name}` must be an int64 vector matching the target rows")
        if len(self.image_fingerprints) != n_samples:
            raise ValueError("Image fingerprints must match the target rows")


@dataclass(frozen=True, slots=True)
class SubjectSplit:
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    train_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]
    n_samples: int
    random_state: int
    test_size: float

    def __post_init__(self) -> None:
        for name, indices in (("train_indices", self.train_indices), ("test_indices", self.test_indices)):
            if indices.ndim != 1 or indices.dtype != np.dtype(np.int64):
                raise TypeError(f"`{name}` must be a one-dimensional int64 array")
            if np.any(indices < 0) or np.any(indices >= self.n_samples):
                raise ValueError(f"`{name}` contains an out-of-range row")
            if not np.array_equal(indices, np.unique(indices)):
                raise ValueError(f"`{name}` must be sorted and unique")
        if np.intersect1d(self.train_indices, self.test_indices).size:
            raise ValueError("Train and test row indices must be disjoint")
        if np.union1d(self.train_indices, self.test_indices).size != self.n_samples:
            raise ValueError("Train and test indices must partition every target row")
        if set(self.train_subjects) & set(self.test_subjects):
            raise ValueError("Train and test subjects must be disjoint")


@dataclass(frozen=True, slots=True)
class EvaluationDirection:
    protocol: EvaluationProtocol
    name: EvaluationDirectionName
    label: str
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    train_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]
    eligible_subjects: tuple[int, ...]
    excluded_subjects: tuple[int, ...]
    n_samples: int
    train_trial: int | None = None
    test_trial: int | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Evaluation direction label must not be empty")
        for name, indices in (
            ("train_indices", self.train_indices),
            ("test_indices", self.test_indices),
        ):
            if indices.ndim != 1 or indices.dtype != np.dtype(np.int64):
                raise TypeError(f"`{name}` must be a one-dimensional int64 array")
            if indices.size < 1 or np.any(indices < 0) or np.any(indices >= self.n_samples):
                raise ValueError(f"`{name}` must contain valid rows")
            if not np.array_equal(indices, np.unique(indices)):
                raise ValueError(f"`{name}` must be sorted and unique")
        if np.intersect1d(self.train_indices, self.test_indices).size:
            raise ValueError("Direction train and test rows must be disjoint")

        for name, subjects in (
            ("train_subjects", self.train_subjects),
            ("test_subjects", self.test_subjects),
            ("eligible_subjects", self.eligible_subjects),
            ("excluded_subjects", self.excluded_subjects),
        ):
            if subjects != tuple(sorted(set(subjects))):
                raise ValueError(f"`{name}` must be sorted and unique")
        if not self.eligible_subjects:
            raise ValueError("Evaluation direction requires eligible subjects")
        if set(self.eligible_subjects) & set(self.excluded_subjects):
            raise ValueError("Eligible and excluded subjects must be disjoint")
        if not set(self.train_subjects) <= set(self.eligible_subjects):
            raise ValueError("Train subjects must be eligible")
        if not set(self.test_subjects) <= set(self.eligible_subjects):
            raise ValueError("Test subjects must be eligible")

        if self.protocol == "cross-subject":
            if self.name != "cross-subject":
                raise ValueError("Cross-subject protocol requires the cross-subject direction")
            if self.train_trial is not None or self.test_trial is not None:
                raise ValueError("Cross-subject direction must not pin trial numbers")
            if set(self.train_subjects) & set(self.test_subjects):
                raise ValueError("Cross-subject train and test identities must be disjoint")
            if set(self.train_subjects) | set(self.test_subjects) != set(self.eligible_subjects):
                raise ValueError("Cross-subject direction must partition eligible identities")
            if np.union1d(self.train_indices, self.test_indices).size != self.n_samples:
                raise ValueError("Cross-subject direction must partition every target row")
        else:
            expected_trials = {
                "trial-1-to-trial-2": (1, 2),
                "trial-2-to-trial-1": (2, 1),
            }
            if self.name not in expected_trials:
                raise ValueError("Within-subject protocol requires a cross-trial direction")
            if (self.train_trial, self.test_trial) != expected_trials[self.name]:
                raise ValueError("Cross-trial direction name and trial numbers disagree")
            if self.train_subjects != self.eligible_subjects:
                raise ValueError("Every eligible identity must occur in cross-trial training")
            if self.test_subjects != self.eligible_subjects:
                raise ValueError("Every eligible identity must occur in cross-trial testing")


@dataclass(frozen=True, slots=True)
class ProtocolLeakageAudit:
    protocol: EvaluationProtocol
    direction_name: EvaluationDirectionName
    overlapping_subjects: tuple[int, ...]
    overlapping_sample_keys: tuple[SampleKey, ...]
    overlapping_seeds: tuple[int, ...]
    overlapping_image_fingerprints: tuple[str, ...]
    overlapping_trial_numbers: tuple[int, ...]
    train_positive_counts: NDArray[np.int64]
    test_positive_counts: NDArray[np.int64]
    all_tasks_have_both_classes: bool
    subject_contract_satisfied: bool
    trial_contract_satisfied: bool

    def __post_init__(self) -> None:
        if self.train_positive_counts.ndim != 1 or self.train_positive_counts.dtype != np.dtype(np.int64):
            raise TypeError("Train positive counts must be a one-dimensional int64 array")
        if (
            self.test_positive_counts.shape != self.train_positive_counts.shape
            or self.test_positive_counts.dtype != np.dtype(np.int64)
        ):
            raise TypeError("Test positive counts must match the train pixel axis")

    @property
    def has_forbidden_leakage(self) -> bool:
        return bool(
            self.overlapping_sample_keys
            or self.overlapping_seeds
            or self.overlapping_image_fingerprints
            or not self.subject_contract_satisfied
            or not self.trial_contract_satisfied
        )


@dataclass(frozen=True, slots=True)
class EvaluationProtocolDefinition:
    protocol: EvaluationProtocol
    label: str
    eligible_subjects: tuple[int, ...]
    excluded_subjects: tuple[int, ...]
    directions: tuple[EvaluationDirection, ...]
    audits: tuple[ProtocolLeakageAudit, ...]

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Evaluation protocol label must not be empty")
        if not self.directions or len(self.directions) != len(self.audits):
            raise ValueError("Evaluation protocol requires one audit per direction")
        if any(direction.protocol != self.protocol for direction in self.directions):
            raise ValueError("Every direction must belong to the declared protocol")
        if any(audit.protocol != self.protocol for audit in self.audits):
            raise ValueError("Every audit must belong to the declared protocol")
        if tuple(direction.name for direction in self.directions) != tuple(
            audit.direction_name for audit in self.audits
        ):
            raise ValueError("Direction and audit order must match")
        if any(direction.eligible_subjects != self.eligible_subjects for direction in self.directions):
            raise ValueError("Directions must share protocol eligibility")
        if any(direction.excluded_subjects != self.excluded_subjects for direction in self.directions):
            raise ValueError("Directions must share excluded-subject provenance")
        expected_names = (
            ("cross-subject",)
            if self.protocol == "cross-subject"
            else ("trial-1-to-trial-2", "trial-2-to-trial-1")
        )
        if tuple(direction.name for direction in self.directions) != expected_names:
            raise ValueError("Evaluation protocol directions are incomplete or out of order")


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    overlapping_subjects: tuple[int, ...]
    overlapping_sample_keys: tuple[SampleKey, ...]
    overlapping_seeds: tuple[int, ...]
    overlapping_image_fingerprints: tuple[str, ...]
    train_positive_counts: NDArray[np.int64]
    test_positive_counts: NDArray[np.int64]
    all_tasks_have_both_classes: bool

    @property
    def has_leakage(self) -> bool:
        return any(
            (
                self.overlapping_subjects,
                self.overlapping_sample_keys,
                self.overlapping_seeds,
                self.overlapping_image_fingerprints,
            )
        )


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    name: str
    probabilities: NDArray[np.float64]
    predictions: NDArray[np.int8]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Baseline name must not be empty")
        if self.probabilities.ndim != 2 or self.predictions.shape != self.probabilities.shape:
            raise ValueError("Baseline probabilities and predictions must share shape (sample, pixel)")
        if self.probabilities.dtype != np.dtype(np.float64):
            raise TypeError("Baseline probabilities must use float64")
        if self.predictions.dtype != np.dtype(np.int8):
            raise TypeError("Baseline predictions must use int8")
        if not np.isfinite(self.probabilities).all():
            raise ValueError("Baseline probabilities must be finite")
        if np.any((self.probabilities < 0.0) | (self.probabilities > 1.0)):
            raise ValueError("Baseline probabilities must be in [0, 1]")
        if not np.isin(self.predictions, (0, 1)).all():
            raise ValueError("Baseline predictions must be binary")


@dataclass(frozen=True, slots=True)
class FeatureFamily:
    block_names: tuple[str, ...]
    X: NDArray[np.floating]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.block_names or len(set(self.block_names)) != len(self.block_names):
            raise ValueError("Feature-family block names must be non-empty and unique")
        if self.X.ndim != 2 or not np.issubdtype(self.X.dtype, np.floating):
            raise TypeError("Feature-family values must be a two-dimensional floating array")
        if not np.isfinite(self.X).all():
            raise ValueError("Feature-family values must be finite")
        if self.X.shape[1] < 1 or len(self.feature_names) != self.X.shape[1]:
            raise ValueError("Feature names must match a non-empty feature axis")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names must be unique")

    @property
    def name(self) -> str:
        return "+".join(self.block_names)


@dataclass(frozen=True, slots=True)
class AlignedTrainingFeatures:
    families: tuple[FeatureFamily, ...]
    target_row_indices: NDArray[np.int64]
    sample_keys: tuple[SampleKey, ...]
    subject_ids: NDArray[np.int64]
    window_bounds_seconds: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("At least one aligned feature family is required")
        candidates = tuple(family.block_names for family in self.families)
        if len(set(candidates)) != len(candidates):
            raise ValueError("Aligned feature families must be unique")

        n_rows = self.target_row_indices.size
        if self.target_row_indices.shape != (n_rows,) or self.target_row_indices.dtype != np.dtype(np.int64):
            raise TypeError("Target row indices must be a one-dimensional int64 array")
        if np.any(self.target_row_indices < 0) or not np.array_equal(
            self.target_row_indices,
            np.unique(self.target_row_indices),
        ):
            raise ValueError("Target row indices must be sorted, unique, and non-negative")
        if len(self.sample_keys) != n_rows or len(set(self.sample_keys)) != n_rows:
            raise ValueError("Sample keys must be unique and match aligned rows")
        if self.subject_ids.shape != (n_rows,) or self.subject_ids.dtype != np.dtype(np.int64):
            raise TypeError("Subject IDs must be an int64 vector matching aligned rows")
        if self.window_bounds_seconds.shape != (n_rows, 2):
            raise ValueError("Window bounds must have shape (row, 2)")
        if not np.isfinite(self.window_bounds_seconds).all():
            raise ValueError("Window bounds must be finite")
        if any(family.X.shape[0] != n_rows for family in self.families):
            raise ValueError("Every feature family must use the same aligned rows")


@dataclass(frozen=True, slots=True)
class PixelFold:
    pixel_index: int
    fold_index: int
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    n_samples: int

    def __post_init__(self) -> None:
        if self.pixel_index < 0 or self.fold_index < 0:
            raise ValueError("Pixel and fold indices must be non-negative")
        for name, indices in (
            ("train_indices", self.train_indices),
            ("validation_indices", self.validation_indices),
        ):
            if indices.ndim != 1 or indices.dtype != np.dtype(np.int64):
                raise TypeError(f"`{name}` must be a one-dimensional int64 array")
            if indices.size < 1 or np.any(indices < 0) or np.any(indices >= self.n_samples):
                raise ValueError(f"`{name}` must contain valid rows")
            if not np.array_equal(indices, np.unique(indices)):
                raise ValueError(f"`{name}` must be sorted and unique")
        if np.intersect1d(self.train_indices, self.validation_indices).size:
            raise ValueError("Fold train and validation rows must be disjoint")
        if set(self.train_subjects) & set(self.validation_subjects):
            raise ValueError("Fold train and validation subjects must be disjoint")


@dataclass(frozen=True, slots=True)
class GroupedPixelCrossValidation:
    folds: tuple[PixelFold, ...]
    n_samples: int
    n_pixels: int
    n_splits: int
    random_state: int

    def __post_init__(self) -> None:
        if self.n_samples < 1 or self.n_pixels < 1 or self.n_splits < 2:
            raise ValueError("Cross-validation dimensions must be positive")
        if len(self.folds) != self.n_pixels * self.n_splits:
            raise ValueError("Cross-validation must contain every pixel-fold combination")
        combinations = {(fold.pixel_index, fold.fold_index) for fold in self.folds}
        expected = {
            (pixel_index, fold_index)
            for pixel_index in range(self.n_pixels)
            for fold_index in range(self.n_splits)
        }
        if combinations != expected:
            raise ValueError("Cross-validation pixel-fold combinations are incomplete")
        if any(fold.n_samples != self.n_samples for fold in self.folds):
            raise ValueError("Every fold must use the configured sample count")

    def for_pixel(self, pixel_index: int) -> tuple[PixelFold, ...]:
        if pixel_index < 0 or pixel_index >= self.n_pixels:
            raise IndexError(f"Pixel index out of range: {pixel_index}")
        return tuple(fold for fold in self.folds if fold.pixel_index == pixel_index)


@dataclass(frozen=True, slots=True)
class CandidateScreeningResult:
    block_names: tuple[str, ...]
    fold_scores: NDArray[np.float64]
    selected_feature_counts: NDArray[np.int64]
    mean_pixel_scores: NDArray[np.float64]
    mean_score: float

    def __post_init__(self) -> None:
        if not self.block_names:
            raise ValueError("Screening result must identify a feature family")
        if self.fold_scores.ndim != 2 or self.fold_scores.dtype != np.dtype(np.float64):
            raise TypeError("Fold scores must be a two-dimensional float64 array")
        if self.selected_feature_counts.shape != self.fold_scores.shape:
            raise ValueError("Selected-feature counts must match fold scores")
        if self.selected_feature_counts.dtype != np.dtype(np.int64):
            raise TypeError("Selected-feature counts must use int64")
        if np.any(self.selected_feature_counts < 1):
            raise ValueError("Every fitted fold must retain at least one feature")
        if self.mean_pixel_scores.shape != (self.fold_scores.shape[0],):
            raise ValueError("Mean pixel scores must match the pixel axis")
        if self.mean_pixel_scores.dtype != np.dtype(np.float64):
            raise TypeError("Mean pixel scores must use float64")
        if not np.isfinite(self.fold_scores).all() or not np.isfinite(self.mean_pixel_scores).all():
            raise ValueError("Screening scores must be finite")
        if np.any((self.fold_scores < 0.0) | (self.fold_scores > 1.0)):
            raise ValueError("Balanced-accuracy scores must be in [0, 1]")
        if not np.isclose(self.mean_pixel_scores.mean(), self.mean_score):
            raise ValueError("Overall screening score must average the per-pixel scores")


@dataclass(frozen=True, slots=True)
class FeatureScreeningResult:
    candidates: tuple[CandidateScreeningResult, ...]
    selected_block_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("At least one candidate screening result is required")
        block_names = tuple(candidate.block_names for candidate in self.candidates)
        if len(set(block_names)) != len(block_names):
            raise ValueError("Candidate screening results must be unique")
        if self.selected_block_names not in block_names:
            raise ValueError("Selected feature family must be one of the candidates")
        best_index = int(np.argmax([candidate.mean_score for candidate in self.candidates]))
        if self.selected_block_names != self.candidates[best_index].block_names:
            raise ValueError("Selected feature family must use score then candidate-order tie-breaking")


@dataclass(frozen=True, slots=True)
class AlignedFeaturePartition:
    block_names: tuple[str, ...]
    X: NDArray[np.floating]
    feature_names: tuple[str, ...]
    target_row_indices: NDArray[np.int64]
    sample_keys: tuple[SampleKey, ...]
    subject_ids: NDArray[np.int64]
    window_bounds_seconds: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.block_names or len(set(self.block_names)) != len(self.block_names):
            raise ValueError("Feature-partition block names must be non-empty and unique")
        if self.X.ndim != 2 or not np.issubdtype(self.X.dtype, np.floating):
            raise TypeError("Feature-partition values must be a two-dimensional floating array")
        if not np.isfinite(self.X).all() or self.X.shape[1] < 1:
            raise ValueError("Feature-partition values must be finite with a non-empty feature axis")
        if len(self.feature_names) != self.X.shape[1] or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("Feature names must be unique and match the feature axis")

        n_rows = self.X.shape[0]
        if self.target_row_indices.shape != (n_rows,) or self.target_row_indices.dtype != np.dtype(np.int64):
            raise TypeError("Target row indices must be an int64 vector matching feature rows")
        if np.any(self.target_row_indices < 0) or not np.array_equal(
            self.target_row_indices,
            np.unique(self.target_row_indices),
        ):
            raise ValueError("Target row indices must be sorted, unique, and non-negative")
        if len(self.sample_keys) != n_rows or len(set(self.sample_keys)) != n_rows:
            raise ValueError("Sample keys must be unique and match feature rows")
        if self.subject_ids.shape != (n_rows,) or self.subject_ids.dtype != np.dtype(np.int64):
            raise TypeError("Subject IDs must be an int64 vector matching feature rows")
        if self.window_bounds_seconds.shape != (n_rows, 2):
            raise ValueError("Window bounds must have shape (row, 2)")
        if not np.isfinite(self.window_bounds_seconds).all():
            raise ValueError("Window bounds must be finite")


@dataclass(frozen=True, slots=True)
class PixelHyperparameters:
    select_k: int
    c: float
    penalty: str
    class_weight: str | None

    def __post_init__(self) -> None:
        if isinstance(self.select_k, bool) or self.select_k < 1:
            raise ValueError("Selected feature count must be positive")
        if not np.isfinite(self.c) or self.c <= 0:
            raise ValueError("Logistic Regression C must be finite and positive")
        if self.penalty not in ("l1", "l2"):
            raise ValueError(f"Unsupported penalty: {self.penalty!r}")
        if self.class_weight not in (None, "balanced"):
            raise ValueError(f"Unsupported class weight: {self.class_weight!r}")


@dataclass(frozen=True, slots=True)
class GridCandidateScore:
    hyperparameters: PixelHyperparameters
    mean_score: float
    std_score: float
    rank: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean_score) or not 0.0 <= self.mean_score <= 1.0:
            raise ValueError("Candidate mean balanced accuracy must be finite and in [0, 1]")
        if not np.isfinite(self.std_score) or self.std_score < 0.0:
            raise ValueError("Candidate score standard deviation must be finite and non-negative")
        if self.rank < 1:
            raise ValueError("Candidate rank must be positive")


@dataclass(frozen=True, slots=True)
class FittedPixelModel:
    pixel_index: int
    pixel_name: str
    pipeline: Pipeline
    best_hyperparameters: PixelHyperparameters
    best_cv_score: float
    candidate_scores: tuple[GridCandidateScore, ...]
    selected_feature_indices: NDArray[np.int64]
    selected_feature_names: tuple[str, ...]
    coefficients: NDArray[np.float64]
    intercept: float
    n_iter: int

    def __post_init__(self) -> None:
        if self.pixel_index < 0 or not self.pixel_name:
            raise ValueError("Pixel model must have a non-negative index and name")
        if not isinstance(self.pipeline, Pipeline):
            raise TypeError("Pixel model must contain a fitted sklearn Pipeline")
        if not np.isfinite(self.best_cv_score) or not 0.0 <= self.best_cv_score <= 1.0:
            raise ValueError("Best CV balanced accuracy must be finite and in [0, 1]")
        if not self.candidate_scores:
            raise ValueError("Pixel model must retain candidate grid scores")
        if self.selected_feature_indices.ndim != 1 or self.selected_feature_indices.dtype != np.dtype(
            np.int64
        ):
            raise TypeError("Selected feature indices must be a one-dimensional int64 array")
        if self.selected_feature_indices.size < 1 or not np.array_equal(
            self.selected_feature_indices,
            np.unique(self.selected_feature_indices),
        ):
            raise ValueError("Selected feature indices must be sorted, unique, and non-empty")
        if len(self.selected_feature_names) != self.selected_feature_indices.size:
            raise ValueError("Selected feature names must match selected indices")
        if self.coefficients.shape != (self.selected_feature_indices.size,):
            raise ValueError("Coefficients must match selected features")
        if self.coefficients.dtype != np.dtype(np.float64) or not np.isfinite(self.coefficients).all():
            raise TypeError("Coefficients must be a finite float64 vector")
        if not np.isfinite(self.intercept) or self.n_iter < 0:
            raise ValueError("Model intercept and iteration count must be valid")


@dataclass(frozen=True, slots=True)
class FittedPixelModels:
    block_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    training_target_indices: NDArray[np.int64]
    training_sample_keys: tuple[SampleKey, ...]
    cross_validation: GroupedPixelCrossValidation
    models: tuple[FittedPixelModel, ...]

    def __post_init__(self) -> None:
        if not self.block_names or not self.feature_names or not self.models:
            raise ValueError("Fitted pixel models require a feature family, features, and models")
        if self.training_target_indices.ndim != 1 or self.training_target_indices.dtype != np.dtype(
            np.int64
        ):
            raise TypeError("Training target indices must be a one-dimensional int64 array")
        if len(self.training_sample_keys) != self.training_target_indices.size:
            raise ValueError("Training sample keys must match training rows")
        if self.cross_validation.n_samples != self.training_target_indices.size:
            raise ValueError("Cross-validation must match training rows")
        if len(self.models) != self.cross_validation.n_pixels:
            raise ValueError("Exactly one fitted model is required per pixel")
        if tuple(model.pixel_index for model in self.models) != tuple(range(len(self.models))):
            raise ValueError("Fitted pixel models must be ordered by contiguous pixel index")


@dataclass(frozen=True, slots=True)
class PixelGridSearchResult:
    fitted_models: FittedPixelModels
    test_target_indices: NDArray[np.int64]
    test_sample_keys: tuple[SampleKey, ...]
    probabilities: NDArray[np.float64]
    predictions: NDArray[np.int8]
    test_balanced_accuracy: NDArray[np.float64]
    threshold: float

    def __post_init__(self) -> None:
        n_test = self.test_target_indices.size
        n_pixels = len(self.fitted_models.models)
        if self.test_target_indices.shape != (n_test,) or self.test_target_indices.dtype != np.dtype(np.int64):
            raise TypeError("Test target indices must be a one-dimensional int64 array")
        if len(self.test_sample_keys) != n_test:
            raise ValueError("Test sample keys must match test rows")
        if self.probabilities.shape != (n_test, n_pixels) or self.probabilities.dtype != np.dtype(np.float64):
            raise TypeError("Test probabilities must be a float64 matrix matching rows and pixels")
        if not np.isfinite(self.probabilities).all() or np.any(
            (self.probabilities < 0.0) | (self.probabilities > 1.0)
        ):
            raise ValueError("Test probabilities must be finite and in [0, 1]")
        if self.predictions.shape != self.probabilities.shape or self.predictions.dtype != np.dtype(np.int8):
            raise TypeError("Test predictions must be an int8 matrix matching probabilities")
        if not np.isin(self.predictions, (0, 1)).all():
            raise ValueError("Test predictions must be binary")
        if self.test_balanced_accuracy.shape != (n_pixels,) or self.test_balanced_accuracy.dtype != np.dtype(
            np.float64
        ):
            raise TypeError("Test balanced accuracy must be a float64 vector matching pixels")
        if not np.isfinite(self.test_balanced_accuracy).all() or np.any(
            (self.test_balanced_accuracy < 0.0) | (self.test_balanced_accuracy > 1.0)
        ):
            raise ValueError("Test balanced accuracy must be finite and in [0, 1]")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("Prediction threshold must be between zero and one")
