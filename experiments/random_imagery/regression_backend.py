import tempfile
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from joblib import Memory
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from experiments.logistic_regression.schemas import (
    AlignedFeaturePartition,
    AlignedTrainingFeatures,
    GroupedPixelCrossValidation,
    PixelTargetDataset,
)
from experiments.logistic_regression.screening import (
    CappedSelectKBest,
    build_grouped_pixel_cross_validation,
)
from experiments.random_imagery.config import (
    ElasticNetGridSearchConfig,
    PLSRegressionGridSearchConfig,
    RandomForestRegressionGridSearchConfig,
    RandomImageryExperimentConfigLike,
    RegressionExperimentConfig,
    RegressionGridSearchConfig,
    RidgeRegressionGridSearchConfig,
)
from experiments.random_imagery.contracts import (
    FittedDirectionModel,
    ModelPrediction,
    ScoreDiagnostics,
)
from experiments.random_imagery.registry import ModelSpec, get_model_spec

RegressionFamily = Literal[
    "ridge_regression",
    "elastic_net",
    "random_forest",
    "pls_regression",
]
RegressionParameter: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class MultiOutputFold:
    fold_index: int
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    n_samples: int

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("Fold index must be non-negative")
        for name, indices in (
            ("train_indices", self.train_indices),
            ("validation_indices", self.validation_indices),
        ):
            if (
                indices.ndim != 1
                or indices.dtype != np.dtype(np.int64)
                or indices.size < 1
            ):
                raise TypeError(f"`{name}` must be a non-empty int64 vector")
            if np.any(indices < 0) or np.any(indices >= self.n_samples):
                raise ValueError(f"`{name}` contains an out-of-range row")
            if not np.array_equal(indices, np.unique(indices)):
                raise ValueError(f"`{name}` must be sorted and unique")
        if np.intersect1d(self.train_indices, self.validation_indices).size:
            raise ValueError("Fold train and validation rows must be disjoint")
        if set(self.train_subjects) & set(self.validation_subjects):
            raise ValueError("Fold train and validation subjects must be disjoint")


@dataclass(frozen=True, slots=True)
class GroupedMultiOutputCrossValidation:
    folds: tuple[MultiOutputFold, ...]
    n_samples: int
    n_targets: int
    n_splits: int
    random_state: int

    def __post_init__(self) -> None:
        if self.n_samples < 1 or self.n_targets < 1 or self.n_splits < 2:
            raise ValueError("Cross-validation dimensions must be positive")
        if len(self.folds) != self.n_splits:
            raise ValueError("Multi-output cross-validation requires one fold per split")
        if tuple(fold.fold_index for fold in self.folds) != tuple(range(self.n_splits)):
            raise ValueError("Multi-output folds must use contiguous ordered indices")
        validation_rows = np.sort(
            np.concatenate([fold.validation_indices for fold in self.folds])
        )
        if not np.array_equal(validation_rows, np.arange(self.n_samples)):
            raise ValueError("Multi-output validation folds must partition all rows")


@dataclass(frozen=True, slots=True)
class RegressionCandidateScreeningResult:
    block_names: tuple[str, ...]
    fold_balanced_accuracy: NDArray[np.float64]
    fold_clipped_mse: NDArray[np.float64]
    selected_feature_counts: NDArray[np.int64]
    mean_balanced_accuracy: float
    mean_clipped_mse: float

    def __post_init__(self) -> None:
        if not self.block_names:
            raise ValueError("Screening result must identify a feature family")
        shape = self.fold_balanced_accuracy.shape
        if (
            self.fold_balanced_accuracy.ndim != 2
            or self.fold_balanced_accuracy.dtype != np.dtype(np.float64)
            or self.fold_clipped_mse.shape != shape
            or self.fold_clipped_mse.dtype != np.dtype(np.float64)
            or self.selected_feature_counts.shape != shape
            or self.selected_feature_counts.dtype != np.dtype(np.int64)
        ):
            raise TypeError("Regression screening arrays must share target-fold shape")
        if not np.isfinite(self.fold_balanced_accuracy).all() or np.any(
            (self.fold_balanced_accuracy < 0.0)
            | (self.fold_balanced_accuracy > 1.0)
        ):
            raise ValueError("Screening balanced accuracy must be finite and in [0, 1]")
        if not np.isfinite(self.fold_clipped_mse).all() or np.any(
            self.fold_clipped_mse < 0.0
        ):
            raise ValueError("Screening clipped MSE must be finite and non-negative")
        if np.any(self.selected_feature_counts < 1):
            raise ValueError("Every screening fold must retain at least one feature")
        if not np.isclose(
            self.mean_balanced_accuracy,
            self.fold_balanced_accuracy.mean(),
        ):
            raise ValueError("Mean screening balanced accuracy is inconsistent")
        if not np.isclose(self.mean_clipped_mse, self.fold_clipped_mse.mean()):
            raise ValueError("Mean screening clipped MSE is inconsistent")


@dataclass(frozen=True, slots=True)
class RegressionFeatureScreeningResult:
    candidates: tuple[RegressionCandidateScreeningResult, ...]
    selected_block_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("At least one regression screening candidate is required")
        block_names = tuple(candidate.block_names for candidate in self.candidates)
        if len(set(block_names)) != len(block_names):
            raise ValueError("Regression screening candidates must be unique")
        expected = self.candidates[_select_screening_candidate(self.candidates)].block_names
        if self.selected_block_names != expected:
            raise ValueError(
                "Regression screening must use balanced accuracy, clipped MSE, then order"
            )


@dataclass(frozen=True, slots=True)
class RegressionHyperparameters:
    estimator_family: RegressionFamily
    select_k: int
    parameters: tuple[tuple[str, RegressionParameter], ...]

    def __post_init__(self) -> None:
        if isinstance(self.select_k, bool) or self.select_k < 1:
            raise ValueError("Selected feature count must be positive")
        names = tuple(name for name, _ in self.parameters)
        if not names or len(set(names)) != len(names):
            raise ValueError("Regression hyperparameters must be named and unique")
        for _, value in self.parameters:
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("Floating hyperparameters must be finite")


@dataclass(frozen=True, slots=True)
class RegressionCandidateScore:
    hyperparameters: RegressionHyperparameters
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    mean_clipped_mse: float
    std_clipped_mse: float
    rank: int

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.mean_balanced_accuracy)
            or not 0.0 <= self.mean_balanced_accuracy <= 1.0
            or not np.isfinite(self.std_balanced_accuracy)
            or self.std_balanced_accuracy < 0.0
        ):
            raise ValueError("Candidate balanced-accuracy summary is invalid")
        if (
            not np.isfinite(self.mean_clipped_mse)
            or self.mean_clipped_mse < 0.0
            or not np.isfinite(self.std_clipped_mse)
            or self.std_clipped_mse < 0.0
        ):
            raise ValueError("Candidate clipped-MSE summary is invalid")
        if self.rank < 1:
            raise ValueError("Candidate rank must be positive")


@dataclass(frozen=True, slots=True)
class FittedIndependentRegressionModel:
    pixel_index: int
    pixel_name: str
    pipeline: Pipeline
    best_hyperparameters: RegressionHyperparameters
    best_cv_balanced_accuracy: float
    best_cv_clipped_mse: float
    candidate_scores: tuple[RegressionCandidateScore, ...]
    selected_feature_indices: NDArray[np.int64]
    selected_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.pixel_index < 0 or not self.pixel_name:
            raise ValueError("Independent regression model requires a pixel identity")
        _validate_fitted_regression_summary(
            pipeline=self.pipeline,
            best_cv_balanced_accuracy=self.best_cv_balanced_accuracy,
            best_cv_clipped_mse=self.best_cv_clipped_mse,
            candidate_scores=self.candidate_scores,
            selected_feature_indices=self.selected_feature_indices,
            selected_feature_names=self.selected_feature_names,
        )


@dataclass(frozen=True, slots=True)
class FittedIndependentRegressionModels:
    block_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    training_target_indices: NDArray[np.int64]
    training_sample_keys: tuple[tuple[int, int, int], ...]
    cross_validation: GroupedPixelCrossValidation
    models: tuple[FittedIndependentRegressionModel, ...]

    def __post_init__(self) -> None:
        _validate_regression_payload_header(
            block_names=self.block_names,
            feature_names=self.feature_names,
            training_target_indices=self.training_target_indices,
            training_sample_keys=self.training_sample_keys,
        )
        if self.cross_validation.n_samples != self.training_target_indices.size:
            raise ValueError("Pixel cross-validation must match training rows")
        if len(self.models) != self.cross_validation.n_pixels:
            raise ValueError("Independent topology requires one model per pixel")
        if tuple(model.pixel_index for model in self.models) != tuple(
            range(len(self.models))
        ):
            raise ValueError("Independent models must use contiguous pixel order")


@dataclass(frozen=True, slots=True)
class FittedMultiOutputRegressionModel:
    block_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    training_target_indices: NDArray[np.int64]
    training_sample_keys: tuple[tuple[int, int, int], ...]
    cross_validation: GroupedMultiOutputCrossValidation
    pipeline: Pipeline
    best_hyperparameters: RegressionHyperparameters
    best_cv_balanced_accuracy: float
    best_cv_clipped_mse: float
    candidate_scores: tuple[RegressionCandidateScore, ...]
    selected_feature_indices: NDArray[np.int64]
    selected_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_regression_payload_header(
            block_names=self.block_names,
            feature_names=self.feature_names,
            training_target_indices=self.training_target_indices,
            training_sample_keys=self.training_sample_keys,
        )
        if self.cross_validation.n_samples != self.training_target_indices.size:
            raise ValueError("Multi-output cross-validation must match training rows")
        _validate_fitted_regression_summary(
            pipeline=self.pipeline,
            best_cv_balanced_accuracy=self.best_cv_balanced_accuracy,
            best_cv_clipped_mse=self.best_cv_clipped_mse,
            candidate_scores=self.candidate_scores,
            selected_feature_indices=self.selected_feature_indices,
            selected_feature_names=self.selected_feature_names,
        )


class MultiTargetSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, *, k: int) -> None:
        self.k = k

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MultiTargetSelectKBest":
        values = np.asarray(X)
        targets = np.asarray(y)
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("`k` must be a positive integer")
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("Feature selection requires a non-empty matrix")
        if targets.ndim != 2 or targets.shape[0] != values.shape[0]:
            raise ValueError("Multi-target selection requires a matching target matrix")
        if not np.isin(targets, (0, 1)).all():
            raise ValueError("Multi-target feature ranking requires binary targets")

        n_features = values.shape[1]
        target_percentiles = np.empty((targets.shape[1], n_features), dtype=np.float64)
        feature_indices = np.arange(n_features, dtype=np.int64)
        for target_index in range(targets.shape[1]):
            scores, _ = _stable_f_classif(values, targets[:, target_index])
            order = np.lexsort((feature_indices, -scores))
            ranks = np.empty(n_features, dtype=np.int64)
            ranks[order] = np.arange(n_features, dtype=np.int64)
            target_percentiles[target_index] = (
                np.ones(1, dtype=np.float64)
                if n_features == 1
                else 1.0 - ranks / float(n_features - 1)
            )
        self.scores_ = target_percentiles.mean(axis=0)
        self.k_ = min(self.k, n_features)
        selected_order = np.lexsort((feature_indices, -self.scores_))[: self.k_]
        self.selected_indices_ = np.sort(selected_order.astype(np.int64, copy=False))
        self.n_features_in_ = n_features
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating[Any]]:
        check_is_fitted(self, ("selected_indices_", "n_features_in_"))
        values = np.asarray(X)
        return values[:, self.selected_indices_]

    def get_support(
        self,
        indices: bool = False,
    ) -> NDArray[np.bool_] | NDArray[np.int64]:
        check_is_fitted(self, ("selected_indices_", "n_features_in_"))
        if indices:
            return self.selected_indices_.copy()
        support = np.zeros(self.n_features_in_, dtype=np.bool_)
        support[self.selected_indices_] = True
        return support


class _RegressionBackend:
    spec: ModelSpec

    def fit(
        self,
        training_features: AlignedTrainingFeatures,
        *,
        targets: PixelTargetDataset,
        config: RandomImageryExperimentConfigLike,
    ) -> FittedDirectionModel[
        FittedIndependentRegressionModels | FittedMultiOutputRegressionModel,
        RegressionFeatureScreeningResult,
    ]:
        regression_config = _require_regression_config(config, spec=self.spec)
        y_train = _validated_targets(
            targets.y[training_features.target_row_indices],
            n_rows=training_features.target_row_indices.size,
        )
        if self.spec.topology == "independent":
            pixel_cv = build_grouped_pixel_cross_validation(
                y_train,
                groups=training_features.subject_ids,
                config=regression_config.cross_validation,
            )
            screening = screen_regression_feature_families(
                training_features,
                y=y_train,
                pixel_cross_validation=pixel_cv,
                multioutput_cross_validation=None,
                config=regression_config,
            )
            selected = _select_training_partition(
                training_features,
                block_names=screening.selected_block_names,
            )
            payload: FittedIndependentRegressionModels | FittedMultiOutputRegressionModel
            payload = fit_independent_regression_models(
                selected,
                y_train=y_train,
                pixel_names=targets.pixel_names,
                cross_validation=pixel_cv,
                config=regression_config,
            )
        else:
            multioutput_cv = build_grouped_multioutput_cross_validation(
                y_train,
                groups=training_features.subject_ids,
                config=regression_config,
            )
            screening = screen_regression_feature_families(
                training_features,
                y=y_train,
                pixel_cross_validation=None,
                multioutput_cross_validation=multioutput_cv,
                config=regression_config,
            )
            selected = _select_training_partition(
                training_features,
                block_names=screening.selected_block_names,
            )
            payload = fit_multioutput_regression_model(
                selected,
                y_train=y_train,
                cross_validation=multioutput_cv,
                config=regression_config,
            )
        return FittedDirectionModel(
            spec=self.spec,
            selected_block_names=screening.selected_block_names,
            feature_names=selected.feature_names,
            training_target_indices=selected.target_row_indices,
            training_sample_keys=selected.sample_keys,
            payload=payload,
            selection=screening,
        )

    def predict(
        self,
        fitted: FittedDirectionModel[
            FittedIndependentRegressionModels | FittedMultiOutputRegressionModel,
            RegressionFeatureScreeningResult,
        ],
        *,
        test_features: AlignedFeaturePartition,
        targets: PixelTargetDataset,
        config: RandomImageryExperimentConfigLike,
    ) -> ModelPrediction:
        regression_config = _require_regression_config(config, spec=self.spec)
        payload = fitted.payload
        if test_features.block_names != payload.block_names:
            raise ValueError("Train and test feature families differ")
        if test_features.feature_names != payload.feature_names:
            raise ValueError("Train and test feature names or channel order differ")

        n_test = test_features.X.shape[0]
        n_targets = targets.y.shape[1]
        if isinstance(payload, FittedIndependentRegressionModels):
            raw_scores = np.empty((n_test, len(payload.models)), dtype=np.float64)
            for model in payload.models:
                raw_scores[:, model.pixel_index] = _as_prediction_matrix(
                    model.pipeline.predict(test_features.X),
                    n_rows=n_test,
                    n_targets=1,
                )[:, 0]
        else:
            raw_scores = _as_prediction_matrix(
                payload.pipeline.predict(test_features.X),
                n_rows=n_test,
                n_targets=n_targets,
            )
        below_fraction = float(np.mean(raw_scores < 0.0))
        above_fraction = float(np.mean(raw_scores > 1.0))
        scores = np.clip(raw_scores, 0.0, 1.0).astype(np.float64, copy=False)
        predictions = (scores >= regression_config.prediction_threshold).astype(np.int8)
        scores.setflags(write=False)
        predictions.setflags(write=False)
        return ModelPrediction(
            test_target_indices=test_features.target_row_indices,
            test_sample_keys=test_features.sample_keys,
            scores=scores,
            predictions=predictions,
            threshold=regression_config.prediction_threshold,
            diagnostics=ScoreDiagnostics(
                score_semantics=self.spec.score_semantics,
                clipped_below_zero_fraction=below_fraction,
                clipped_above_one_fraction=above_fraction,
            ),
        )


class RidgeRegressionIndependentBackend(_RegressionBackend):
    spec = get_model_spec("ridge-regression-independent")


class RidgeRegressionMultiOutputBackend(_RegressionBackend):
    spec = get_model_spec("ridge-regression-multioutput")


class ElasticNetIndependentBackend(_RegressionBackend):
    spec = get_model_spec("elastic-net-independent")


class ElasticNetMultiOutputBackend(_RegressionBackend):
    spec = get_model_spec("elastic-net-multioutput")


class RandomForestIndependentBackend(_RegressionBackend):
    spec = get_model_spec("random-forest-independent")


class RandomForestMultiOutputBackend(_RegressionBackend):
    spec = get_model_spec("random-forest-multioutput")


class PLSRegressionMultiOutputBackend(_RegressionBackend):
    spec = get_model_spec("pls-regression-multioutput")


def build_grouped_multioutput_cross_validation(
    y: ArrayLike,
    *,
    groups: ArrayLike,
    config: RegressionExperimentConfig,
) -> GroupedMultiOutputCrossValidation:
    targets = _validated_targets(y)
    subject_groups = np.asarray(groups)
    if subject_groups.shape != (targets.shape[0],):
        raise ValueError("Groups must match multi-output target rows")
    if np.unique(subject_groups).size < config.cross_validation.n_splits:
        raise ValueError("Grouped cross-validation requires enough distinct subjects")

    splitter = GroupKFold(
        n_splits=config.cross_validation.n_splits,
        shuffle=config.cross_validation.shuffle,
        random_state=config.cross_validation.random_state,
    )
    folds: list[MultiOutputFold] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros((targets.shape[0], 1)), groups=subject_groups)
    ):
        train_indices = np.sort(train_indices.astype(np.int64, copy=False))
        validation_indices = np.sort(validation_indices.astype(np.int64, copy=False))
        for target_index in range(targets.shape[1]):
            if np.unique(targets[train_indices, target_index]).size != 2:
                raise ValueError(
                    f"Target {target_index} fold {fold_index} training rows lack both classes"
                )
            if np.unique(targets[validation_indices, target_index]).size != 2:
                raise ValueError(
                    f"Target {target_index} fold {fold_index} validation rows lack both classes"
                )
        train_indices.setflags(write=False)
        validation_indices.setflags(write=False)
        folds.append(
            MultiOutputFold(
                fold_index=fold_index,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_subjects=tuple(
                    int(value) for value in np.unique(subject_groups[train_indices])
                ),
                validation_subjects=tuple(
                    int(value) for value in np.unique(subject_groups[validation_indices])
                ),
                n_samples=targets.shape[0],
            )
        )
    return GroupedMultiOutputCrossValidation(
        folds=tuple(folds),
        n_samples=targets.shape[0],
        n_targets=targets.shape[1],
        n_splits=config.cross_validation.n_splits,
        random_state=config.cross_validation.random_state,
    )


def screen_regression_feature_families(
    features: AlignedTrainingFeatures,
    *,
    y: ArrayLike,
    pixel_cross_validation: GroupedPixelCrossValidation | None,
    multioutput_cross_validation: GroupedMultiOutputCrossValidation | None,
    config: RegressionExperimentConfig,
) -> RegressionFeatureScreeningResult:
    targets = _validated_targets(y, n_rows=features.target_row_indices.size)
    if (pixel_cross_validation is None) == (multioutput_cross_validation is None):
        raise ValueError("Pass exactly one regression cross-validation contract")
    if tuple(family.block_names for family in features.families) != (
        config.feature_screening.candidates
    ):
        raise ValueError("Aligned feature families must preserve configured candidate order")

    results: list[RegressionCandidateScreeningResult] = []
    for family in features.families:
        balanced_scores = np.empty(
            (targets.shape[1], config.cross_validation.n_splits),
            dtype=np.float64,
        )
        clipped_mse = np.empty_like(balanced_scores)
        selected_counts = np.empty_like(balanced_scores, dtype=np.int64)
        if pixel_cross_validation is not None:
            for target_index in range(targets.shape[1]):
                for fold in pixel_cross_validation.for_pixel(target_index):
                    pipeline = _build_regression_pipeline(
                        config=config,
                        topology="independent",
                        select_k=config.feature_screening.select_k,
                        parameters=_screening_parameters(config.grid_search),
                        cache_dir=None,
                    )
                    _fit_with_convergence_errors(
                        pipeline,
                        family.X[fold.train_indices],
                        targets[fold.train_indices, target_index],
                    )
                    raw = _as_prediction_matrix(
                        pipeline.predict(family.X[fold.validation_indices]),
                        n_rows=fold.validation_indices.size,
                        n_targets=1,
                    )
                    clipped = np.clip(raw[:, 0], 0.0, 1.0)
                    balanced_scores[target_index, fold.fold_index] = (
                        balanced_accuracy_score(
                            targets[fold.validation_indices, target_index],
                            clipped >= config.prediction_threshold,
                        )
                    )
                    clipped_mse[target_index, fold.fold_index] = float(
                        np.mean(
                            (
                                clipped
                                - targets[fold.validation_indices, target_index]
                            )
                            ** 2
                        )
                    )
                    selected_counts[target_index, fold.fold_index] = (
                        _selected_feature_count(pipeline)
                    )
        else:
            if multioutput_cross_validation is None:
                raise RuntimeError("Multi-output screening requires its cross-validation")
            for fold in multioutput_cross_validation.folds:
                pipeline = _build_regression_pipeline(
                    config=config,
                    topology="multioutput",
                    select_k=config.feature_screening.select_k,
                    parameters=_screening_parameters(config.grid_search),
                    cache_dir=None,
                )
                _fit_with_convergence_errors(
                    pipeline,
                    family.X[fold.train_indices],
                    targets[fold.train_indices],
                )
                raw = _as_prediction_matrix(
                    pipeline.predict(family.X[fold.validation_indices]),
                    n_rows=fold.validation_indices.size,
                    n_targets=targets.shape[1],
                )
                clipped = np.clip(raw, 0.0, 1.0)
                for target_index in range(targets.shape[1]):
                    balanced_scores[target_index, fold.fold_index] = (
                        balanced_accuracy_score(
                            targets[fold.validation_indices, target_index],
                            clipped[:, target_index] >= config.prediction_threshold,
                        )
                    )
                    clipped_mse[target_index, fold.fold_index] = float(
                        np.mean(
                            (
                                clipped[:, target_index]
                                - targets[fold.validation_indices, target_index]
                            )
                            ** 2
                        )
                    )
                selected_counts[:, fold.fold_index] = _selected_feature_count(
                    pipeline
                )
        for array in (balanced_scores, clipped_mse, selected_counts):
            array.setflags(write=False)
        results.append(
            RegressionCandidateScreeningResult(
                block_names=family.block_names,
                fold_balanced_accuracy=balanced_scores,
                fold_clipped_mse=clipped_mse,
                selected_feature_counts=selected_counts,
                mean_balanced_accuracy=float(balanced_scores.mean()),
                mean_clipped_mse=float(clipped_mse.mean()),
            )
        )
    selected_index = _select_screening_candidate(tuple(results))
    return RegressionFeatureScreeningResult(
        candidates=tuple(results),
        selected_block_names=results[selected_index].block_names,
    )


def fit_independent_regression_models(
    training_features: AlignedFeaturePartition,
    *,
    y_train: ArrayLike,
    pixel_names: tuple[str, ...],
    cross_validation: GroupedPixelCrossValidation,
    config: RegressionExperimentConfig,
) -> FittedIndependentRegressionModels:
    targets = _validated_targets(
        y_train,
        n_rows=training_features.X.shape[0],
        n_targets=len(pixel_names),
    )
    models: list[FittedIndependentRegressionModel] = []
    for pixel_index, pixel_name in enumerate(pixel_names):
        folds = cross_validation.for_pixel(pixel_index)
        cv = tuple((fold.train_indices, fold.validation_indices) for fold in folds)
        search = _fit_regression_search(
            training_features.X,
            targets[:, pixel_index],
            topology="independent",
            cv=cv,
            config=config,
            prefix=f"{config.grid_search.estimator_family}-pixel-{pixel_index:02d}-",
        )
        best_pipeline = search.best_estimator_
        best_pipeline.set_params(memory=None)
        models.append(
            FittedIndependentRegressionModel(
                pixel_index=pixel_index,
                pixel_name=pixel_name,
                pipeline=best_pipeline,
                **_build_search_summary(
                    search,
                    feature_names=training_features.feature_names,
                    estimator_family=config.grid_search.estimator_family,
                ),
            )
        )
    return FittedIndependentRegressionModels(
        block_names=training_features.block_names,
        feature_names=training_features.feature_names,
        training_target_indices=training_features.target_row_indices,
        training_sample_keys=training_features.sample_keys,
        cross_validation=cross_validation,
        models=tuple(models),
    )


def fit_multioutput_regression_model(
    training_features: AlignedFeaturePartition,
    *,
    y_train: ArrayLike,
    cross_validation: GroupedMultiOutputCrossValidation,
    config: RegressionExperimentConfig,
) -> FittedMultiOutputRegressionModel:
    targets = _validated_targets(
        y_train,
        n_rows=training_features.X.shape[0],
        n_targets=cross_validation.n_targets,
    )
    cv = tuple(
        (fold.train_indices, fold.validation_indices)
        for fold in cross_validation.folds
    )
    search = _fit_regression_search(
        training_features.X,
        targets,
        topology="multioutput",
        cv=cv,
        config=config,
        prefix=f"{config.grid_search.estimator_family}-multioutput-",
    )
    best_pipeline = search.best_estimator_
    best_pipeline.set_params(memory=None)
    return FittedMultiOutputRegressionModel(
        block_names=training_features.block_names,
        feature_names=training_features.feature_names,
        training_target_indices=training_features.target_row_indices,
        training_sample_keys=training_features.sample_keys,
        cross_validation=cross_validation,
        pipeline=best_pipeline,
        **_build_search_summary(
            search,
            feature_names=training_features.feature_names,
            estimator_family=config.grid_search.estimator_family,
        ),
    )


def _fit_regression_search(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.int8],
    *,
    topology: Literal["independent", "multioutput"],
    cv: tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...],
    config: RegressionExperimentConfig,
    prefix: str,
) -> GridSearchCV:
    with tempfile.TemporaryDirectory(prefix=prefix) as cache_dir:
        pipeline = _build_regression_pipeline(
            config=config,
            topology=topology,
            select_k=config.grid_search.select_k[0],
            parameters=_first_grid_parameters(config.grid_search),
            cache_dir=Path(cache_dir),
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=_build_parameter_grid(config.grid_search),
            scoring={
                "balanced_accuracy": partial(
                    _threshold_balanced_accuracy_scorer,
                    threshold=config.prediction_threshold,
                ),
                "negative_clipped_mse": _negative_clipped_mse_scorer,
            },
            cv=cv,
            refit=_select_best_grid_candidate,
            n_jobs=config.grid_search.n_jobs,
            error_score=config.grid_search.error_score,
            return_train_score=False,
        )
        _fit_with_convergence_errors(search, X, y)
    return search


def _build_regression_pipeline(
    *,
    config: RegressionExperimentConfig,
    topology: Literal["independent", "multioutput"],
    select_k: int,
    parameters: dict[str, RegressionParameter],
    cache_dir: Path | None,
) -> Pipeline:
    selector: CappedSelectKBest | MultiTargetSelectKBest
    selector = (
        CappedSelectKBest(k=select_k)
        if topology == "independent"
        else MultiTargetSelectKBest(k=select_k)
    )
    scale: StandardScaler | Literal["passthrough"] = (
        "passthrough"
        if config.grid_search.estimator_family == "random_forest"
        else StandardScaler()
    )
    return Pipeline(
        steps=(
            (
                "variance",
                VarianceThreshold(
                    threshold=config.feature_screening.variance_threshold
                ),
            ),
            ("select", selector),
            ("scale", scale),
            ("model", _build_regressor(config=config, parameters=parameters)),
        ),
        memory=None if cache_dir is None else Memory(location=cache_dir, verbose=0),
    )


def _build_regressor(
    *,
    config: RegressionExperimentConfig,
    parameters: dict[str, RegressionParameter],
) -> Ridge | ElasticNet | RandomForestRegressor | PLSRegression:
    grid = config.grid_search
    if isinstance(grid, RidgeRegressionGridSearchConfig):
        return Ridge(
            alpha=float(parameters["alpha"]),
            solver=grid.solver,
            tol=grid.tolerance,
            random_state=config.random_state,
        )
    if isinstance(grid, ElasticNetGridSearchConfig):
        return ElasticNet(
            alpha=float(parameters["alpha"]),
            l1_ratio=float(parameters["l1_ratio"]),
            max_iter=grid.max_iter,
            tol=grid.tolerance,
            selection=grid.selection,
            random_state=config.random_state,
        )
    if isinstance(grid, RandomForestRegressionGridSearchConfig):
        max_depth = parameters["max_depth"]
        return RandomForestRegressor(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=None if max_depth is None else int(max_depth),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            n_jobs=grid.estimator_n_jobs,
            random_state=config.random_state,
        )
    if isinstance(grid, PLSRegressionGridSearchConfig):
        return PLSRegression(
            n_components=int(parameters["n_components"]),
            scale=grid.scale,
            max_iter=grid.max_iter,
            tol=grid.tolerance,
        )
    raise TypeError(f"Unsupported regression grid: {type(grid).__name__}")


def _build_parameter_grid(
    config: RegressionGridSearchConfig,
) -> dict[str, tuple[RegressionParameter, ...]]:
    grid: dict[str, tuple[RegressionParameter, ...]] = {
        "select__k": tuple(config.select_k),
    }
    if isinstance(config, RidgeRegressionGridSearchConfig):
        grid["model__alpha"] = tuple(config.alpha_values)
    elif isinstance(config, ElasticNetGridSearchConfig):
        grid["model__alpha"] = tuple(config.alpha_values)
        grid["model__l1_ratio"] = tuple(config.l1_ratios)
    elif isinstance(config, RandomForestRegressionGridSearchConfig):
        grid.update(
            {
                "model__n_estimators": tuple(config.n_estimators),
                "model__max_depth": tuple(config.max_depth),
                "model__min_samples_leaf": tuple(config.min_samples_leaf),
                "model__max_features": tuple(config.max_features),
            }
        )
    elif isinstance(config, PLSRegressionGridSearchConfig):
        grid["model__n_components"] = tuple(config.n_components)
    else:
        raise TypeError(f"Unsupported regression grid: {type(config).__name__}")
    return grid


def _first_grid_parameters(
    config: RegressionGridSearchConfig,
) -> dict[str, RegressionParameter]:
    parameter_grid = _build_parameter_grid(config)
    return {
        key.removeprefix("model__"): values[0]
        for key, values in parameter_grid.items()
        if key.startswith("model__")
    }


def _screening_parameters(
    config: RegressionGridSearchConfig,
) -> dict[str, RegressionParameter]:
    if isinstance(config, RidgeRegressionGridSearchConfig):
        return {"alpha": config.screening_alpha}
    if isinstance(config, ElasticNetGridSearchConfig):
        return {
            "alpha": config.screening_alpha,
            "l1_ratio": config.screening_l1_ratio,
        }
    if isinstance(config, RandomForestRegressionGridSearchConfig):
        return {
            "n_estimators": config.screening_n_estimators,
            "max_depth": config.screening_max_depth,
            "min_samples_leaf": config.screening_min_samples_leaf,
            "max_features": config.screening_max_features,
        }
    if isinstance(config, PLSRegressionGridSearchConfig):
        return {"n_components": config.screening_n_components}
    raise TypeError(f"Unsupported regression grid: {type(config).__name__}")


def _threshold_balanced_accuracy_scorer(
    estimator: Pipeline,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.int8],
    *,
    threshold: float,
) -> float:
    targets = np.asarray(y)
    n_targets = 1 if targets.ndim == 1 else targets.shape[1]
    raw = _as_prediction_matrix(
        estimator.predict(X),
        n_rows=targets.shape[0],
        n_targets=n_targets,
    )
    predictions = np.clip(raw, 0.0, 1.0) >= threshold
    target_matrix = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    return float(
        np.mean(
            [
                balanced_accuracy_score(
                    target_matrix[:, target_index],
                    predictions[:, target_index],
                )
                for target_index in range(n_targets)
            ]
        )
    )


def _negative_clipped_mse_scorer(
    estimator: Pipeline,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.int8],
) -> float:
    targets = np.asarray(y)
    n_targets = 1 if targets.ndim == 1 else targets.shape[1]
    raw = _as_prediction_matrix(
        estimator.predict(X),
        n_rows=targets.shape[0],
        n_targets=n_targets,
    )
    target_matrix = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    return -float(np.mean((np.clip(raw, 0.0, 1.0) - target_matrix) ** 2))


def _select_best_grid_candidate(results: dict[str, Any]) -> int:
    balanced = np.asarray(results["mean_test_balanced_accuracy"], dtype=np.float64)
    clipped_mse = -np.asarray(
        results["mean_test_negative_clipped_mse"],
        dtype=np.float64,
    )
    if not np.isfinite(balanced).all() or not np.isfinite(clipped_mse).all():
        raise ValueError("Regression grid search produced non-finite validation scores")
    order = np.lexsort(
        (
            np.arange(balanced.size, dtype=np.int64),
            clipped_mse,
            -balanced,
        )
    )
    return int(order[0])


def _build_search_summary(
    search: GridSearchCV,
    *,
    feature_names: tuple[str, ...],
    estimator_family: RegressionFamily,
) -> dict[str, Any]:
    best_index = int(search.best_index_)
    balanced = np.asarray(
        search.cv_results_["mean_test_balanced_accuracy"],
        dtype=np.float64,
    )
    balanced_std = np.asarray(
        search.cv_results_["std_test_balanced_accuracy"],
        dtype=np.float64,
    )
    clipped_mse = -np.asarray(
        search.cv_results_["mean_test_negative_clipped_mse"],
        dtype=np.float64,
    )
    clipped_mse_std = np.asarray(
        search.cv_results_["std_test_negative_clipped_mse"],
        dtype=np.float64,
    )
    ranks = _candidate_ranks(balanced, clipped_mse)
    candidate_scores = tuple(
        RegressionCandidateScore(
            hyperparameters=_params_to_hyperparameters(
                params,
                estimator_family=estimator_family,
            ),
            mean_balanced_accuracy=float(balanced[index]),
            std_balanced_accuracy=float(balanced_std[index]),
            mean_clipped_mse=float(clipped_mse[index]),
            std_clipped_mse=float(clipped_mse_std[index]),
            rank=int(ranks[index]),
        )
        for index, params in enumerate(search.cv_results_["params"])
    )
    selected_indices = _selected_feature_indices(search.best_estimator_)
    selected_names = tuple(feature_names[int(index)] for index in selected_indices)
    return {
        "best_hyperparameters": _params_to_hyperparameters(
            search.best_params_,
            estimator_family=estimator_family,
        ),
        "best_cv_balanced_accuracy": float(balanced[best_index]),
        "best_cv_clipped_mse": float(clipped_mse[best_index]),
        "candidate_scores": candidate_scores,
        "selected_feature_indices": selected_indices,
        "selected_feature_names": selected_names,
    }


def _candidate_ranks(
    balanced_accuracy: NDArray[np.float64],
    clipped_mse: NDArray[np.float64],
) -> NDArray[np.int64]:
    order = np.lexsort(
        (
            np.arange(balanced_accuracy.size, dtype=np.int64),
            clipped_mse,
            -balanced_accuracy,
        )
    )
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.int64)
    return ranks


def _params_to_hyperparameters(
    params: dict[str, Any],
    *,
    estimator_family: RegressionFamily,
) -> RegressionHyperparameters:
    select_k = int(params["select__k"])
    parameters = tuple(
        sorted(
            (
                key.removeprefix("model__"),
                _normalize_parameter(value),
            )
            for key, value in params.items()
            if key.startswith("model__")
        )
    )
    return RegressionHyperparameters(
        estimator_family=estimator_family,
        select_k=select_k,
        parameters=parameters,
    )


def _normalize_parameter(value: Any) -> RegressionParameter:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Unsupported regression parameter value: {value!r}")


def _selected_feature_count(pipeline: Pipeline) -> int:
    selector = pipeline.named_steps["select"]
    if not isinstance(selector, (CappedSelectKBest, MultiTargetSelectKBest)):
        raise TypeError("Unexpected regression feature selector")
    return int(selector.get_support().sum())


def _selected_feature_indices(pipeline: Pipeline) -> NDArray[np.int64]:
    variance = pipeline.named_steps["variance"]
    selector = pipeline.named_steps["select"]
    if not isinstance(variance, VarianceThreshold):
        raise TypeError("Unexpected fitted variance selector")
    if not isinstance(selector, (CappedSelectKBest, MultiTargetSelectKBest)):
        raise TypeError("Unexpected fitted regression selector")
    variance_indices = variance.get_support(indices=True).astype(np.int64, copy=False)
    selected = variance_indices[selector.get_support(indices=True)]
    selected = np.sort(selected.astype(np.int64, copy=False))
    selected.setflags(write=False)
    return selected


def _select_screening_candidate(
    candidates: tuple[RegressionCandidateScreeningResult, ...],
) -> int:
    balanced = np.asarray(
        [candidate.mean_balanced_accuracy for candidate in candidates],
        dtype=np.float64,
    )
    clipped_mse = np.asarray(
        [candidate.mean_clipped_mse for candidate in candidates],
        dtype=np.float64,
    )
    return int(
        np.lexsort(
            (
                np.arange(len(candidates), dtype=np.int64),
                clipped_mse,
                -balanced,
            )
        )[0]
    )


def _as_prediction_matrix(
    values: ArrayLike,
    *,
    n_rows: int,
    n_targets: int,
) -> NDArray[np.float64]:
    predictions = np.asarray(values, dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if predictions.shape != (n_rows, n_targets):
        raise ValueError(
            f"Regression predictions have shape {predictions.shape}, "
            f"expected {(n_rows, n_targets)}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Regression predictions must be finite")
    return predictions


def _select_training_partition(
    features: AlignedTrainingFeatures,
    *,
    block_names: tuple[str, ...],
) -> AlignedFeaturePartition:
    matches = tuple(
        family for family in features.families if family.block_names == block_names
    )
    if len(matches) != 1:
        raise ValueError("Selected feature family is absent or ambiguous")
    family = matches[0]
    return AlignedFeaturePartition(
        block_names=family.block_names,
        X=family.X,
        feature_names=family.feature_names,
        target_row_indices=features.target_row_indices,
        sample_keys=features.sample_keys,
        subject_ids=features.subject_ids,
        window_bounds_seconds=features.window_bounds_seconds,
    )


def _validated_targets(
    y: ArrayLike,
    *,
    n_rows: int | None = None,
    n_targets: int | None = None,
) -> NDArray[np.int8]:
    targets = np.asarray(y)
    if targets.ndim != 2 or targets.shape[0] < 1 or targets.shape[1] < 1:
        raise ValueError("Regression targets must be a non-empty sample-target matrix")
    if n_rows is not None and targets.shape[0] != n_rows:
        raise ValueError("Regression targets do not match expected rows")
    if n_targets is not None and targets.shape[1] != n_targets:
        raise ValueError("Regression targets do not match expected target count")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Regression targets must be binary")
    return targets.astype(np.int8, copy=False)


def _fit_with_convergence_errors(
    estimator: Any,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.int8],
) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return estimator.fit(X, y)


def _stable_f_classif(
    X: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, p_values = f_classif(X, y)
    finite_limit = np.finfo(np.float64).max
    return (
        np.nan_to_num(scores, nan=0.0, posinf=finite_limit, neginf=0.0),
        np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=0.0),
    )


def _validate_regression_payload_header(
    *,
    block_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    training_target_indices: NDArray[np.int64],
    training_sample_keys: tuple[tuple[int, int, int], ...],
) -> None:
    if not block_names or not feature_names:
        raise ValueError("Regression payload requires a feature family and features")
    if (
        training_target_indices.ndim != 1
        or training_target_indices.dtype != np.dtype(np.int64)
    ):
        raise TypeError("Training target indices must be a one-dimensional int64 array")
    if len(training_sample_keys) != training_target_indices.size:
        raise ValueError("Training sample keys must match training rows")


def _validate_fitted_regression_summary(
    *,
    pipeline: Pipeline,
    best_cv_balanced_accuracy: float,
    best_cv_clipped_mse: float,
    candidate_scores: tuple[RegressionCandidateScore, ...],
    selected_feature_indices: NDArray[np.int64],
    selected_feature_names: tuple[str, ...],
) -> None:
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Regression model must contain a fitted sklearn Pipeline")
    if (
        not np.isfinite(best_cv_balanced_accuracy)
        or not 0.0 <= best_cv_balanced_accuracy <= 1.0
        or not np.isfinite(best_cv_clipped_mse)
        or best_cv_clipped_mse < 0.0
    ):
        raise ValueError("Best regression CV scores are invalid")
    if not candidate_scores:
        raise ValueError("Regression model must retain candidate scores")
    if (
        selected_feature_indices.ndim != 1
        or selected_feature_indices.dtype != np.dtype(np.int64)
        or selected_feature_indices.size < 1
        or not np.array_equal(
            selected_feature_indices,
            np.unique(selected_feature_indices),
        )
    ):
        raise TypeError("Selected feature indices must be sorted non-empty int64")
    if len(selected_feature_names) != selected_feature_indices.size:
        raise ValueError("Selected feature names must match selected indices")


def _require_regression_config(
    config: RandomImageryExperimentConfigLike,
    *,
    spec: ModelSpec,
) -> RegressionExperimentConfig:
    if not isinstance(config, RegressionExperimentConfig):
        raise TypeError("Regression backend requires its experiment configuration")
    if config.model_id != spec.model_id:
        raise ValueError(
            f"Backend model {spec.model_id!r} does not match config model {config.model_id!r}"
        )
    return config


__all__ = [
    "ElasticNetIndependentBackend",
    "ElasticNetMultiOutputBackend",
    "FittedIndependentRegressionModel",
    "FittedIndependentRegressionModels",
    "FittedMultiOutputRegressionModel",
    "GroupedMultiOutputCrossValidation",
    "MultiOutputFold",
    "MultiTargetSelectKBest",
    "PLSRegressionMultiOutputBackend",
    "RandomForestIndependentBackend",
    "RandomForestMultiOutputBackend",
    "RegressionCandidateScore",
    "RegressionCandidateScreeningResult",
    "RegressionFeatureScreeningResult",
    "RegressionHyperparameters",
    "RidgeRegressionIndependentBackend",
    "RidgeRegressionMultiOutputBackend",
    "build_grouped_multioutput_cross_validation",
    "fit_independent_regression_models",
    "fit_multioutput_regression_model",
    "screen_regression_feature_families",
]
