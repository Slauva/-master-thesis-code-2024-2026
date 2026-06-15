import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from joblib import Memory
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from experiments.logistic_regression.schemas import (
    AlignedFeaturePartition,
    AlignedTrainingFeatures,
    CandidateScreeningResult,
    FeatureScreeningResult,
    GroupedPixelCrossValidation,
    PixelTargetDataset,
)
from experiments.logistic_regression.screening import (
    CappedSelectKBest,
    build_grouped_pixel_cross_validation,
)
from experiments.random_imagery.config import (
    CalibratedClassifierExperimentConfig,
    ClassifierFeatureScreeningConfig,
    LinearSVMGridSearchConfig,
    RandomImageryExperimentConfigLike,
    RidgeClassifierGridSearchConfig,
)
from experiments.random_imagery.contracts import (
    FittedDirectionModel,
    ModelPrediction,
    ScoreDiagnostics,
)
from experiments.random_imagery.registry import ModelSpec, get_model_spec

ClassifierFamily = Literal["linear_svm", "ridge_classifier"]


@dataclass(frozen=True, slots=True)
class ClassifierHyperparameters:
    estimator_family: ClassifierFamily
    select_k: int
    regularization: float
    class_weight: str | None

    def __post_init__(self) -> None:
        if isinstance(self.select_k, bool) or self.select_k < 1:
            raise ValueError("Selected feature count must be positive")
        if not np.isfinite(self.regularization) or self.regularization <= 0.0:
            raise ValueError("Classifier regularization must be finite and positive")
        if self.class_weight not in (None, "balanced"):
            raise ValueError(f"Unsupported class weight: {self.class_weight!r}")


@dataclass(frozen=True, slots=True)
class ClassifierCandidateScore:
    hyperparameters: ClassifierHyperparameters
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
class PlattCalibration:
    calibrator: LogisticRegression
    oof_decision_scores: NDArray[np.float64]
    oof_fold_indices: NDArray[np.int64]
    coefficient: float
    intercept: float

    def __post_init__(self) -> None:
        n_rows = self.oof_decision_scores.size
        if (
            self.oof_decision_scores.shape != (n_rows,)
            or self.oof_decision_scores.dtype != np.dtype(np.float64)
            or not np.isfinite(self.oof_decision_scores).all()
        ):
            raise TypeError("OOF decision scores must be a finite float64 vector")
        if (
            self.oof_fold_indices.shape != (n_rows,)
            or self.oof_fold_indices.dtype != np.dtype(np.int64)
            or np.any(self.oof_fold_indices < 0)
        ):
            raise TypeError("OOF fold indices must be a non-negative int64 vector")
        if not np.isfinite(self.coefficient) or not np.isfinite(self.intercept):
            raise ValueError("Platt calibration parameters must be finite")
        if not isinstance(self.calibrator, LogisticRegression):
            raise TypeError("Platt calibration requires a fitted LogisticRegression")


@dataclass(frozen=True, slots=True)
class FittedCalibratedPixelModel:
    pixel_index: int
    pixel_name: str
    pipeline: Pipeline
    calibration: PlattCalibration
    best_hyperparameters: ClassifierHyperparameters
    best_cv_score: float
    candidate_scores: tuple[ClassifierCandidateScore, ...]
    selected_feature_indices: NDArray[np.int64]
    selected_feature_names: tuple[str, ...]
    coefficients: NDArray[np.float64]
    intercept: float

    def __post_init__(self) -> None:
        if self.pixel_index < 0 or not self.pixel_name:
            raise ValueError("Pixel model must have a non-negative index and name")
        if not isinstance(self.pipeline, Pipeline):
            raise TypeError("Pixel model must contain a fitted sklearn Pipeline")
        if not np.isfinite(self.best_cv_score) or not 0.0 <= self.best_cv_score <= 1.0:
            raise ValueError("Best CV balanced accuracy must be finite and in [0, 1]")
        if not self.candidate_scores:
            raise ValueError("Pixel model must retain candidate grid scores")
        if (
            self.selected_feature_indices.ndim != 1
            or self.selected_feature_indices.dtype != np.dtype(np.int64)
            or self.selected_feature_indices.size < 1
            or not np.array_equal(
                self.selected_feature_indices,
                np.unique(self.selected_feature_indices),
            )
        ):
            raise TypeError("Selected feature indices must be sorted, unique, non-empty int64")
        if len(self.selected_feature_names) != self.selected_feature_indices.size:
            raise ValueError("Selected feature names must match selected indices")
        if (
            self.coefficients.shape != (self.selected_feature_indices.size,)
            or self.coefficients.dtype != np.dtype(np.float64)
            or not np.isfinite(self.coefficients).all()
        ):
            raise TypeError("Classifier coefficients must be a finite float64 feature vector")
        if not np.isfinite(self.intercept):
            raise ValueError("Classifier intercept must be finite")


@dataclass(frozen=True, slots=True)
class FittedCalibratedPixelModels:
    block_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    training_target_indices: NDArray[np.int64]
    training_sample_keys: tuple[tuple[int, int, int], ...]
    cross_validation: GroupedPixelCrossValidation
    models: tuple[FittedCalibratedPixelModel, ...]

    def __post_init__(self) -> None:
        if not self.block_names or not self.feature_names or not self.models:
            raise ValueError("Fitted classifier models require features and pixel models")
        if (
            self.training_target_indices.ndim != 1
            or self.training_target_indices.dtype != np.dtype(np.int64)
        ):
            raise TypeError("Training target indices must be a one-dimensional int64 array")
        if len(self.training_sample_keys) != self.training_target_indices.size:
            raise ValueError("Training sample keys must match training rows")
        if self.cross_validation.n_samples != self.training_target_indices.size:
            raise ValueError("Cross-validation must match training rows")
        if len(self.models) != self.cross_validation.n_pixels:
            raise ValueError("Exactly one fitted classifier is required per pixel")
        if tuple(model.pixel_index for model in self.models) != tuple(range(len(self.models))):
            raise ValueError("Fitted classifier models must use contiguous pixel order")


class _CalibratedClassifierBackend:
    spec: ModelSpec

    def fit(
        self,
        training_features: AlignedTrainingFeatures,
        *,
        targets: PixelTargetDataset,
        config: RandomImageryExperimentConfigLike,
    ) -> FittedDirectionModel[FittedCalibratedPixelModels, FeatureScreeningResult]:
        classifier_config = _require_classifier_config(config, spec=self.spec)
        y_train = targets.y[training_features.target_row_indices]
        cross_validation = build_grouped_pixel_cross_validation(
            y_train,
            groups=training_features.subject_ids,
            config=classifier_config.cross_validation,
        )
        screening = screen_classifier_feature_families(
            training_features,
            y=y_train,
            cross_validation=cross_validation,
            config=classifier_config,
        )
        selected = _select_training_partition(
            training_features,
            block_names=screening.selected_block_names,
        )
        fitted_models = fit_calibrated_pixel_classifiers(
            selected,
            y_train=y_train,
            pixel_names=targets.pixel_names,
            cross_validation=cross_validation,
            config=classifier_config,
        )
        return FittedDirectionModel(
            spec=self.spec,
            selected_block_names=screening.selected_block_names,
            feature_names=selected.feature_names,
            training_target_indices=selected.target_row_indices,
            training_sample_keys=selected.sample_keys,
            payload=fitted_models,
            selection=screening,
        )

    def predict(
        self,
        fitted: FittedDirectionModel[
            FittedCalibratedPixelModels,
            FeatureScreeningResult,
        ],
        *,
        test_features: AlignedFeaturePartition,
        targets: PixelTargetDataset,
        config: RandomImageryExperimentConfigLike,
    ) -> ModelPrediction:
        classifier_config = _require_classifier_config(config, spec=self.spec)
        if test_features.block_names != fitted.payload.block_names:
            raise ValueError("Train and test feature families differ")
        if test_features.feature_names != fitted.payload.feature_names:
            raise ValueError("Train and test feature names or channel order differ")

        n_test = test_features.X.shape[0]
        n_pixels = len(fitted.payload.models)
        scores = np.empty((n_test, n_pixels), dtype=np.float64)
        for model in fitted.payload.models:
            decision_scores = _decision_function(model.pipeline, test_features.X)
            scores[:, model.pixel_index] = _positive_probabilities(
                model.calibration.calibrator,
                decision_scores,
            )
        predictions = (scores >= classifier_config.prediction_threshold).astype(np.int8)
        scores.setflags(write=False)
        predictions.setflags(write=False)
        return ModelPrediction(
            test_target_indices=test_features.target_row_indices,
            test_sample_keys=test_features.sample_keys,
            scores=scores,
            predictions=predictions,
            threshold=classifier_config.prediction_threshold,
            diagnostics=ScoreDiagnostics(score_semantics=self.spec.score_semantics),
        )


class LinearSVMBackend(_CalibratedClassifierBackend):
    spec = get_model_spec("linear-svm-independent")


class RidgeClassifierBackend(_CalibratedClassifierBackend):
    spec = get_model_spec("ridge-classifier-independent")


def screen_classifier_feature_families(
    features: AlignedTrainingFeatures,
    *,
    y: ArrayLike,
    cross_validation: GroupedPixelCrossValidation,
    config: CalibratedClassifierExperimentConfig,
) -> FeatureScreeningResult:
    targets = _validated_targets(
        y,
        n_rows=features.target_row_indices.size,
        n_pixels=cross_validation.n_pixels,
    )
    if tuple(family.block_names for family in features.families) != (
        config.feature_screening.candidates
    ):
        raise ValueError("Aligned feature families must preserve configured candidate order")

    candidate_results: list[CandidateScreeningResult] = []
    for family in features.families:
        fold_scores = np.empty(
            (cross_validation.n_pixels, cross_validation.n_splits),
            dtype=np.float64,
        )
        selected_counts = np.empty_like(fold_scores, dtype=np.int64)
        for pixel_index in range(cross_validation.n_pixels):
            pixel_targets = targets[:, pixel_index]
            for fold in cross_validation.for_pixel(pixel_index):
                pipeline = _build_classifier_pipeline(
                    config=config,
                    selection=config.feature_screening,
                    regularization=config.feature_screening.regularization,
                    class_weight=config.feature_screening.class_weight,
                    cache_dir=None,
                )
                _fit_with_convergence_errors(
                    pipeline,
                    family.X[fold.train_indices],
                    pixel_targets[fold.train_indices],
                )
                predictions = pipeline.predict(family.X[fold.validation_indices])
                fold_scores[pixel_index, fold.fold_index] = balanced_accuracy_score(
                    pixel_targets[fold.validation_indices],
                    predictions,
                )
                selector = pipeline.named_steps["select"]
                if not isinstance(selector, CappedSelectKBest):
                    raise TypeError("Unexpected classifier screening selector type")
                selected_counts[pixel_index, fold.fold_index] = int(
                    selector.get_support().sum()
                )

        mean_pixel_scores = fold_scores.mean(axis=1, dtype=np.float64)
        for array in (fold_scores, selected_counts, mean_pixel_scores):
            array.setflags(write=False)
        candidate_results.append(
            CandidateScreeningResult(
                block_names=family.block_names,
                fold_scores=fold_scores,
                selected_feature_counts=selected_counts,
                mean_pixel_scores=mean_pixel_scores,
                mean_score=float(mean_pixel_scores.mean()),
            )
        )

    best_index = int(np.argmax([candidate.mean_score for candidate in candidate_results]))
    return FeatureScreeningResult(
        candidates=tuple(candidate_results),
        selected_block_names=candidate_results[best_index].block_names,
    )


def fit_calibrated_pixel_classifiers(
    training_features: AlignedFeaturePartition,
    *,
    y_train: ArrayLike,
    pixel_names: tuple[str, ...],
    cross_validation: GroupedPixelCrossValidation,
    config: CalibratedClassifierExperimentConfig,
) -> FittedCalibratedPixelModels:
    targets = _validated_targets(
        y_train,
        n_rows=training_features.X.shape[0],
        n_pixels=len(pixel_names),
    )
    if targets.shape != (cross_validation.n_samples, cross_validation.n_pixels):
        raise ValueError("Training targets must match grouped cross-validation")
    if config.cross_validation.scoring != "balanced_accuracy":
        raise ValueError(
            f"Unsupported classifier grid-search scoring: "
            f"{config.cross_validation.scoring!r}"
        )

    models: list[FittedCalibratedPixelModel] = []
    for pixel_index, pixel_name in enumerate(pixel_names):
        folds = cross_validation.for_pixel(pixel_index)
        cv = tuple((fold.train_indices, fold.validation_indices) for fold in folds)
        with tempfile.TemporaryDirectory(
            prefix=f"{config.grid_search.estimator_family}-pixel-{pixel_index:02d}-"
        ) as cache_dir:
            pipeline = _build_grid_pipeline(config=config, cache_dir=Path(cache_dir))
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=_build_parameter_grid(config.grid_search),
                scoring=config.cross_validation.scoring,
                cv=cv,
                refit=True,
                n_jobs=config.grid_search.n_jobs,
                error_score=config.grid_search.error_score,
                return_train_score=False,
            )
            _fit_with_convergence_errors(
                search,
                training_features.X,
                targets[:, pixel_index],
            )
            best_pipeline = search.best_estimator_
            best_pipeline.set_params(memory=None)

        calibration = _fit_oof_platt_calibration(
            best_pipeline,
            X=training_features.X,
            y=targets[:, pixel_index],
            cross_validation=cross_validation,
            pixel_index=pixel_index,
            config=config,
        )
        models.append(
            _build_fitted_pixel_model(
                pixel_index=pixel_index,
                pixel_name=pixel_name,
                pipeline=best_pipeline,
                calibration=calibration,
                search=search,
                feature_names=training_features.feature_names,
                estimator_family=config.grid_search.estimator_family,
            )
        )

    return FittedCalibratedPixelModels(
        block_names=training_features.block_names,
        feature_names=training_features.feature_names,
        training_target_indices=training_features.target_row_indices,
        training_sample_keys=training_features.sample_keys,
        cross_validation=cross_validation,
        models=tuple(models),
    )


def _fit_oof_platt_calibration(
    fitted_pipeline: Pipeline,
    *,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.int8],
    cross_validation: GroupedPixelCrossValidation,
    pixel_index: int,
    config: CalibratedClassifierExperimentConfig,
) -> PlattCalibration:
    oof_scores = np.empty(y.shape[0], dtype=np.float64)
    fold_indices = np.full(y.shape[0], -1, dtype=np.int64)
    for fold in cross_validation.for_pixel(pixel_index):
        fold_pipeline = clone(fitted_pipeline)
        _fit_with_convergence_errors(
            fold_pipeline,
            X[fold.train_indices],
            y[fold.train_indices],
        )
        oof_scores[fold.validation_indices] = _decision_function(
            fold_pipeline,
            X[fold.validation_indices],
        )
        if np.any(fold_indices[fold.validation_indices] != -1):
            raise ValueError("Calibration validation folds overlap")
        fold_indices[fold.validation_indices] = fold.fold_index
    if np.any(fold_indices < 0):
        raise ValueError("Calibration folds do not cover every training row")

    calibration_config = config.calibration
    calibrator = LogisticRegression(
        C=calibration_config.c,
        l1_ratio=0.0,
        solver=calibration_config.solver,
        max_iter=calibration_config.max_iter,
        tol=calibration_config.tolerance,
        random_state=config.random_state,
    )
    _fit_with_convergence_errors(calibrator, oof_scores.reshape(-1, 1), y)
    if calibrator.classes_.tolist() != [0, 1]:
        raise ValueError("Platt calibrator must retain binary classes [0, 1]")
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    oof_scores.setflags(write=False)
    fold_indices.setflags(write=False)
    return PlattCalibration(
        calibrator=calibrator,
        oof_decision_scores=oof_scores,
        oof_fold_indices=fold_indices,
        coefficient=coefficient,
        intercept=intercept,
    )


def _build_grid_pipeline(
    *,
    config: CalibratedClassifierExperimentConfig,
    cache_dir: Path,
) -> Pipeline:
    return _build_classifier_pipeline(
        config=config,
        selection=config.feature_screening,
        regularization=_regularization_values(config.grid_search)[0],
        class_weight=config.grid_search.class_weights[0],
        cache_dir=cache_dir,
    )


def _build_classifier_pipeline(
    *,
    config: CalibratedClassifierExperimentConfig,
    selection: ClassifierFeatureScreeningConfig,
    regularization: float,
    class_weight: str | None,
    cache_dir: Path | None,
) -> Pipeline:
    return Pipeline(
        steps=(
            ("variance", VarianceThreshold(threshold=selection.variance_threshold)),
            ("select", CappedSelectKBest(k=selection.select_k)),
            ("scale", StandardScaler()),
            (
                "model",
                _build_classifier(
                    config=config,
                    regularization=regularization,
                    class_weight=class_weight,
                ),
            ),
        ),
        memory=None if cache_dir is None else Memory(location=cache_dir, verbose=0),
    )


def _build_classifier(
    *,
    config: CalibratedClassifierExperimentConfig,
    regularization: float,
    class_weight: str | None,
) -> LinearSVC | RidgeClassifier:
    grid = config.grid_search
    if isinstance(grid, LinearSVMGridSearchConfig):
        return LinearSVC(
            C=regularization,
            class_weight=class_weight,
            dual="auto",
            max_iter=grid.max_iter,
            tol=grid.tolerance,
            random_state=config.random_state,
        )
    if isinstance(grid, RidgeClassifierGridSearchConfig):
        return RidgeClassifier(
            alpha=regularization,
            class_weight=class_weight,
            solver=grid.solver,
            tol=grid.tolerance,
            random_state=config.random_state,
        )
    raise TypeError(f"Unsupported classifier grid: {type(grid).__name__}")


def _build_parameter_grid(
    config: LinearSVMGridSearchConfig | RidgeClassifierGridSearchConfig,
) -> dict[str, tuple[object, ...]]:
    regularization_name = (
        "model__C"
        if isinstance(config, LinearSVMGridSearchConfig)
        else "model__alpha"
    )
    return {
        "select__k": tuple(config.select_k),
        regularization_name: tuple(_regularization_values(config)),
        "model__class_weight": tuple(config.class_weights),
    }


def _build_fitted_pixel_model(
    *,
    pixel_index: int,
    pixel_name: str,
    pipeline: Pipeline,
    calibration: PlattCalibration,
    search: GridSearchCV,
    feature_names: tuple[str, ...],
    estimator_family: ClassifierFamily,
) -> FittedCalibratedPixelModel:
    variance = pipeline.named_steps["variance"]
    selector = pipeline.named_steps["select"]
    classifier = pipeline.named_steps["model"]
    if not isinstance(variance, VarianceThreshold):
        raise TypeError("Unexpected fitted variance selector type")
    if not isinstance(selector, CappedSelectKBest):
        raise TypeError("Unexpected fitted univariate selector type")
    if not isinstance(classifier, (LinearSVC, RidgeClassifier)):
        raise TypeError("Unexpected fitted calibrated classifier type")

    variance_indices = variance.get_support(indices=True).astype(np.int64, copy=False)
    selected_indices = variance_indices[selector.get_support(indices=True)]
    selected_indices = np.sort(selected_indices.astype(np.int64, copy=False))
    selected_feature_names = tuple(feature_names[int(index)] for index in selected_indices)
    coefficients = np.asarray(classifier.coef_, dtype=np.float64).reshape(-1)
    if coefficients.shape != (selected_indices.size,):
        raise ValueError("Classifier coefficients do not match selected features")
    intercept_values = np.asarray(classifier.intercept_, dtype=np.float64).reshape(-1)
    if intercept_values.shape != (1,):
        raise ValueError("Binary classifier must expose exactly one intercept")
    selected_indices.setflags(write=False)
    coefficients.setflags(write=False)

    candidate_scores = tuple(
        ClassifierCandidateScore(
            hyperparameters=_params_to_hyperparameters(
                params,
                estimator_family=estimator_family,
            ),
            mean_score=float(mean_score),
            std_score=float(std_score),
            rank=int(rank),
        )
        for params, mean_score, std_score, rank in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
            search.cv_results_["rank_test_score"],
            strict=True,
        )
    )
    return FittedCalibratedPixelModel(
        pixel_index=pixel_index,
        pixel_name=pixel_name,
        pipeline=pipeline,
        calibration=calibration,
        best_hyperparameters=_params_to_hyperparameters(
            search.best_params_,
            estimator_family=estimator_family,
        ),
        best_cv_score=float(search.best_score_),
        candidate_scores=candidate_scores,
        selected_feature_indices=selected_indices,
        selected_feature_names=selected_feature_names,
        coefficients=coefficients,
        intercept=float(intercept_values[0]),
    )


def _params_to_hyperparameters(
    params: dict[str, Any],
    *,
    estimator_family: ClassifierFamily,
) -> ClassifierHyperparameters:
    regularization_key = (
        "model__C" if estimator_family == "linear_svm" else "model__alpha"
    )
    return ClassifierHyperparameters(
        estimator_family=estimator_family,
        select_k=int(params["select__k"]),
        regularization=float(params[regularization_key]),
        class_weight=params["model__class_weight"],
    )


def _regularization_values(
    config: LinearSVMGridSearchConfig | RidgeClassifierGridSearchConfig,
) -> tuple[float, ...]:
    if isinstance(config, LinearSVMGridSearchConfig):
        return config.c_values
    return config.alpha_values


def _decision_function(
    pipeline: Pipeline,
    X: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    values = np.asarray(pipeline.decision_function(X), dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.shape != (X.shape[0],) or not np.isfinite(values).all():
        raise ValueError("Binary classifier decision scores must be a finite row vector")
    return values


def _positive_probabilities(
    calibrator: LogisticRegression,
    decision_scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    positive_columns = np.flatnonzero(calibrator.classes_ == 1)
    if positive_columns.size != 1:
        raise ValueError("Platt calibrator lacks exactly one positive class")
    probabilities = calibrator.predict_proba(decision_scores.reshape(-1, 1))[
        :,
        int(positive_columns[0]),
    ].astype(np.float64, copy=False)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Platt-calibrated probabilities must be finite and in [0, 1]")
    return probabilities


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
    n_rows: int,
    n_pixels: int,
) -> NDArray[np.int8]:
    targets = np.asarray(y)
    if targets.shape != (n_rows, n_pixels):
        raise ValueError("Binary targets do not match expected rows and pixels")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Classifier targets must be binary")
    return targets.astype(np.int8, copy=False)


def _fit_with_convergence_errors(
    estimator: Any,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.int8],
) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return estimator.fit(X, y)


def _require_classifier_config(
    config: RandomImageryExperimentConfigLike,
    *,
    spec: ModelSpec,
) -> CalibratedClassifierExperimentConfig:
    if not isinstance(config, CalibratedClassifierExperimentConfig):
        raise TypeError("Calibrated classifier backend requires its experiment configuration")
    if config.model_id != spec.model_id:
        raise ValueError(
            f"Backend model {spec.model_id!r} does not match config model {config.model_id!r}"
        )
    return config


__all__ = [
    "ClassifierCandidateScore",
    "ClassifierHyperparameters",
    "FittedCalibratedPixelModel",
    "FittedCalibratedPixelModels",
    "LinearSVMBackend",
    "PlattCalibration",
    "RidgeClassifierBackend",
    "fit_calibrated_pixel_classifiers",
    "screen_classifier_feature_families",
]
