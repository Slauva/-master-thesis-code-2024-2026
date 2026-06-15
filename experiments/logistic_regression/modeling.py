import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Memory
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.logistic_regression.config import (
    CrossValidationConfig,
    GridSearchConfig,
)
from experiments.logistic_regression.schemas import (
    AlignedFeaturePartition,
    FittedPixelModel,
    FittedPixelModels,
    GridCandidateScore,
    GroupedPixelCrossValidation,
    PixelGridSearchResult,
    PixelHyperparameters,
    PixelTargetDataset,
    SubjectSplit,
)
from experiments.logistic_regression.screening import (
    CappedSelectKBest,
    build_grouped_pixel_cross_validation,
)
from features import FeatureSet, flatten_feature_set


class FeatureSetDataset(Sequence[FeatureSet]):
    def __getitem__(self, key: int | tuple[int, int, int]) -> FeatureSet: ...


def build_aligned_feature_partition(
    dataset: FeatureSetDataset,
    *,
    targets: PixelTargetDataset,
    row_indices: NDArray[np.int64],
    block_names: tuple[str, ...],
) -> AlignedFeaturePartition:
    if row_indices.ndim != 1 or row_indices.dtype != np.dtype(np.int64):
        raise TypeError("Feature partition row indices must be a one-dimensional int64 array")
    if row_indices.size < 1 or np.any(row_indices < 0) or np.any(row_indices >= targets.y.shape[0]):
        raise ValueError("Feature partition row indices must be non-empty and in range")
    if not np.array_equal(row_indices, np.unique(row_indices)):
        raise ValueError("Feature partition row indices must be sorted and unique")
    if not block_names or len(set(block_names)) != len(block_names):
        raise ValueError("Feature partition block names must be non-empty and unique")

    matrices: list[NDArray[np.floating[Any]]] = []
    expected_names: tuple[str, ...] | None = None
    sample_keys: list[tuple[int, int, int]] = []
    subject_ids: list[int] = []
    window_bounds: list[NDArray[np.float64]] = []
    for target_index in row_indices:
        row_index = int(target_index)
        expected_key = targets.sample_keys[row_index]
        feature_set = dataset[expected_key]
        actual_key = (
            feature_set.sample.subject_id,
            feature_set.sample.trial_number,
            feature_set.sample.block_index,
        )
        if actual_key != expected_key:
            raise ValueError(f"Feature row {actual_key} does not match target row {expected_key}")
        if feature_set.window_bounds_seconds.shape != (1, 2):
            raise ValueError("Pixel reconstruction requires exactly one full-epoch feature row per block")

        matrix, feature_names = flatten_feature_set(feature_set, block_names=block_names)
        if matrix.shape[0] != 1:
            raise ValueError("Pixel reconstruction requires exactly one feature row per block")
        if expected_names is None:
            expected_names = feature_names
        elif feature_names != expected_names:
            raise ValueError("Feature names or EEG channel order differ between partition samples")
        matrices.append(matrix)
        sample_keys.append(expected_key)
        subject_ids.append(int(targets.subject_ids[row_index]))
        window_bounds.append(feature_set.window_bounds_seconds)

    if expected_names is None:
        raise RuntimeError("Feature partition produced no feature names")
    X = np.concatenate(matrices, axis=0)
    target_row_indices = row_indices.copy()
    partition_subject_ids = np.asarray(subject_ids, dtype=np.int64)
    partition_bounds = np.concatenate(window_bounds, axis=0).astype(np.float64, copy=False)
    for array in (X, target_row_indices, partition_subject_ids, partition_bounds):
        array.setflags(write=False)
    return AlignedFeaturePartition(
        block_names=block_names,
        X=X,
        feature_names=expected_names,
        target_row_indices=target_row_indices,
        sample_keys=tuple(sample_keys),
        subject_ids=partition_subject_ids,
        window_bounds_seconds=partition_bounds,
    )


def fit_pixel_models(
    training_features: AlignedFeaturePartition,
    *,
    y_train: ArrayLike,
    pixel_names: tuple[str, ...],
    cross_validation: GroupedPixelCrossValidation,
    config: GridSearchConfig,
    scoring: str,
    random_state: int,
) -> FittedPixelModels:
    targets = np.asarray(y_train)
    if targets.shape != (training_features.X.shape[0], len(pixel_names)):
        raise ValueError("Training targets must match feature rows and pixel names")
    if targets.shape != (cross_validation.n_samples, cross_validation.n_pixels):
        raise ValueError("Training targets must match grouped cross-validation")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Training pixel targets must be binary")
    if scoring != "balanced_accuracy":
        raise ValueError(f"Unsupported grid-search scoring: {scoring!r}")

    models: list[FittedPixelModel] = []
    for pixel_index, pixel_name in enumerate(pixel_names):
        folds = cross_validation.for_pixel(pixel_index)
        cv = tuple((fold.train_indices, fold.validation_indices) for fold in folds)
        with tempfile.TemporaryDirectory(prefix=f"logistic-pixel-{pixel_index:02d}-") as cache_dir:
            pipeline = _build_grid_pipeline(
                config=config,
                random_state=random_state,
                cache_dir=Path(cache_dir),
            )
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=_build_parameter_grid(config),
                scoring=scoring,
                cv=cv,
                refit=True,
                n_jobs=config.n_jobs,
                error_score=config.error_score,
                return_train_score=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                search.fit(training_features.X, targets[:, pixel_index])
            best_pipeline = search.best_estimator_
            best_pipeline.set_params(memory=None)

        models.append(
            _build_fitted_pixel_model(
                pixel_index=pixel_index,
                pixel_name=pixel_name,
                pipeline=best_pipeline,
                search=search,
                feature_names=training_features.feature_names,
            )
        )

    return FittedPixelModels(
        block_names=training_features.block_names,
        feature_names=training_features.feature_names,
        training_target_indices=training_features.target_row_indices,
        training_sample_keys=training_features.sample_keys,
        cross_validation=cross_validation,
        models=tuple(models),
    )


def predict_pixel_models(
    fitted_models: FittedPixelModels,
    *,
    test_features: AlignedFeaturePartition,
    y_test: ArrayLike,
    threshold: float,
) -> PixelGridSearchResult:
    targets = np.asarray(y_test)
    n_pixels = len(fitted_models.models)
    if test_features.block_names != fitted_models.block_names:
        raise ValueError("Train and test feature families differ")
    if test_features.feature_names != fitted_models.feature_names:
        raise ValueError("Train and test feature names or channel order differ")
    if targets.shape != (test_features.X.shape[0], n_pixels):
        raise ValueError("Test targets must match feature rows and fitted pixel models")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Test pixel targets must be binary")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Prediction threshold must be between zero and one")

    probabilities = np.empty((test_features.X.shape[0], n_pixels), dtype=np.float64)
    for model in fitted_models.models:
        classifier = model.pipeline.named_steps["model"]
        if not isinstance(classifier, LogisticRegression):
            raise TypeError("Unexpected fitted classifier type")
        positive_columns = np.flatnonzero(classifier.classes_ == 1)
        if positive_columns.size != 1:
            raise ValueError(f"Pixel {model.pixel_index} classifier lacks one positive class")
        probabilities[:, model.pixel_index] = model.pipeline.predict_proba(test_features.X)[
            :,
            int(positive_columns[0]),
        ]

    predictions = (probabilities >= threshold).astype(np.int8)
    test_balanced_accuracy = np.asarray(
        [
            balanced_accuracy_score(targets[:, pixel_index], predictions[:, pixel_index])
            for pixel_index in range(n_pixels)
        ],
        dtype=np.float64,
    )
    probabilities.setflags(write=False)
    predictions.setflags(write=False)
    test_balanced_accuracy.setflags(write=False)
    return PixelGridSearchResult(
        fitted_models=fitted_models,
        test_target_indices=test_features.target_row_indices,
        test_sample_keys=test_features.sample_keys,
        probabilities=probabilities,
        predictions=predictions,
        test_balanced_accuracy=test_balanced_accuracy,
        threshold=threshold,
    )


def run_per_pixel_grid_search(
    dataset: FeatureSetDataset,
    *,
    targets: PixelTargetDataset,
    split: SubjectSplit,
    block_names: tuple[str, ...],
    cross_validation_config: CrossValidationConfig,
    grid_search_config: GridSearchConfig,
    scoring: str,
    threshold: float,
    random_state: int,
) -> PixelGridSearchResult:
    training_features = build_aligned_feature_partition(
        dataset,
        targets=targets,
        row_indices=split.train_indices,
        block_names=block_names,
    )
    y_train = targets.y[training_features.target_row_indices]
    cross_validation = build_grouped_pixel_cross_validation(
        y_train,
        groups=training_features.subject_ids,
        config=cross_validation_config,
    )
    fitted_models = fit_pixel_models(
        training_features,
        y_train=y_train,
        pixel_names=targets.pixel_names,
        cross_validation=cross_validation,
        config=grid_search_config,
        scoring=scoring,
        random_state=random_state,
    )

    test_features = build_aligned_feature_partition(
        dataset,
        targets=targets,
        row_indices=split.test_indices,
        block_names=block_names,
    )
    return predict_pixel_models(
        fitted_models,
        test_features=test_features,
        y_test=targets.y[test_features.target_row_indices],
        threshold=threshold,
    )


def _build_grid_pipeline(
    *,
    config: GridSearchConfig,
    random_state: int,
    cache_dir: Path,
) -> Pipeline:
    return Pipeline(
        steps=(
            ("variance", VarianceThreshold(threshold=0.0)),
            ("select", CappedSelectKBest(k=config.select_k[0])),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=config.c_values[0],
                    l1_ratio=_penalty_to_l1_ratio(config.penalties[0]),
                    class_weight=config.class_weights[0],
                    solver=config.solver,
                    max_iter=config.max_iter,
                    random_state=random_state,
                ),
            ),
        ),
        memory=Memory(location=cache_dir, verbose=0),
    )


def _build_parameter_grid(config: GridSearchConfig) -> dict[str, tuple[object, ...]]:
    return {
        "select__k": tuple(config.select_k),
        "model__C": tuple(config.c_values),
        "model__l1_ratio": tuple(_penalty_to_l1_ratio(value) for value in config.penalties),
        "model__class_weight": tuple(config.class_weights),
    }


def _build_fitted_pixel_model(
    *,
    pixel_index: int,
    pixel_name: str,
    pipeline: Pipeline,
    search: GridSearchCV,
    feature_names: tuple[str, ...],
) -> FittedPixelModel:
    variance = pipeline.named_steps["variance"]
    selector = pipeline.named_steps["select"]
    classifier = pipeline.named_steps["model"]
    if not isinstance(variance, VarianceThreshold):
        raise TypeError("Unexpected fitted variance selector type")
    if not isinstance(selector, CappedSelectKBest):
        raise TypeError("Unexpected fitted univariate selector type")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("Unexpected fitted classifier type")

    variance_indices = variance.get_support(indices=True).astype(np.int64, copy=False)
    selected_indices = variance_indices[selector.get_support(indices=True)]
    selected_indices = np.sort(selected_indices.astype(np.int64, copy=False))
    selected_feature_names = tuple(feature_names[int(index)] for index in selected_indices)
    coefficients = np.asarray(classifier.coef_[0], dtype=np.float64)
    if coefficients.shape != (selected_indices.size,):
        raise ValueError("Classifier coefficients do not match selected features")
    selected_indices.setflags(write=False)
    coefficients.setflags(write=False)

    candidate_scores = tuple(
        GridCandidateScore(
            hyperparameters=_params_to_hyperparameters(params),
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
    return FittedPixelModel(
        pixel_index=pixel_index,
        pixel_name=pixel_name,
        pipeline=pipeline,
        best_hyperparameters=_params_to_hyperparameters(search.best_params_),
        best_cv_score=float(search.best_score_),
        candidate_scores=candidate_scores,
        selected_feature_indices=selected_indices,
        selected_feature_names=selected_feature_names,
        coefficients=coefficients,
        intercept=float(classifier.intercept_[0]),
        n_iter=int(classifier.n_iter_[0]),
    )


def _params_to_hyperparameters(params: dict[str, Any]) -> PixelHyperparameters:
    return PixelHyperparameters(
        select_k=int(params["select__k"]),
        c=float(params["model__C"]),
        penalty=_l1_ratio_to_penalty(float(params["model__l1_ratio"])),
        class_weight=params["model__class_weight"],
    )


def _penalty_to_l1_ratio(penalty: str) -> float:
    if penalty == "l1":
        return 1.0
    if penalty == "l2":
        return 0.0
    raise ValueError(f"Unsupported penalty: {penalty!r}")


def _l1_ratio_to_penalty(l1_ratio: float) -> str:
    if l1_ratio == 1.0:
        return "l1"
    if l1_ratio == 0.0:
        return "l2"
    raise ValueError(f"Unsupported Logistic Regression l1_ratio: {l1_ratio!r}")
