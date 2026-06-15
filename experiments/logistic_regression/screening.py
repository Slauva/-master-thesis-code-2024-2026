import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from experiments.logistic_regression.config import CrossValidationConfig, FeatureScreeningConfig
from experiments.logistic_regression.schemas import (
    AlignedTrainingFeatures,
    CandidateScreeningResult,
    FeatureFamily,
    FeatureScreeningResult,
    GroupedPixelCrossValidation,
    PixelFold,
    PixelTargetDataset,
    SubjectSplit,
)
from features import FeatureSet, flatten_feature_set


class FeatureSetDataset(Sequence[FeatureSet]):
    def __getitem__(self, key: int | tuple[int, int, int]) -> FeatureSet: ...


class CappedSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, *, k: int) -> None:
        self.k = k

    def fit(self, X: ArrayLike, y: ArrayLike) -> "CappedSelectKBest":
        values = np.asarray(X)
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("`k` must be a positive integer")
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("Feature selection requires a non-empty two-dimensional matrix")
        self.k_ = min(self.k, values.shape[1])
        self.selector_ = SelectKBest(score_func=_stable_f_classif, k=self.k_).fit(values, y)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.floating[Any]]:
        check_is_fitted(self, ("selector_", "k_", "n_features_in_"))
        return self.selector_.transform(X)

    def get_support(self, indices: bool = False) -> NDArray[np.bool_] | NDArray[np.int64]:
        check_is_fitted(self, "selector_")
        return self.selector_.get_support(indices=indices)


def build_aligned_training_features(
    dataset: FeatureSetDataset,
    *,
    targets: PixelTargetDataset,
    split: SubjectSplit,
    candidates: tuple[tuple[str, ...], ...],
) -> AlignedTrainingFeatures:
    if not candidates or any(not candidate for candidate in candidates):
        raise ValueError("Feature-family candidates must be non-empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Feature-family candidates must be unique")

    matrices: list[list[NDArray[np.floating[Any]]]] = [[] for _ in candidates]
    expected_names: list[tuple[str, ...] | None] = [None] * len(candidates)
    sample_keys: list[tuple[int, int, int]] = []
    subject_ids: list[int] = []
    window_bounds: list[NDArray[np.float64]] = []

    for target_index in split.train_indices:
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

        for candidate_index, candidate in enumerate(candidates):
            matrix, feature_names = flatten_feature_set(feature_set, block_names=candidate)
            if matrix.shape[0] != 1:
                raise ValueError("Pixel reconstruction requires exactly one feature row per block")
            if expected_names[candidate_index] is None:
                expected_names[candidate_index] = feature_names
            elif feature_names != expected_names[candidate_index]:
                raise ValueError("Feature names or EEG channel order differ between training samples")
            matrices[candidate_index].append(matrix)

        sample_keys.append(expected_key)
        subject_ids.append(int(targets.subject_ids[row_index]))
        window_bounds.append(feature_set.window_bounds_seconds)

    families: list[FeatureFamily] = []
    for candidate, candidate_matrices, feature_names in zip(
        candidates,
        matrices,
        expected_names,
        strict=True,
    ):
        if feature_names is None:
            raise RuntimeError("Feature alignment produced no feature names")
        X = np.concatenate(candidate_matrices, axis=0)
        X.setflags(write=False)
        families.append(
            FeatureFamily(
                block_names=candidate,
                X=X,
                feature_names=feature_names,
            )
        )

    target_row_indices = split.train_indices.copy()
    aligned_subject_ids = np.asarray(subject_ids, dtype=np.int64)
    aligned_window_bounds = np.concatenate(window_bounds, axis=0).astype(np.float64, copy=False)
    for array in (target_row_indices, aligned_subject_ids, aligned_window_bounds):
        array.setflags(write=False)
    return AlignedTrainingFeatures(
        families=tuple(families),
        target_row_indices=target_row_indices,
        sample_keys=tuple(sample_keys),
        subject_ids=aligned_subject_ids,
        window_bounds_seconds=aligned_window_bounds,
    )


def build_grouped_pixel_cross_validation(
    y: ArrayLike,
    *,
    groups: ArrayLike,
    config: CrossValidationConfig,
) -> GroupedPixelCrossValidation:
    targets = np.asarray(y)
    subject_groups = np.asarray(groups)
    if targets.ndim != 2 or targets.shape[0] < 1 or targets.shape[1] < 1:
        raise ValueError("Targets must have shape (sample, pixel) with non-empty axes")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Pixel targets must be binary")
    if subject_groups.shape != (targets.shape[0],):
        raise ValueError("Groups must match the target rows")
    if np.unique(subject_groups).size < config.n_splits:
        raise ValueError("Grouped cross-validation requires at least one subject per split")

    folds: list[PixelFold] = []
    for pixel_index in range(targets.shape[1]):
        pixel_targets = targets[:, pixel_index]
        splitter = StratifiedGroupKFold(
            n_splits=config.n_splits,
            shuffle=config.shuffle,
            random_state=config.random_state,
        )
        validation_rows: list[NDArray[np.int64]] = []
        for fold_index, (train_indices, validation_indices) in enumerate(
            splitter.split(np.zeros((targets.shape[0], 1)), pixel_targets, subject_groups)
        ):
            train_indices = np.sort(train_indices.astype(np.int64, copy=False))
            validation_indices = np.sort(validation_indices.astype(np.int64, copy=False))
            if np.unique(pixel_targets[train_indices]).size != 2:
                raise ValueError(f"Pixel {pixel_index} fold {fold_index} training rows lack both classes")
            if np.unique(pixel_targets[validation_indices]).size != 2:
                raise ValueError(f"Pixel {pixel_index} fold {fold_index} validation rows lack both classes")

            train_indices.setflags(write=False)
            validation_indices.setflags(write=False)
            validation_rows.append(validation_indices)
            folds.append(
                PixelFold(
                    pixel_index=pixel_index,
                    fold_index=fold_index,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    train_subjects=tuple(int(value) for value in np.unique(subject_groups[train_indices])),
                    validation_subjects=tuple(
                        int(value) for value in np.unique(subject_groups[validation_indices])
                    ),
                    n_samples=targets.shape[0],
                )
            )
        if not np.array_equal(
            np.sort(np.concatenate(validation_rows)),
            np.arange(targets.shape[0]),
        ):
            raise ValueError(f"Pixel {pixel_index} validation folds do not partition training rows")

    return GroupedPixelCrossValidation(
        folds=tuple(folds),
        n_samples=targets.shape[0],
        n_pixels=targets.shape[1],
        n_splits=config.n_splits,
        random_state=config.random_state,
    )


def screen_feature_families(
    features: AlignedTrainingFeatures,
    *,
    y: ArrayLike,
    cross_validation: GroupedPixelCrossValidation,
    config: FeatureScreeningConfig,
    random_state: int,
) -> FeatureScreeningResult:
    targets = np.asarray(y)
    if targets.shape != (cross_validation.n_samples, cross_validation.n_pixels):
        raise ValueError("Targets must match the grouped cross-validation dimensions")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Pixel targets must be binary")
    if targets.shape[0] != features.target_row_indices.size:
        raise ValueError("Targets must match the aligned feature rows")
    if tuple(family.block_names for family in features.families) != config.candidates:
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
                pipeline = _build_screening_pipeline(
                    config=config,
                    random_state=random_state,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    pipeline.fit(family.X[fold.train_indices], pixel_targets[fold.train_indices])
                predictions = pipeline.predict(family.X[fold.validation_indices])
                fold_scores[pixel_index, fold.fold_index] = balanced_accuracy_score(
                    pixel_targets[fold.validation_indices],
                    predictions,
                )
                selector = pipeline.named_steps["select"]
                if not isinstance(selector, CappedSelectKBest):
                    raise TypeError("Unexpected screening selector type")
                selected_counts[pixel_index, fold.fold_index] = int(selector.get_support().sum())

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


def _build_screening_pipeline(
    *,
    config: FeatureScreeningConfig,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        steps=(
            ("variance", VarianceThreshold(threshold=config.variance_threshold)),
            ("select", CappedSelectKBest(k=config.select_k)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=config.c,
                    l1_ratio=0.0,
                    class_weight=config.class_weight,
                    solver=config.solver,
                    max_iter=config.max_iter,
                    random_state=random_state,
                ),
            ),
        )
    )


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
