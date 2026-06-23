from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.logistic_regression.baselines import build_non_eeg_baselines
from experiments.logistic_regression.config import LogisticRegressionExperimentConfig
from experiments.logistic_regression.data import (
    build_evaluation_protocol,
    build_random_imagery_targets,
)
from experiments.logistic_regression.metrics import (
    PredictionMetrics,
    SubjectBootstrapInterval,
    bootstrap_subject_mean_balanced_accuracy,
    evaluate_prediction_matrix,
)
from experiments.logistic_regression.modeling import (
    build_aligned_feature_partition,
    fit_pixel_models,
    predict_pixel_models,
)
from experiments.logistic_regression.schemas import (
    AlignedFeaturePartition,
    AlignedTrainingFeatures,
    BaselinePrediction,
    EvaluationDirection,
    EvaluationProtocol,
    EvaluationProtocolDefinition,
    FeatureScreeningResult,
    PixelGridSearchResult,
    PixelTargetDataset,
    ProtocolLeakageAudit,
)
from experiments.logistic_regression.screening import (
    FeatureSetDataset,
    build_aligned_training_features,
    build_grouped_pixel_cross_validation,
    screen_feature_families,
)


@dataclass(frozen=True, slots=True)
class BaselineEvaluation:
    prediction: BaselinePrediction
    metrics: PredictionMetrics


@dataclass(frozen=True, slots=True)
class DirectionEvaluationResult:
    direction: EvaluationDirection
    audit: ProtocolLeakageAudit
    screening: FeatureScreeningResult
    grid_search: PixelGridSearchResult
    model_metrics: PredictionMetrics
    model_bootstrap: SubjectBootstrapInterval
    baselines: tuple[BaselineEvaluation, ...]

    def __post_init__(self) -> None:
        if self.audit.direction_name != self.direction.name:
            raise ValueError("Direction result audit does not match its direction")
        if self.audit.has_forbidden_leakage:
            raise ValueError("Direction result contains forbidden leakage")
        if self.grid_search.test_target_indices.shape != self.direction.test_indices.shape:
            raise ValueError("Direction result test rows do not match its split")
        if not np.array_equal(
            self.grid_search.test_target_indices,
            self.direction.test_indices,
        ):
            raise ValueError("Direction result test rows do not preserve split order")
        if tuple(item.prediction.name for item in self.baselines) != (
            "global_majority",
            "pixel_frequency",
            "seeded_bernoulli",
        ):
            raise ValueError("Direction result must contain the canonical baselines")


@dataclass(frozen=True, slots=True)
class CombinedProtocolEvaluation:
    protocol: EvaluationProtocol
    direction_names: tuple[str, ...]
    target_indices: NDArray[np.int64]
    subject_ids: NDArray[np.int64]
    y_true: NDArray[np.int8]
    probabilities: NDArray[np.float64]
    predictions: NDArray[np.int8]
    model_metrics: PredictionMetrics
    model_bootstrap: SubjectBootstrapInterval
    baselines: tuple[BaselineEvaluation, ...]

    def __post_init__(self) -> None:
        if self.protocol != "within-subject":
            raise ValueError("Only bidirectional within-subject results are combined")
        n_rows, n_pixels = self.y_true.shape
        if self.target_indices.shape != (n_rows,) or self.target_indices.dtype != np.dtype(np.int64):
            raise TypeError("Combined target indices must be an int64 vector")
        if self.subject_ids.shape != (n_rows,) or self.subject_ids.dtype != np.dtype(np.int64):
            raise TypeError("Combined subject IDs must be an int64 vector")
        if self.y_true.dtype != np.dtype(np.int8) or not np.isin(self.y_true, (0, 1)).all():
            raise TypeError("Combined targets must be a binary int8 matrix")
        if self.probabilities.shape != (n_rows, n_pixels) or self.probabilities.dtype != np.dtype(np.float64):
            raise TypeError("Combined probabilities must be a float64 matrix matching targets")
        if self.predictions.shape != (n_rows, n_pixels) or self.predictions.dtype != np.dtype(np.int8):
            raise TypeError("Combined predictions must be an int8 matrix matching targets")
        if not np.isfinite(self.probabilities).all() or not np.isin(self.predictions, (0, 1)).all():
            raise ValueError("Combined predictions and probabilities must be valid")
        if len(set(self.target_indices.tolist())) != n_rows:
            raise ValueError("Combined directions must evaluate every target row at most once")


@dataclass(frozen=True, slots=True)
class ProtocolEvaluationResult:
    definition: EvaluationProtocolDefinition
    directions: tuple[DirectionEvaluationResult, ...]
    combined: CombinedProtocolEvaluation | None

    def __post_init__(self) -> None:
        if tuple(result.direction.name for result in self.directions) != tuple(
            direction.name for direction in self.definition.directions
        ):
            raise ValueError("Protocol results must preserve declared direction order")
        if self.definition.protocol == "cross-subject" and self.combined is not None:
            raise ValueError("Cross-subject protocol must not expose a combined result")
        if self.definition.protocol == "within-subject" and self.combined is None:
            raise ValueError("Bidirectional within-subject protocol requires a combined result")


def run_evaluation_protocol(
    protocol: EvaluationProtocol,
    *,
    config: LogisticRegressionExperimentConfig,
    dataset: FeatureSetDataset | None = None,
    targets: PixelTargetDataset | None = None,
) -> ProtocolEvaluationResult:
    resolved_dataset, resolved_targets = _resolve_inputs(
        config,
        dataset=dataset,
        targets=targets,
    )
    definition = build_evaluation_protocol(
        resolved_targets,
        protocol=protocol,
        split_config=config.split,
    )
    direction_results = tuple(
        _run_direction(
            resolved_dataset,
            targets=resolved_targets,
            direction=direction,
            audit=audit,
            config=config,
        )
        for direction, audit in zip(
            definition.directions,
            definition.audits,
            strict=True,
        )
    )
    combined = (
        _combine_within_subject_results(
            direction_results,
            targets=resolved_targets,
            config=config,
        )
        if protocol == "within-subject"
        else None
    )
    return ProtocolEvaluationResult(
        definition=definition,
        directions=direction_results,
        combined=combined,
    )


def _resolve_inputs(
    config: LogisticRegressionExperimentConfig,
    *,
    dataset: FeatureSetDataset | None,
    targets: PixelTargetDataset | None,
) -> tuple[FeatureSetDataset, PixelTargetDataset]:
    if (dataset is None) != (targets is None):
        raise ValueError("Pass both `dataset` and `targets`, or let the runner build both")
    if dataset is not None and targets is not None:
        return dataset, targets

    from utils.datasets import FeatureDataset

    configured_dataset = FeatureDataset(
        config.dataset.dataset_dir,
        dataset_step_type=config.dataset.recording_family,
        dataset_pattern_type=config.dataset.pattern_type,
        config_path=config.dataset.feature_config_path,
        cache_policy="disk",
        source_cache_policy="disk",
    )
    configured_targets = build_random_imagery_targets(
        configured_dataset.samples,
        image_rows=config.dataset.image_rows,
        image_columns=config.dataset.image_columns,
        allowed_sample_types=config.dataset.target_sample_types,
    )
    return configured_dataset, configured_targets


def _run_direction(
    dataset: FeatureSetDataset,
    *,
    targets: PixelTargetDataset,
    direction: EvaluationDirection,
    audit: ProtocolLeakageAudit,
    config: LogisticRegressionExperimentConfig,
) -> DirectionEvaluationResult:
    screening_features = build_aligned_training_features(
        dataset,
        targets=targets,
        split=direction,
        candidates=config.feature_screening.candidates,
    )
    y_train = targets.y[screening_features.target_row_indices]
    cross_validation = build_grouped_pixel_cross_validation(
        y_train,
        groups=screening_features.subject_ids,
        config=config.cross_validation,
    )
    screening = screen_feature_families(
        screening_features,
        y=y_train,
        cross_validation=cross_validation,
        config=config.feature_screening,
        random_state=config.random_state,
    )
    training_features = _select_training_partition(
        screening_features,
        block_names=screening.selected_block_names,
    )
    fitted_models = fit_pixel_models(
        training_features,
        y_train=y_train,
        pixel_names=targets.pixel_names,
        cross_validation=cross_validation,
        config=config.grid_search,
        scoring=config.cross_validation.scoring,
        random_state=config.random_state,
    )

    test_features = build_aligned_feature_partition(
        dataset,
        targets=targets,
        row_indices=direction.test_indices,
        block_names=screening.selected_block_names,
    )
    y_test = targets.y[test_features.target_row_indices]
    grid_search = predict_pixel_models(
        fitted_models,
        test_features=test_features,
        y_test=y_test,
        threshold=config.prediction_threshold,
    )
    model_metrics = evaluate_prediction_matrix(
        y_test,
        grid_search.predictions,
        grid_search.probabilities,
    )
    model_bootstrap = bootstrap_subject_mean_balanced_accuracy(
        y_test,
        grid_search.predictions,
        targets.subject_ids[grid_search.test_target_indices],
        n_resamples=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    baseline_evaluations = tuple(
        BaselineEvaluation(
            prediction=prediction,
            metrics=evaluate_prediction_matrix(
                y_test,
                prediction.predictions,
                prediction.probabilities,
            ),
        )
        for prediction in build_non_eeg_baselines(
            y_train,
            n_test_samples=y_test.shape[0],
            threshold=config.prediction_threshold,
            random_state=config.random_state,
        )
    )
    return DirectionEvaluationResult(
        direction=direction,
        audit=audit,
        screening=screening,
        grid_search=grid_search,
        model_metrics=model_metrics,
        model_bootstrap=model_bootstrap,
        baselines=baseline_evaluations,
    )


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


def _combine_within_subject_results(
    results: tuple[DirectionEvaluationResult, ...],
    *,
    targets: PixelTargetDataset,
    config: LogisticRegressionExperimentConfig,
) -> CombinedProtocolEvaluation:
    if tuple(result.direction.name for result in results) != (
        "trial-1-to-trial-2",
        "trial-2-to-trial-1",
    ):
        raise ValueError("Within-subject combination requires both ordered directions")

    target_indices = np.concatenate(
        [result.grid_search.test_target_indices for result in results]
    ).astype(np.int64, copy=False)
    subject_ids = targets.subject_ids[target_indices].astype(np.int64, copy=False)
    y_true = targets.y[target_indices].astype(np.int8, copy=False)
    probabilities = np.concatenate(
        [result.grid_search.probabilities for result in results],
        axis=0,
    ).astype(np.float64, copy=False)
    predictions = np.concatenate(
        [result.grid_search.predictions for result in results],
        axis=0,
    ).astype(np.int8, copy=False)
    for array in (target_indices, subject_ids, y_true, probabilities, predictions):
        array.setflags(write=False)

    model_metrics = evaluate_prediction_matrix(y_true, predictions, probabilities)
    model_bootstrap = bootstrap_subject_mean_balanced_accuracy(
        y_true,
        predictions,
        subject_ids,
        n_resamples=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    baseline_evaluations: list[BaselineEvaluation] = []
    baseline_names = tuple(item.prediction.name for item in results[0].baselines)
    for baseline_index, name in enumerate(baseline_names):
        direction_baselines = tuple(
            result.baselines[baseline_index].prediction for result in results
        )
        if any(prediction.name != name for prediction in direction_baselines):
            raise ValueError("Within-subject baseline order differs between directions")
        baseline_prediction = BaselinePrediction(
            name=name,
            probabilities=np.concatenate(
                [prediction.probabilities for prediction in direction_baselines],
                axis=0,
            ),
            predictions=np.concatenate(
                [prediction.predictions for prediction in direction_baselines],
                axis=0,
            ),
        )
        baseline_evaluations.append(
            BaselineEvaluation(
                prediction=baseline_prediction,
                metrics=evaluate_prediction_matrix(
                    y_true,
                    baseline_prediction.predictions,
                    baseline_prediction.probabilities,
                ),
            )
        )

    return CombinedProtocolEvaluation(
        protocol="within-subject",
        direction_names=tuple(result.direction.name for result in results),
        target_indices=target_indices,
        subject_ids=subject_ids,
        y_true=y_true,
        probabilities=probabilities,
        predictions=predictions,
        model_metrics=model_metrics,
        model_bootstrap=model_bootstrap,
        baselines=tuple(baseline_evaluations),
    )
