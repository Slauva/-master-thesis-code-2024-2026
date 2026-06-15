from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.random_imagery.config import RandomImageryExperimentConfigLike
from experiments.random_imagery.contracts import (
    AnyFittedDirectionModel,
    AnyRandomImageryModelBackend,
    ModelPrediction,
    ScoreDiagnostics,
)
from experiments.random_imagery.shared import (
    BaselinePrediction,
    EvaluationDirection,
    EvaluationProtocol,
    EvaluationProtocolDefinition,
    FeatureSetDataset,
    PixelTargetDataset,
    PredictionMetrics,
    ProtocolLeakageAudit,
    SubjectBootstrapInterval,
    bootstrap_subject_mean_balanced_accuracy,
    build_aligned_feature_partition,
    build_aligned_training_features,
    build_evaluation_protocol,
    build_non_eeg_baselines,
    build_random_imagery_targets,
    evaluate_prediction_matrix,
)


@dataclass(frozen=True, slots=True)
class BaselineEvaluation:
    prediction: BaselinePrediction
    metrics: PredictionMetrics


@dataclass(frozen=True, slots=True)
class DirectionEvaluationResult:
    direction: EvaluationDirection
    audit: ProtocolLeakageAudit
    fitted_model: AnyFittedDirectionModel
    prediction: ModelPrediction
    model_metrics: PredictionMetrics
    model_bootstrap: SubjectBootstrapInterval
    baselines: tuple[BaselineEvaluation, ...]

    def __post_init__(self) -> None:
        if self.audit.direction_name != self.direction.name:
            raise ValueError("Direction result audit does not match its direction")
        if self.audit.has_forbidden_leakage:
            raise ValueError("Direction result contains forbidden leakage")
        if self.prediction.test_target_indices.shape != self.direction.test_indices.shape:
            raise ValueError("Direction prediction rows do not match its split")
        if not np.array_equal(
            self.prediction.test_target_indices,
            self.direction.test_indices,
        ):
            raise ValueError("Direction prediction rows do not preserve split order")
        if self.fitted_model.spec.score_semantics != self.prediction.diagnostics.score_semantics:
            raise ValueError("Prediction score semantics do not match the fitted model")
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
    prediction: ModelPrediction
    model_metrics: PredictionMetrics
    model_bootstrap: SubjectBootstrapInterval
    baselines: tuple[BaselineEvaluation, ...]

    def __post_init__(self) -> None:
        if self.protocol != "within-subject":
            raise ValueError("Only bidirectional within-subject results are combined")
        n_rows, n_targets = self.y_true.shape
        if self.target_indices.shape != (n_rows,) or self.target_indices.dtype != np.dtype(np.int64):
            raise TypeError("Combined target indices must be an int64 vector")
        if self.subject_ids.shape != (n_rows,) or self.subject_ids.dtype != np.dtype(np.int64):
            raise TypeError("Combined subject IDs must be an int64 vector")
        if self.y_true.dtype != np.dtype(np.int8) or not np.isin(self.y_true, (0, 1)).all():
            raise TypeError("Combined targets must be a binary int8 matrix")
        if self.prediction.scores.shape != (n_rows, n_targets):
            raise ValueError("Combined model scores must match combined targets")
        if not np.array_equal(self.prediction.test_target_indices, self.target_indices):
            raise ValueError("Combined prediction rows must match combined target indices")
        if len(set(self.target_indices.tolist())) != n_rows:
            raise ValueError("Combined directions must evaluate every target row at most once")


@dataclass(frozen=True, slots=True)
class ProtocolEvaluationResult:
    definition: EvaluationProtocolDefinition
    model_id: str
    directions: tuple[DirectionEvaluationResult, ...]
    combined: CombinedProtocolEvaluation | None

    def __post_init__(self) -> None:
        if tuple(result.direction.name for result in self.directions) != tuple(
            direction.name for direction in self.definition.directions
        ):
            raise ValueError("Protocol results must preserve declared direction order")
        if any(result.fitted_model.spec.model_id != self.model_id for result in self.directions):
            raise ValueError("Every direction must use the declared model")
        if self.definition.protocol == "cross-subject" and self.combined is not None:
            raise ValueError("Cross-subject protocol must not expose a combined result")
        if self.definition.protocol == "within-subject" and self.combined is None:
            raise ValueError("Bidirectional within-subject protocol requires a combined result")


def run_model_evaluation_protocol(
    protocol: EvaluationProtocol,
    *,
    config: RandomImageryExperimentConfigLike,
    backend: AnyRandomImageryModelBackend,
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
            backend=backend,
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
        model_id=backend.spec.model_id,
        directions=direction_results,
        combined=combined,
    )


def _resolve_inputs(
    config: RandomImageryExperimentConfigLike,
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
    )
    return configured_dataset, configured_targets


def _run_direction(
    dataset: FeatureSetDataset,
    *,
    targets: PixelTargetDataset,
    direction: EvaluationDirection,
    audit: ProtocolLeakageAudit,
    config: RandomImageryExperimentConfigLike,
    backend: AnyRandomImageryModelBackend,
) -> DirectionEvaluationResult:
    training_features = build_aligned_training_features(
        dataset,
        targets=targets,
        split=direction,
        candidates=config.feature_screening.candidates,
    )
    fitted_model = backend.fit(
        training_features,
        targets=targets,
        config=config,
    )
    if not np.array_equal(
        fitted_model.training_target_indices,
        direction.train_indices,
    ):
        raise ValueError("Fitted model training rows do not preserve the direction split")

    # Test features are deliberately materialized only after the backend has completed fitting.
    test_features = build_aligned_feature_partition(
        dataset,
        targets=targets,
        row_indices=direction.test_indices,
        block_names=fitted_model.selected_block_names,
    )
    prediction = backend.predict(
        fitted_model,
        test_features=test_features,
        targets=targets,
        config=config,
    )
    y_train = targets.y[fitted_model.training_target_indices]
    y_test = targets.y[prediction.test_target_indices]
    model_metrics = evaluate_prediction_matrix(
        y_test,
        prediction.predictions,
        prediction.scores,
    )
    model_bootstrap = bootstrap_subject_mean_balanced_accuracy(
        y_test,
        prediction.predictions,
        targets.subject_ids[prediction.test_target_indices],
        n_resamples=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    baselines = tuple(
        BaselineEvaluation(
            prediction=baseline,
            metrics=evaluate_prediction_matrix(
                y_test,
                baseline.predictions,
                baseline.probabilities,
            ),
        )
        for baseline in build_non_eeg_baselines(
            y_train,
            n_test_samples=y_test.shape[0],
            threshold=config.prediction_threshold,
            random_state=config.random_state,
        )
    )
    return DirectionEvaluationResult(
        direction=direction,
        audit=audit,
        fitted_model=fitted_model,
        prediction=prediction,
        model_metrics=model_metrics,
        model_bootstrap=model_bootstrap,
        baselines=baselines,
    )


def _combine_within_subject_results(
    results: tuple[DirectionEvaluationResult, ...],
    *,
    targets: PixelTargetDataset,
    config: RandomImageryExperimentConfigLike,
) -> CombinedProtocolEvaluation:
    if tuple(result.direction.name for result in results) != (
        "trial-1-to-trial-2",
        "trial-2-to-trial-1",
    ):
        raise ValueError("Within-subject combination requires both ordered directions")
    if len({result.fitted_model.spec.model_id for result in results}) != 1:
        raise ValueError("Within-subject directions must use the same model")
    if len({result.prediction.diagnostics.score_semantics for result in results}) != 1:
        raise ValueError("Within-subject directions must use the same score semantics")

    target_indices = np.concatenate(
        [result.prediction.test_target_indices for result in results]
    ).astype(np.int64, copy=False)
    subject_ids = targets.subject_ids[target_indices].astype(np.int64, copy=False)
    y_true = targets.y[target_indices].astype(np.int8, copy=False)
    scores = np.concatenate(
        [result.prediction.scores for result in results],
        axis=0,
    ).astype(np.float64, copy=False)
    predictions = np.concatenate(
        [result.prediction.predictions for result in results],
        axis=0,
    ).astype(np.int8, copy=False)
    sample_keys = tuple(
        key
        for result in results
        for key in result.prediction.test_sample_keys
    )
    for array in (target_indices, subject_ids, y_true, scores, predictions):
        array.setflags(write=False)

    element_counts = np.asarray(
        [result.prediction.scores.size for result in results],
        dtype=np.float64,
    )
    diagnostics = ScoreDiagnostics(
        score_semantics=results[0].prediction.diagnostics.score_semantics,
        clipped_below_zero_fraction=float(
            np.average(
                [
                    result.prediction.diagnostics.clipped_below_zero_fraction
                    for result in results
                ],
                weights=element_counts,
            )
        ),
        clipped_above_one_fraction=float(
            np.average(
                [
                    result.prediction.diagnostics.clipped_above_one_fraction
                    for result in results
                ],
                weights=element_counts,
            )
        ),
    )
    prediction = ModelPrediction(
        test_target_indices=target_indices,
        test_sample_keys=sample_keys,
        scores=scores,
        predictions=predictions,
        threshold=config.prediction_threshold,
        diagnostics=diagnostics,
    )
    model_metrics = evaluate_prediction_matrix(y_true, predictions, scores)
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
        if any(baseline.name != name for baseline in direction_baselines):
            raise ValueError("Within-subject baseline order differs between directions")
        baseline_prediction = BaselinePrediction(
            name=name,
            probabilities=np.concatenate(
                [baseline.probabilities for baseline in direction_baselines],
                axis=0,
            ),
            predictions=np.concatenate(
                [baseline.predictions for baseline in direction_baselines],
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
        prediction=prediction,
        model_metrics=model_metrics,
        model_bootstrap=model_bootstrap,
        baselines=tuple(baseline_evaluations),
    )


__all__ = [
    "BaselineEvaluation",
    "CombinedProtocolEvaluation",
    "DirectionEvaluationResult",
    "ProtocolEvaluationResult",
    "run_model_evaluation_protocol",
]
