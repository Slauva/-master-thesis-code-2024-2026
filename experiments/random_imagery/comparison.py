from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from experiments.logistic_regression.artifacts import (
    LoadedEvaluationRun,
    load_evaluation_run,
)
from experiments.random_imagery.artifacts import LoadedModelRun, load_model_run
from experiments.random_imagery.metrics import (
    PredictionMetrics,
    evaluate_prediction_matrix,
)
from experiments.random_imagery.registry import REFERENCE_MODEL_ID, get_model_spec

ComparisonProtocol = Literal["cross-subject", "within-subject"]
ComparisonMetric = Literal[
    "mean_balanced_accuracy",
    "mean_score_mse",
    "mean_sample_iou",
    "hamming_loss",
]

_DIRECTION_ORDER = {
    "cross-subject": ("cross-subject",),
    "within-subject": ("trial-1-to-trial-2", "trial-2-to-trial-1"),
}
_COMPARISON_METRICS: tuple[ComparisonMetric, ...] = (
    "mean_balanced_accuracy",
    "mean_score_mse",
    "mean_sample_iou",
    "hamming_loss",
)
_HIGHER_IS_BETTER = {
    "mean_balanced_accuracy": True,
    "mean_score_mse": False,
    "mean_sample_iou": True,
    "hamming_loss": False,
}


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_score: float
    observed_frequency: float


@dataclass(frozen=True, slots=True)
class PairedMetricDifference:
    metric: ComparisonMetric
    higher_is_better: bool
    model_estimate: float
    reference_estimate: float
    difference: float
    improvement: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class ModelProtocolSummary:
    model_id: str
    label: str
    task: str
    topology: str
    score_semantics: str
    run_dirs: tuple[Path, ...]
    selected_feature_families: tuple[tuple[str, ...], ...]
    metrics: PredictionMetrics
    balanced_accuracy_lower: float
    balanced_accuracy_upper: float
    paired_differences: tuple[PairedMetricDifference, ...]
    calibration_ece: float | None
    calibration_bins: tuple[CalibrationBin, ...]
    calibration_coefficients: tuple[float, ...]
    clipping_below_zero_fraction: float | None
    clipping_above_one_fraction: float | None

    def paired(self, metric: ComparisonMetric) -> PairedMetricDifference:
        matches = tuple(item for item in self.paired_differences if item.metric == metric)
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate paired metric: {metric}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class BaselineProtocolSummary:
    name: str
    metrics: PredictionMetrics


@dataclass(frozen=True, slots=True)
class ProtocolComparison:
    protocol: ComparisonProtocol
    n_test_rows: int
    n_subjects: int
    n_resamples: int
    n_attempts: int
    reference_model_id: str
    models: tuple[ModelProtocolSummary, ...]
    baselines: tuple[BaselineProtocolSummary, ...]

    def model(self, model_id: str) -> ModelProtocolSummary:
        matches = tuple(item for item in self.models if item.model_id == model_id)
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate model summary: {model_id}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class _ComparisonArrays:
    model_id: str
    run_dirs: tuple[Path, ...]
    selected_feature_families: tuple[tuple[str, ...], ...]
    test_sample_keys: tuple[tuple[int, int, int], ...]
    targets: NDArray[np.int8]
    scores: NDArray[np.float64]
    predictions: NDArray[np.int8]
    subject_ids: NDArray[np.int64]
    calibration_coefficients: tuple[float, ...]
    clipping_below_zero_fraction: float | None
    clipping_above_one_fraction: float | None
    baseline_scores: dict[str, NDArray[np.float64]] | None = None
    baseline_predictions: dict[str, NDArray[np.int8]] | None = None


def compare_protocol_models(
    protocol: ComparisonProtocol,
    *,
    reference_run_dirs: Sequence[Path],
    model_run_dirs: Mapping[str, Sequence[Path]],
    n_resamples: int = 2_000,
    random_state: int = 42,
    confidence_level: float = 0.95,
    calibration_bins: int = 10,
) -> ProtocolComparison:
    if protocol not in _DIRECTION_ORDER:
        raise ValueError(f"Unsupported comparison protocol: {protocol!r}")
    if not model_run_dirs:
        raise ValueError("At least one schema-v3 model is required")
    if isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("`n_resamples` must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("`confidence_level` must be between zero and one")

    reference = _load_reference_arrays(protocol, reference_run_dirs)
    model_arrays = tuple(
        _load_model_arrays(protocol, model_id, run_dirs)
        for model_id, run_dirs in model_run_dirs.items()
    )
    for arrays in model_arrays:
        _validate_compatible_arrays(reference, arrays)

    bootstrap_rows, n_attempts = _build_subject_bootstrap_rows(
        reference.targets,
        reference.subject_ids,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    reference_metrics = evaluate_prediction_matrix(
        reference.targets,
        reference.predictions,
        reference.scores,
    )
    reference_bootstrap = _bootstrap_metric_matrix(reference, bootstrap_rows)
    alpha = (1.0 - confidence_level) / 2.0

    summaries = [
        _build_model_summary(
            arrays=reference,
            reference=reference,
            metrics=reference_metrics,
            bootstrap_metrics=reference_bootstrap,
            reference_bootstrap=reference_bootstrap,
            alpha=alpha,
            calibration_bins=calibration_bins,
        )
    ]
    for arrays in model_arrays:
        metrics = evaluate_prediction_matrix(
            arrays.targets,
            arrays.predictions,
            arrays.scores,
        )
        bootstrap_metrics = _bootstrap_metric_matrix(arrays, bootstrap_rows)
        summaries.append(
            _build_model_summary(
                arrays=arrays,
                reference=reference,
                metrics=metrics,
                bootstrap_metrics=bootstrap_metrics,
                reference_bootstrap=reference_bootstrap,
                alpha=alpha,
                calibration_bins=calibration_bins,
            )
        )

    baselines = _build_baseline_summaries(reference)
    return ProtocolComparison(
        protocol=protocol,
        n_test_rows=reference.targets.shape[0],
        n_subjects=int(np.unique(reference.subject_ids).size),
        n_resamples=n_resamples,
        n_attempts=n_attempts,
        reference_model_id=REFERENCE_MODEL_ID,
        models=tuple(summaries),
        baselines=baselines,
    )


def _load_reference_arrays(
    protocol: ComparisonProtocol,
    run_dirs: Sequence[Path],
) -> _ComparisonArrays:
    runs = tuple(load_evaluation_run(Path(path)) for path in run_dirs)
    ordered = _order_runs(protocol, runs)
    return _combine_runs(
        model_id=REFERENCE_MODEL_ID,
        ordered=ordered,
        score_attribute="probabilities",
        baseline_score_attribute="baseline_probabilities",
    )


def _load_model_arrays(
    protocol: ComparisonProtocol,
    model_id: str,
    run_dirs: Sequence[Path],
) -> _ComparisonArrays:
    spec = get_model_spec(model_id)
    if spec.reference:
        raise ValueError("Schema-v3 model mapping must not contain the reference model")
    runs = tuple(load_model_run(Path(path)) for path in run_dirs)
    if any(run.config.model_id != model_id for run in runs):
        raise ValueError(f"Run model ID differs from mapping key: {model_id}")
    ordered = _order_runs(protocol, runs)
    return _combine_runs(
        model_id=model_id,
        ordered=ordered,
        score_attribute="scores",
        baseline_score_attribute=None,
    )


def _order_runs(
    protocol: ComparisonProtocol,
    runs: Sequence[LoadedEvaluationRun | LoadedModelRun],
) -> tuple[LoadedEvaluationRun | LoadedModelRun, ...]:
    expected = _DIRECTION_ORDER[protocol]
    by_direction = {
        str(run.evaluation["direction"]["name"]): run
        for run in runs
    }
    if len(by_direction) != len(runs) or set(by_direction) != set(expected):
        raise ValueError(
            f"{protocol} comparison requires directions {expected}, "
            f"received {tuple(by_direction)}"
        )
    ordered = tuple(by_direction[direction] for direction in expected)
    if any(run.evaluation["protocol"] != protocol for run in ordered):
        raise ValueError("Run protocol differs from requested comparison protocol")
    return ordered


def _combine_runs(
    *,
    model_id: str,
    ordered: Sequence[LoadedEvaluationRun | LoadedModelRun],
    score_attribute: str,
    baseline_score_attribute: str | None,
) -> _ComparisonArrays:
    keys: list[tuple[int, int, int]] = []
    targets = []
    scores = []
    predictions = []
    subject_ids = []
    selected_feature_families = []
    run_dirs = []
    calibration_coefficients: list[float] = []
    clipping_below = []
    clipping_above = []
    baseline_scores: dict[str, list[NDArray[np.float64]]] | None = None
    baseline_predictions: dict[str, list[NDArray[np.int8]]] | None = None

    for run in ordered:
        run_dirs.append(run.run_dir)
        run_keys = tuple(tuple(key) for key in run.split["test_sample_keys"])
        if set(keys) & set(run_keys):
            raise ValueError("Combined comparison directions overlap test sample keys")
        keys.extend(run_keys)
        targets.append(run.test_targets)
        scores.append(getattr(run, score_attribute))
        predictions.append(run.predictions)
        subject_ids.append(run.test_subject_ids)
        selected_feature_families.append(
            tuple(run.evaluation["selected_feature_family"])
        )

        diagnostics = run.results.get("score_diagnostics", {})
        if diagnostics.get("score_semantics") == "clipped_regression":
            weight = int(run.predictions.size)
            clipping_below.append(
                (float(diagnostics["clipped_below_zero_fraction"]), weight)
            )
            clipping_above.append(
                (float(diagnostics["clipped_above_one_fraction"]), weight)
            )
        for item in run.results.get("models", []):
            calibration = item.get("calibration")
            if isinstance(calibration, dict):
                calibration_coefficients.append(float(calibration["coefficient"]))

        if baseline_score_attribute is not None:
            run_baseline_scores = getattr(run, baseline_score_attribute)
            run_baseline_predictions = run.baseline_predictions
            if baseline_scores is None:
                baseline_scores = {name: [] for name in run_baseline_scores}
                baseline_predictions = {
                    name: [] for name in run_baseline_predictions
                }
            if (
                set(run_baseline_scores) != set(baseline_scores)
                or baseline_predictions is None
                or set(run_baseline_predictions) != set(baseline_predictions)
            ):
                raise ValueError("Reference baseline names differ across directions")
            for name in baseline_scores:
                baseline_scores[name].append(run_baseline_scores[name])
                baseline_predictions[name].append(run_baseline_predictions[name])

    combined_baseline_scores = (
        {
            name: _readonly(np.concatenate(parts, axis=0), np.float64)
            for name, parts in baseline_scores.items()
        }
        if baseline_scores is not None
        else None
    )
    combined_baseline_predictions = (
        {
            name: _readonly(np.concatenate(parts, axis=0), np.int8)
            for name, parts in baseline_predictions.items()
        }
        if baseline_predictions is not None
        else None
    )
    return _ComparisonArrays(
        model_id=model_id,
        run_dirs=tuple(run_dirs),
        selected_feature_families=tuple(selected_feature_families),
        test_sample_keys=tuple(keys),
        targets=_readonly(np.concatenate(targets, axis=0), np.int8),
        scores=_readonly(np.concatenate(scores, axis=0), np.float64),
        predictions=_readonly(np.concatenate(predictions, axis=0), np.int8),
        subject_ids=_readonly(np.concatenate(subject_ids), np.int64),
        calibration_coefficients=tuple(calibration_coefficients),
        clipping_below_zero_fraction=_weighted_fraction(clipping_below),
        clipping_above_one_fraction=_weighted_fraction(clipping_above),
        baseline_scores=combined_baseline_scores,
        baseline_predictions=combined_baseline_predictions,
    )


def _validate_compatible_arrays(
    reference: _ComparisonArrays,
    model: _ComparisonArrays,
) -> None:
    if model.test_sample_keys != reference.test_sample_keys:
        raise ValueError(
            f"Model {model.model_id} uses different ordered test sample keys"
        )
    if not np.array_equal(model.targets, reference.targets):
        raise ValueError(f"Model {model.model_id} uses different test targets")
    if not np.array_equal(model.subject_ids, reference.subject_ids):
        raise ValueError(f"Model {model.model_id} uses different test subject IDs")


def _build_subject_bootstrap_rows(
    targets: NDArray[np.int8],
    subject_ids: NDArray[np.int64],
    *,
    n_resamples: int,
    random_state: int,
) -> tuple[tuple[NDArray[np.int64], ...], int]:
    unique_subjects = np.unique(subject_ids)
    if unique_subjects.size < 2:
        raise ValueError("Paired subject bootstrap requires at least two subjects")
    subject_rows = {
        int(subject): np.flatnonzero(subject_ids == subject).astype(np.int64)
        for subject in unique_subjects
    }
    rng = np.random.default_rng(random_state)
    rows: list[NDArray[np.int64]] = []
    attempts = 0
    max_attempts = n_resamples * 100
    while len(rows) < n_resamples and attempts < max_attempts:
        attempts += 1
        drawn = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
        indices = np.concatenate([subject_rows[int(subject)] for subject in drawn])
        sampled_targets = targets[indices]
        positive_counts = sampled_targets.sum(axis=0, dtype=np.int64)
        if np.any(positive_counts == 0) or np.any(
            positive_counts == sampled_targets.shape[0]
        ):
            continue
        indices.setflags(write=False)
        rows.append(indices)
    if len(rows) != n_resamples:
        raise RuntimeError(
            "Could not draw enough class-complete paired subject bootstrap samples"
        )
    return tuple(rows), attempts


def _bootstrap_metric_matrix(
    arrays: _ComparisonArrays,
    bootstrap_rows: Sequence[NDArray[np.int64]],
) -> NDArray[np.float64]:
    values = np.empty((len(bootstrap_rows), len(_COMPARISON_METRICS)), dtype=np.float64)
    for index, rows in enumerate(bootstrap_rows):
        metrics = evaluate_prediction_matrix(
            arrays.targets[rows],
            arrays.predictions[rows],
            arrays.scores[rows],
        )
        values[index] = _metric_values(metrics)
    values.setflags(write=False)
    return values


def _build_model_summary(
    *,
    arrays: _ComparisonArrays,
    reference: _ComparisonArrays,
    metrics: PredictionMetrics,
    bootstrap_metrics: NDArray[np.float64],
    reference_bootstrap: NDArray[np.float64],
    alpha: float,
    calibration_bins: int,
) -> ModelProtocolSummary:
    spec = get_model_spec(arrays.model_id)
    paired = []
    metric_values = _metric_values(metrics)
    reference_metrics = evaluate_prediction_matrix(
        reference.targets,
        reference.predictions,
        reference.scores,
    )
    reference_values = _metric_values(reference_metrics)
    for metric_index, metric in enumerate(_COMPARISON_METRICS):
        difference_samples = (
            bootstrap_metrics[:, metric_index]
            - reference_bootstrap[:, metric_index]
        )
        sign = 1.0 if _HIGHER_IS_BETTER[metric] else -1.0
        improvement_samples = sign * difference_samples
        lower, upper = np.quantile(
            improvement_samples,
            (alpha, 1.0 - alpha),
            method="linear",
        )
        difference = float(metric_values[metric_index] - reference_values[metric_index])
        paired.append(
            PairedMetricDifference(
                metric=metric,
                higher_is_better=_HIGHER_IS_BETTER[metric],
                model_estimate=float(metric_values[metric_index]),
                reference_estimate=float(reference_values[metric_index]),
                difference=difference,
                improvement=sign * difference,
                lower=float(lower),
                upper=float(upper),
            )
        )

    balanced_accuracy_samples = bootstrap_metrics[:, 0]
    ba_lower, ba_upper = np.quantile(
        balanced_accuracy_samples,
        (alpha, 1.0 - alpha),
        method="linear",
    )
    bins: tuple[CalibrationBin, ...] = ()
    ece: float | None = None
    if spec.task == "classifier":
        bins, ece = _calibration_summary(
            arrays.targets,
            arrays.scores,
            n_bins=calibration_bins,
        )
    return ModelProtocolSummary(
        model_id=arrays.model_id,
        label=spec.label,
        task=spec.task,
        topology=spec.topology,
        score_semantics=spec.score_semantics,
        run_dirs=arrays.run_dirs,
        selected_feature_families=arrays.selected_feature_families,
        metrics=metrics,
        balanced_accuracy_lower=float(ba_lower),
        balanced_accuracy_upper=float(ba_upper),
        paired_differences=tuple(paired),
        calibration_ece=ece,
        calibration_bins=bins,
        calibration_coefficients=arrays.calibration_coefficients,
        clipping_below_zero_fraction=arrays.clipping_below_zero_fraction,
        clipping_above_one_fraction=arrays.clipping_above_one_fraction,
    )


def _build_baseline_summaries(
    reference: _ComparisonArrays,
) -> tuple[BaselineProtocolSummary, ...]:
    if reference.baseline_scores is None or reference.baseline_predictions is None:
        return ()
    return tuple(
        BaselineProtocolSummary(
            name=name,
            metrics=evaluate_prediction_matrix(
                reference.targets,
                reference.baseline_predictions[name],
                reference.baseline_scores[name],
            ),
        )
        for name in reference.baseline_scores
    )


def _calibration_summary(
    targets: NDArray[np.int8],
    scores: NDArray[np.float64],
    *,
    n_bins: int,
) -> tuple[tuple[CalibrationBin, ...], float]:
    if isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("Calibration requires at least two bins")
    flat_targets = targets.reshape(-1)
    flat_scores = scores.reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(
        np.searchsorted(edges, flat_scores, side="right") - 1,
        n_bins - 1,
    )
    assignments = np.maximum(assignments, 0)
    bins = []
    weighted_error = 0.0
    for index in range(n_bins):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_score = float(flat_scores[mask].mean())
        observed = float(flat_targets[mask].mean())
        weighted_error += count * abs(mean_score - observed)
        bins.append(
            CalibrationBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                mean_score=mean_score,
                observed_frequency=observed,
            )
        )
    return tuple(bins), float(weighted_error / flat_scores.size)


def _metric_values(metrics: PredictionMetrics) -> NDArray[np.float64]:
    return np.asarray(
        (
            metrics.mean_balanced_accuracy,
            metrics.mean_score_mse,
            metrics.mean_sample_iou,
            metrics.hamming_loss,
        ),
        dtype=np.float64,
    )


def _weighted_fraction(values: Sequence[tuple[float, int]]) -> float | None:
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    return float(sum(value * weight for value, weight in values) / total_weight)


def _readonly(
    array: NDArray[np.generic],
    dtype: type[np.generic],
) -> NDArray:
    result = np.asarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


__all__ = [
    "BaselineProtocolSummary",
    "CalibrationBin",
    "ComparisonMetric",
    "ComparisonProtocol",
    "ModelProtocolSummary",
    "PairedMetricDifference",
    "ProtocolComparison",
    "compare_protocol_models",
]
