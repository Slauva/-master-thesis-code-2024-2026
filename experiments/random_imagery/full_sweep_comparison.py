"""Full-imagery sweep comparison helpers.

The module intentionally loads only JSON metadata and saved NumPy arrays from immutable
run directories. It does not deserialize sklearn pipelines or Torch checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from experiments.logistic_regression.artifacts import load_evaluation_run
from experiments.random_imagery.artifacts import load_model_run
from experiments.random_imagery.matrix import (
    CLASSICAL_MATRIX_MODEL_IDS,
    FULL_IMAGERY_CLASSICAL_FAILURES_PATH,
    FULL_IMAGERY_CLASSICAL_SUMMARY_PATH,
    TABULAR_FEATURE_FAMILIES,
    feature_family_slug,
)
from experiments.random_imagery.metrics import evaluate_prediction_matrix
from experiments.random_imagery.registry import REFERENCE_MODEL_ID, get_model_spec
from experiments.random_imagery_torch.artifacts import load_torch_run
from experiments.random_imagery_torch.config import PRIMARY_TORCH_MODEL_IDS
from experiments.random_imagery_torch.matrix import (
    FULL_IMAGERY_TORCH_FAILURES_PATH,
    FULL_IMAGERY_TORCH_SUMMARY_PATH,
)

ComparisonProtocol = Literal["cross-subject", "within-subject"]
ModelFamily = Literal["classical", "torch"]

FULL_IMAGERY_COMPARISON_SUMMARY_PATH = Path(
    "artifacts/experiments/full-imagery/stage5_comparison_summary.json"
)
FULL_IMAGERY_FIGURE_DIR = Path("artifacts/experiments/full-imagery/stage5_figures")

_DIRECTION_ORDER: dict[ComparisonProtocol, tuple[str, ...]] = {
    "cross-subject": ("cross-subject",),
    "within-subject": ("trial-1-to-trial-2", "trial-2-to-trial-1"),
}
_PROTOCOLS: tuple[ComparisonProtocol, ...] = ("cross-subject", "within-subject")
_REFERENCE_FEATURE_SLUG = "lbp"


@dataclass(frozen=True, slots=True)
class CombinedRunArrays:
    model_family: ModelFamily
    model_id: str
    protocol: ComparisonProtocol
    feature_slug: str | None
    feature_family: tuple[str, ...]
    method: str | None
    architecture: str | None
    run_dirs: tuple[Path, ...]
    test_sample_keys: tuple[tuple[int, int, int], ...]
    targets: NDArray[np.int8]
    scores: NDArray[np.float64]
    predictions: NDArray[np.int8]
    subject_ids: NDArray[np.int64]
    baseline_scores: dict[str, NDArray[np.float64]]
    baseline_predictions: dict[str, NDArray[np.int8]]

    @property
    def run_key(self) -> str:
        condition = self.feature_slug if self.model_family == "classical" else self.method
        return f"{self.model_family}:{self.model_id}:{condition}:{self.protocol}"


def write_full_imagery_comparison_summary(
    *,
    output_path: Path = FULL_IMAGERY_COMPARISON_SUMMARY_PATH,
    n_resamples: int = 2_000,
    random_state: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    payload = build_full_imagery_comparison_summary(
        n_resamples=n_resamples,
        random_state=random_state,
        confidence_level=confidence_level,
    )
    validate_full_imagery_comparison_summary(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_full_imagery_comparison_summary(
    *,
    n_resamples: int = 2_000,
    random_state: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("`n_resamples` must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("`confidence_level` must be between zero and one")

    classical_summary = _load_json(FULL_IMAGERY_CLASSICAL_SUMMARY_PATH)
    torch_summary = _load_json(FULL_IMAGERY_TORCH_SUMMARY_PATH)
    classical_failures = _load_json(FULL_IMAGERY_CLASSICAL_FAILURES_PATH).get(
        "failures", []
    )
    torch_failures = _load_json(FULL_IMAGERY_TORCH_FAILURES_PATH).get("failures", [])

    runs = _load_completed_runs(classical_summary, torch_summary)
    protocols = {
        protocol: _summarize_protocol(
            protocol,
            runs=tuple(run for run in runs if run.protocol == protocol),
            n_resamples=n_resamples,
            random_state=random_state,
            confidence_level=confidence_level,
        )
        for protocol in _PROTOCOLS
    }
    learned_rows = [
        row
        for protocol_payload in protocols.values()
        for row in protocol_payload["learned_runs"]
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "classical_summary": FULL_IMAGERY_CLASSICAL_SUMMARY_PATH.as_posix(),
            "classical_failures": FULL_IMAGERY_CLASSICAL_FAILURES_PATH.as_posix(),
            "torch_summary": FULL_IMAGERY_TORCH_SUMMARY_PATH.as_posix(),
            "torch_failures": FULL_IMAGERY_TORCH_FAILURES_PATH.as_posix(),
        },
        "coverage": {
            "classical": _coverage_payload(classical_summary, planned_protocols=180),
            "torch": _coverage_payload(torch_summary, planned_protocols=24),
            "completed_learned_condition_count": len(learned_rows),
            "completed_direction_run_count": int(
                classical_summary["completed_direction_run_count"]
                + torch_summary["completed_direction_run_count"]
            ),
            "failed_protocol_run_count": int(
                classical_summary["failed_protocol_run_count"]
                + torch_summary["failed_protocol_run_count"]
            ),
        },
        "planned_conditions": {
            "classical_model_ids": list(CLASSICAL_MATRIX_MODEL_IDS),
            "classical_feature_families": [
                feature_family_slug(feature_family)
                for feature_family in TABULAR_FEATURE_FAMILIES
            ],
            "torch_model_ids": list(PRIMARY_TORCH_MODEL_IDS),
            "torch_methods": sorted(
                {model_id.split("-")[-2] for model_id in PRIMARY_TORCH_MODEL_IDS}
            ),
            "protocols": list(_PROTOCOLS),
        },
        "bootstrap": {
            "metric": "mean_balanced_accuracy",
            "unit": "subject-cluster",
            "n_resamples": n_resamples,
            "random_state": random_state,
            "confidence_level": confidence_level,
            "reference": f"{REFERENCE_MODEL_ID}/{_REFERENCE_FEATURE_SLUG}",
            "interval_scope": "exploratory pointwise intervals, not multiplicity-adjusted",
        },
        "protocol_summaries": protocols,
        "failures": {
            "classical": classical_failures,
            "torch": torch_failures,
        },
    }


def validate_full_imagery_comparison_summary(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("Unexpected comparison summary schema version")
    coverage = payload["coverage"]
    if coverage["classical"]["completed_protocol_run_count"] != 167:
        raise ValueError("Classical completed protocol count changed")
    if coverage["torch"]["completed_protocol_run_count"] != 24:
        raise ValueError("Torch completed protocol count changed")
    if coverage["failed_protocol_run_count"] != 13:
        raise ValueError("Expected the 13 explicit Stage 3 convergence failures")
    for protocol in _PROTOCOLS:
        summary = payload["protocol_summaries"][protocol]
        if summary["protocol"] != protocol:
            raise ValueError(f"Protocol summary label mismatch for {protocol}")
        if summary["reference_run_key"] != (
            f"classical:{REFERENCE_MODEL_ID}:{_REFERENCE_FEATURE_SLUG}:{protocol}"
        ):
            raise ValueError(f"Unexpected reference condition for {protocol}")
        if summary["paired_compatible_run_count"] != len(summary["learned_runs"]):
            raise ValueError(f"Not all {protocol} runs were paired-compatible")
        if len(summary["descriptive_leaders"]) < 5:
            raise ValueError(f"Expected at least five {protocol} leaders")


def _summarize_protocol(
    protocol: ComparisonProtocol,
    *,
    runs: tuple[CombinedRunArrays, ...],
    n_resamples: int,
    random_state: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not runs:
        raise ValueError(f"No completed runs found for {protocol}")
    reference = _select_reference(protocol, runs)
    bootstrap_rows, n_attempts = _build_subject_bootstrap_rows(
        reference.targets,
        reference.subject_ids,
        n_resamples=n_resamples,
        random_state=random_state + (0 if protocol == "cross-subject" else 10_000),
    )
    reference_bootstrap = _bootstrap_balanced_accuracy(reference, bootstrap_rows)
    alpha = (1.0 - confidence_level) / 2.0

    learned_rows = []
    compatible_count = 0
    for run in runs:
        _validate_compatible_arrays(reference, run)
        compatible_count += 1
        bootstrap = _bootstrap_balanced_accuracy(run, bootstrap_rows)
        learned_rows.append(
            _learned_run_payload(
                run,
                reference=reference,
                bootstrap=bootstrap,
                reference_bootstrap=reference_bootstrap,
                alpha=alpha,
            )
        )
    learned_rows.sort(key=lambda row: row["metrics"]["mean_balanced_accuracy"], reverse=True)

    baseline_rows = _baseline_payloads(
        reference,
        reference_bootstrap=reference_bootstrap,
        bootstrap_rows=bootstrap_rows,
        alpha=alpha,
    )
    return {
        "protocol": protocol,
        "n_test_rows": int(reference.targets.shape[0]),
        "n_subjects": int(np.unique(reference.subject_ids).size),
        "reference_run_key": reference.run_key,
        "paired_compatible_run_count": compatible_count,
        "bootstrap_attempts": n_attempts,
        "learned_runs": learned_rows,
        "descriptive_leaders": learned_rows[:10],
        "baseline_rows": baseline_rows,
        "top_learned": learned_rows[0],
        "top_baseline": max(
            baseline_rows,
            key=lambda row: row["metrics"]["mean_balanced_accuracy"],
        ),
        "interpretation": _interpret_protocol(learned_rows[0], protocol),
    }


def _learned_run_payload(
    run: CombinedRunArrays,
    *,
    reference: CombinedRunArrays,
    bootstrap: NDArray[np.float64],
    reference_bootstrap: NDArray[np.float64],
    alpha: float,
) -> dict[str, Any]:
    metrics = evaluate_prediction_matrix(run.targets, run.predictions, run.scores)
    reference_metrics = evaluate_prediction_matrix(
        reference.targets,
        reference.predictions,
        reference.scores,
    )
    lower, upper = np.quantile(bootstrap, (alpha, 1.0 - alpha), method="linear")
    delta_samples = bootstrap - reference_bootstrap
    delta_lower, delta_upper = np.quantile(
        delta_samples,
        (alpha, 1.0 - alpha),
        method="linear",
    )
    return {
        "run_key": run.run_key,
        "model_family": run.model_family,
        "model_id": run.model_id,
        "model_label": _model_label(run),
        "protocol": run.protocol,
        "feature_slug": run.feature_slug,
        "feature_family": list(run.feature_family),
        "method": run.method,
        "architecture": run.architecture,
        "run_dirs": [path.as_posix() for path in run.run_dirs],
        "metrics": _metrics_payload(metrics),
        "mean_balanced_accuracy_ci": {
            "lower": float(lower),
            "upper": float(upper),
        },
        "delta_vs_reference": {
            "reference_run_key": reference.run_key,
            "mean_balanced_accuracy": float(
                metrics.mean_balanced_accuracy
                - reference_metrics.mean_balanced_accuracy
            ),
            "ci_lower": float(delta_lower),
            "ci_upper": float(delta_upper),
        },
    }


def _baseline_payloads(
    reference: CombinedRunArrays,
    *,
    reference_bootstrap: NDArray[np.float64],
    bootstrap_rows: tuple[NDArray[np.int64], ...],
    alpha: float,
) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(reference.baseline_scores):
        scores = reference.baseline_scores[name]
        predictions = reference.baseline_predictions[name]
        metrics = evaluate_prediction_matrix(reference.targets, predictions, scores)
        samples = _bootstrap_balanced_accuracy_from_arrays(
            reference.targets,
            predictions,
            bootstrap_rows,
        )
        lower, upper = np.quantile(samples, (alpha, 1.0 - alpha), method="linear")
        delta_samples = samples - reference_bootstrap
        delta_lower, delta_upper = np.quantile(
            delta_samples,
            (alpha, 1.0 - alpha),
            method="linear",
        )
        rows.append(
            {
                "baseline_name": name,
                "reference_run_key": reference.run_key,
                "metrics": _metrics_payload(metrics),
                "mean_balanced_accuracy_ci": {
                    "lower": float(lower),
                    "upper": float(upper),
                },
                "delta_vs_reference": {
                    "mean_balanced_accuracy": float(
                        metrics.mean_balanced_accuracy
                        - evaluate_prediction_matrix(
                            reference.targets,
                            reference.predictions,
                            reference.scores,
                        ).mean_balanced_accuracy
                    ),
                    "ci_lower": float(delta_lower),
                    "ci_upper": float(delta_upper),
                },
            }
        )
    return rows


def _load_completed_runs(
    classical_summary: dict[str, Any],
    torch_summary: dict[str, Any],
) -> tuple[CombinedRunArrays, ...]:
    runs: list[CombinedRunArrays] = []
    for item in classical_summary["results"]:
        if item["status"] != "completed":
            continue
        runs.append(_load_classical_condition(item))
    for item in torch_summary["results"]:
        if item["status"] != "completed":
            continue
        runs.append(_load_torch_condition(item))
    return tuple(runs)


def _load_classical_condition(item: dict[str, Any]) -> CombinedRunArrays:
    model_id = str(item["model_id"])
    protocol = _parse_protocol(item["protocol"])
    loaders = [
        load_evaluation_run(Path(path))
        if item["runner"] == "logistic-regression"
        else load_model_run(Path(path))
        for path in item["run_dirs"]
    ]
    ordered = _order_loaded_runs(protocol, loaders)
    scores = (
        tuple(run.probabilities for run in ordered)
        if item["runner"] == "logistic-regression"
        else tuple(run.scores for run in ordered)
    )
    baseline_scores_name = (
        "baseline_probabilities"
        if item["runner"] == "logistic-regression"
        else "baseline_scores"
    )
    return _combine_loaded_runs(
        model_family="classical",
        model_id=model_id,
        protocol=protocol,
        feature_slug=str(item["feature_slug"]),
        feature_family=tuple(str(part) for part in item["feature_family"]),
        method=None,
        architecture=None,
        ordered=ordered,
        scores=scores,
        baseline_scores_name=baseline_scores_name,
    )


def _load_torch_condition(item: dict[str, Any]) -> CombinedRunArrays:
    protocol = _parse_protocol(item["protocol"])
    ordered = _order_loaded_runs(
        protocol,
        [load_torch_run(Path(path)) for path in item["run_dirs"]],
    )
    return _combine_loaded_runs(
        model_family="torch",
        model_id=str(item["model_id"]),
        protocol=protocol,
        feature_slug=None,
        feature_family=(),
        method=str(item["method"]),
        architecture=str(item["architecture"]),
        ordered=ordered,
        scores=tuple(run.scores for run in ordered),
        baseline_scores_name="baseline_scores",
    )


def _combine_loaded_runs(
    *,
    model_family: ModelFamily,
    model_id: str,
    protocol: ComparisonProtocol,
    feature_slug: str | None,
    feature_family: tuple[str, ...],
    method: str | None,
    architecture: str | None,
    ordered: tuple[Any, ...],
    scores: tuple[NDArray[np.float64], ...],
    baseline_scores_name: str,
) -> CombinedRunArrays:
    keys: list[tuple[int, int, int]] = []
    baseline_scores: dict[str, list[NDArray[np.float64]]] = {}
    baseline_predictions: dict[str, list[NDArray[np.int8]]] = {}
    for run in ordered:
        run_keys = tuple(tuple(int(value) for value in key) for key in run.split["test_sample_keys"])
        if set(keys) & set(run_keys):
            raise ValueError(f"{model_id} {protocol} directions overlap test keys")
        keys.extend(run_keys)
        run_baseline_scores = getattr(run, baseline_scores_name)
        for name, values in run_baseline_scores.items():
            baseline_scores.setdefault(name, []).append(values)
        for name, values in run.baseline_predictions.items():
            baseline_predictions.setdefault(name, []).append(values)

    return CombinedRunArrays(
        model_family=model_family,
        model_id=model_id,
        protocol=protocol,
        feature_slug=feature_slug,
        feature_family=feature_family,
        method=method,
        architecture=architecture,
        run_dirs=tuple(run.run_dir for run in ordered),
        test_sample_keys=tuple(keys),
        targets=_readonly(np.concatenate([run.test_targets for run in ordered], axis=0), np.int8),
        scores=_readonly(np.concatenate(scores, axis=0), np.float64),
        predictions=_readonly(np.concatenate([run.predictions for run in ordered], axis=0), np.int8),
        subject_ids=_readonly(np.concatenate([run.test_subject_ids for run in ordered]), np.int64),
        baseline_scores={
            name: _readonly(np.concatenate(parts, axis=0), np.float64)
            for name, parts in baseline_scores.items()
        },
        baseline_predictions={
            name: _readonly(np.concatenate(parts, axis=0), np.int8)
            for name, parts in baseline_predictions.items()
        },
    )


def _order_loaded_runs(protocol: ComparisonProtocol, runs: list[Any]) -> tuple[Any, ...]:
    expected = _DIRECTION_ORDER[protocol]
    by_direction = {str(run.evaluation["direction"]["name"]): run for run in runs}
    if set(by_direction) != set(expected) or len(by_direction) != len(runs):
        raise ValueError(
            f"{protocol} comparison requires directions {expected}, "
            f"received {tuple(by_direction)}"
        )
    ordered = tuple(by_direction[direction] for direction in expected)
    if any(run.evaluation["protocol"] != protocol for run in ordered):
        raise ValueError("Run protocol differs from requested comparison protocol")
    return ordered


def _select_reference(
    protocol: ComparisonProtocol,
    runs: tuple[CombinedRunArrays, ...],
) -> CombinedRunArrays:
    matches = tuple(
        run
        for run in runs
        if run.model_family == "classical"
        and run.model_id == REFERENCE_MODEL_ID
        and run.feature_slug == _REFERENCE_FEATURE_SLUG
        and run.protocol == protocol
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {REFERENCE_MODEL_ID}/{_REFERENCE_FEATURE_SLUG} "
            f"reference for {protocol}, found {len(matches)}"
        )
    return matches[0]


def _validate_compatible_arrays(
    reference: CombinedRunArrays,
    run: CombinedRunArrays,
) -> None:
    if run.test_sample_keys != reference.test_sample_keys:
        raise ValueError(f"{run.run_key} uses different ordered test sample keys")
    if not np.array_equal(run.targets, reference.targets):
        raise ValueError(f"{run.run_key} uses different targets")
    if not np.array_equal(run.subject_ids, reference.subject_ids):
        raise ValueError(f"{run.run_key} uses different subject IDs")


def _build_subject_bootstrap_rows(
    targets: NDArray[np.int8],
    subject_ids: NDArray[np.int64],
    *,
    n_resamples: int,
    random_state: int,
) -> tuple[tuple[NDArray[np.int64], ...], int]:
    unique_subjects = np.unique(subject_ids)
    if unique_subjects.size < 2:
        raise ValueError("Paired bootstrap requires at least two subjects")
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
        raise RuntimeError("Could not draw enough class-complete bootstrap samples")
    return tuple(rows), attempts


def _bootstrap_balanced_accuracy(
    run: CombinedRunArrays,
    bootstrap_rows: tuple[NDArray[np.int64], ...],
) -> NDArray[np.float64]:
    return _bootstrap_balanced_accuracy_from_arrays(
        run.targets,
        run.predictions,
        bootstrap_rows,
    )


def _bootstrap_balanced_accuracy_from_arrays(
    targets: NDArray[np.int8],
    predictions: NDArray[np.int8],
    bootstrap_rows: tuple[NDArray[np.int64], ...],
) -> NDArray[np.float64]:
    values = np.empty(len(bootstrap_rows), dtype=np.float64)
    for index, rows in enumerate(bootstrap_rows):
        values[index] = _mean_balanced_accuracy(targets[rows], predictions[rows])
    values.setflags(write=False)
    return values


def _mean_balanced_accuracy(
    targets: NDArray[np.int8],
    predictions: NDArray[np.int8],
) -> float:
    positives = targets == 1
    negatives = ~positives
    true_positive_rate = np.divide(
        np.count_nonzero((predictions == 1) & positives, axis=0),
        np.count_nonzero(positives, axis=0),
    )
    true_negative_rate = np.divide(
        np.count_nonzero((predictions == 0) & negatives, axis=0),
        np.count_nonzero(negatives, axis=0),
    )
    return float(np.mean((true_positive_rate + true_negative_rate) / 2.0))


def _coverage_payload(payload: dict[str, Any], *, planned_protocols: int) -> dict[str, int]:
    return {
        "planned_protocol_run_count": planned_protocols,
        "completed_protocol_run_count": int(payload["completed_protocol_run_count"]),
        "completed_direction_run_count": int(payload["completed_direction_run_count"]),
        "failed_protocol_run_count": int(payload["failed_protocol_run_count"]),
    }


def _metrics_payload(metrics: Any) -> dict[str, float]:
    return {
        "mean_balanced_accuracy": float(metrics.mean_balanced_accuracy),
        "mean_macro_f1": float(metrics.mean_macro_f1),
        "mean_score_mse": float(metrics.mean_score_mse),
        "mean_sample_iou": float(metrics.mean_sample_iou),
        "bit_accuracy": float(metrics.bit_accuracy),
        "exact_match_accuracy": float(metrics.exact_match_accuracy),
        "hamming_loss": float(metrics.hamming_loss),
    }


def _model_label(run: CombinedRunArrays) -> str:
    if run.model_family == "torch":
        return f"{run.architecture} / {run.method}"
    return get_model_spec(run.model_id).label


def _interpret_protocol(top_row: dict[str, Any], protocol: ComparisonProtocol) -> str:
    metric = top_row["metrics"]["mean_balanced_accuracy"]
    delta = top_row["delta_vs_reference"]
    return (
        f"Best descriptive {protocol} condition is {top_row['run_key']} "
        f"with mean balanced accuracy {metric:.6f}; pointwise paired delta vs "
        f"{delta['reference_run_key']} is {delta['mean_balanced_accuracy']:.6f} "
        f"[{delta['ci_lower']:.6f}, {delta['ci_upper']:.6f}]."
    )


def _parse_protocol(value: str) -> ComparisonProtocol:
    if value not in _PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {value!r}")
    return value  # type: ignore[return-value]


def _readonly(array: NDArray[Any], dtype: type[np.generic]) -> NDArray[Any]:
    converted = np.asarray(array, dtype=dtype)
    converted.setflags(write=False)
    return converted


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
