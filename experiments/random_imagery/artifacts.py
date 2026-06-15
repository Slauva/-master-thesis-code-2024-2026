import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.pipeline import Pipeline

from experiments.logistic_regression.artifacts import load_evaluation_run
from experiments.random_imagery.classifier_backend import (
    FittedCalibratedPixelModels,
)
from experiments.random_imagery.config import (
    RandomImageryModelConfig,
    build_model_run_hash,
    validate_model_config_payload,
)
from experiments.random_imagery.contracts import ModelPrediction
from experiments.random_imagery.registry import get_model_spec
from experiments.random_imagery.regression_backend import (
    FittedIndependentRegressionModels,
    FittedMultiOutputRegressionModel,
)
from experiments.random_imagery.runner import (
    DirectionEvaluationResult,
    ProtocolEvaluationResult,
)
from experiments.random_imagery.shared import (
    AlignedFeaturePartition,
    PixelTargetDataset,
    PredictionMetrics,
    bootstrap_subject_mean_balanced_accuracy,
    evaluate_prediction_matrix,
)

SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class LoadedModelRun:
    run_dir: Path
    manifest: dict[str, Any]
    config: RandomImageryModelConfig
    environment: dict[str, Any]
    split: dict[str, Any]
    features: dict[str, Any]
    screening: dict[str, Any]
    results: dict[str, Any]
    evaluation: dict[str, Any]
    scores: NDArray[np.float64]
    predictions: NDArray[np.int8]
    test_balanced_accuracy: NDArray[np.float64]
    train_targets: NDArray[np.int8]
    test_targets: NDArray[np.int8]
    test_subject_ids: NDArray[np.int64]
    baseline_scores: dict[str, NDArray[np.float64]]
    baseline_predictions: dict[str, NDArray[np.int8]]
    pipelines: tuple[Pipeline, ...] = ()

    def __post_init__(self) -> None:
        _validate_loaded_arrays(
            scores=self.scores,
            predictions=self.predictions,
            test_balanced_accuracy=self.test_balanced_accuracy,
            train_targets=self.train_targets,
            test_targets=self.test_targets,
            test_subject_ids=self.test_subject_ids,
        )
        expected_pipeline_count = (
            int(self.results["pipeline_count"]) if self.pipelines else 0
        )
        if self.pipelines and len(self.pipelines) != expected_pipeline_count:
            raise ValueError("Loaded pipeline count does not match results metadata")
        if any(not isinstance(pipeline, Pipeline) for pipeline in self.pipelines):
            raise TypeError("Loaded models must be sklearn Pipelines")


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    run_dir: Path
    schema_version: int
    model_id: str
    protocol: str
    direction: str
    test_sample_keys: tuple[tuple[int, int, int], ...]
    evaluation: dict[str, Any]


def model_run_dir(
    config: RandomImageryModelConfig,
    *,
    protocol: str,
    direction: str,
) -> Path:
    return (
        Path(config.artifacts.root)
        / config.model_id
        / build_model_run_hash(
            config,
            protocol=protocol,
            direction=direction,
        )
    )


def write_model_protocol_runs(
    result: ProtocolEvaluationResult,
    *,
    targets: PixelTargetDataset,
    config: RandomImageryModelConfig,
    feature_config_hash: str,
) -> tuple[Path, ...]:
    if config.artifacts.schema_version != SCHEMA_VERSION:
        raise ValueError("Model-aware artifacts require schema version 3")
    if config.artifacts.overwrite:
        raise ValueError("Schema-v3 runs are immutable and cannot enable overwrite")
    if result.model_id != config.model_id:
        raise ValueError("Protocol result model does not match artifact configuration")
    return tuple(
        _write_direction_run(
            direction_result,
            protocol_label=result.definition.label,
            targets=targets,
            config=config,
            feature_config_hash=feature_config_hash,
        )
        for direction_result in result.directions
    )


def _write_direction_run(
    result: DirectionEvaluationResult,
    *,
    protocol_label: str,
    targets: PixelTargetDataset,
    config: RandomImageryModelConfig,
    feature_config_hash: str,
) -> Path:
    _validate_direction_result(result, targets=targets, config=config)
    run_dir = model_run_dir(
        config,
        protocol=result.direction.protocol,
        direction=result.direction.name,
    )
    root = run_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"Experiment run already exists and is immutable: {run_dir}")
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}-", dir=root))
    try:
        _write_direction_payload(
            temporary_dir,
            result=result,
            protocol_label=protocol_label,
            targets=targets,
            config=config,
            feature_config_hash=feature_config_hash,
            config_hash=run_dir.name,
        )
        _publish_run(temporary_dir, run_dir=run_dir)
        return run_dir
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def _write_direction_payload(
    run_dir: Path,
    *,
    result: DirectionEvaluationResult,
    protocol_label: str,
    targets: PixelTargetDataset,
    config: RandomImageryModelConfig,
    feature_config_hash: str,
    config_hash: str,
) -> None:
    arrays_dir = run_dir / "arrays"
    pipelines_dir = run_dir / "pipelines"
    arrays_dir.mkdir()
    pipelines_dir.mkdir()

    pipeline_files, model_details = _write_fitted_payload(
        pipelines_dir,
        result=result,
    )
    _write_json(run_dir / "config.json", config.model_dump(mode="json"))
    _write_json(run_dir / "environment.json", _build_environment_payload())
    _write_json(
        run_dir / "split.json",
        _build_split_payload(result, targets=targets),
    )
    _write_json(
        run_dir / "features.json",
        {
            "feature_config_hash": feature_config_hash,
            "block_names": list(result.fitted_model.selected_block_names),
            "feature_names": list(result.fitted_model.feature_names),
        },
    )
    _write_json(
        run_dir / "screening.json",
        _serialize_screening(result.fitted_model.selection),
    )
    _write_json(
        run_dir / "results.json",
        {
            "model_id": result.fitted_model.spec.model_id,
            "estimator_family": result.fitted_model.spec.estimator_family,
            "topology": result.fitted_model.spec.topology,
            "task": result.fitted_model.spec.task,
            "score_semantics": result.fitted_model.spec.score_semantics,
            "prediction_threshold": result.prediction.threshold,
            "score_diagnostics": _diagnostics_payload(result.prediction),
            "pipeline_count": len(pipeline_files),
            "pipeline_files": pipeline_files,
            "models": model_details,
        },
    )
    evaluation = _build_evaluation_payload(
        result,
        protocol_label=protocol_label,
    )
    _write_json(run_dir / "evaluation.json", evaluation)

    train_indices = result.fitted_model.training_target_indices
    test_indices = result.prediction.test_target_indices
    _write_array(arrays_dir / "scores.npy", result.prediction.scores)
    _write_array(arrays_dir / "predictions.npy", result.prediction.predictions)
    _write_array(
        arrays_dir / "test_balanced_accuracy.npy",
        result.model_metrics.per_pixel_balanced_accuracy,
    )
    _write_array(arrays_dir / "train_targets.npy", targets.y[train_indices])
    _write_array(arrays_dir / "test_targets.npy", targets.y[test_indices])
    _write_array(arrays_dir / "test_subject_ids.npy", targets.subject_ids[test_indices])
    for baseline in result.baselines:
        name = baseline.prediction.name
        _write_array(
            arrays_dir / f"baseline_{name}_scores.npy",
            baseline.prediction.probabilities,
        )
        _write_array(
            arrays_dir / f"baseline_{name}_predictions.npy",
            baseline.prediction.predictions,
        )

    files = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "config_hash": config_hash,
            "model_id": config.model_id,
            "topology": result.fitted_model.spec.topology,
            "protocol": result.direction.protocol,
            "direction": result.direction.name,
            "generated_at": datetime.now(UTC).isoformat(),
            "writer": "experiments.random_imagery.artifacts.write_model_protocol_runs",
            "file_count": len(files),
            "files": files,
        },
    )
    _fsync_directory(run_dir)


def load_model_run(
    run_dir: Path,
    *,
    trusted: bool = False,
) -> LoadedModelRun:
    resolved_dir = Path(run_dir)
    manifest_path = resolved_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    _validate_manifest(resolved_dir, manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Schema-v3 model reader received a different artifact schema")

    config = validate_model_config_payload(_load_json(resolved_dir / "config.json"))
    evaluation = _load_json(resolved_dir / "evaluation.json")
    direction = evaluation.get("direction")
    if not isinstance(direction, dict):
        raise ValueError("Evaluation direction metadata is invalid")
    expected_hash = build_model_run_hash(
        config,
        protocol=str(evaluation.get("protocol")),
        direction=str(direction.get("name")),
    )
    if manifest.get("config_hash") != expected_hash or resolved_dir.name != expected_hash:
        raise ValueError("Experiment config hash does not match manifest and run directory")
    if manifest.get("model_id") != config.model_id:
        raise ValueError("Experiment manifest model does not match configuration")

    environment = _load_json(resolved_dir / "environment.json")
    split = _load_json(resolved_dir / "split.json")
    features = _load_json(resolved_dir / "features.json")
    screening = _load_json(resolved_dir / "screening.json")
    results = _load_json(resolved_dir / "results.json")
    arrays_dir = resolved_dir / "arrays"
    scores = _load_array(arrays_dir / "scores.npy", np.float64)
    predictions = _load_array(arrays_dir / "predictions.npy", np.int8)
    test_balanced_accuracy = _load_array(
        arrays_dir / "test_balanced_accuracy.npy",
        np.float64,
    )
    train_targets = _load_array(arrays_dir / "train_targets.npy", np.int8)
    test_targets = _load_array(arrays_dir / "test_targets.npy", np.int8)
    test_subject_ids = _load_array(arrays_dir / "test_subject_ids.npy", np.int64)
    baseline_scores, baseline_predictions = _load_baselines(
        resolved_dir,
        evaluation=evaluation,
    )
    _validate_evaluation_consistency(
        evaluation,
        split=split,
        features=features,
        results=results,
        scores=scores,
        predictions=predictions,
        test_balanced_accuracy=test_balanced_accuracy,
        test_targets=test_targets,
        test_subject_ids=test_subject_ids,
        baseline_scores=baseline_scores,
        baseline_predictions=baseline_predictions,
        config=config,
    )
    pipeline_files = results.get("pipeline_files")
    pipeline_count = results.get("pipeline_count")
    if (
        not isinstance(pipeline_files, list)
        or not pipeline_files
        or not isinstance(pipeline_count, int)
        or pipeline_count != len(pipeline_files)
    ):
        raise ValueError("Experiment results contain invalid pipeline metadata")
    pipeline_paths = tuple(
        _resolve_pipeline_path(
            resolved_dir,
            relative_path,
            manifest_files=set(manifest["files"]),
        )
        for relative_path in pipeline_files
    )
    pipelines: tuple[Pipeline, ...] = ()
    if trusted:
        pipelines = tuple(
            _load_pipeline(pipeline_path) for pipeline_path in pipeline_paths
        )
    return LoadedModelRun(
        run_dir=resolved_dir,
        manifest=manifest,
        config=config,
        environment=environment,
        split=split,
        features=features,
        screening=screening,
        results=results,
        evaluation=evaluation,
        scores=scores,
        predictions=predictions,
        test_balanced_accuracy=test_balanced_accuracy,
        train_targets=train_targets,
        test_targets=test_targets,
        test_subject_ids=test_subject_ids,
        baseline_scores=baseline_scores,
        baseline_predictions=baseline_predictions,
        pipelines=pipelines,
    )


def replay_model_predictions(
    run: LoadedModelRun,
    *,
    test_features: AlignedFeaturePartition,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    if not run.pipelines:
        raise PermissionError("Trusted replay requires `load_model_run(..., trusted=True)`")
    expected_blocks = tuple(run.features["block_names"])
    expected_names = tuple(run.features["feature_names"])
    expected_keys = tuple(tuple(key) for key in run.split["test_sample_keys"])
    if test_features.block_names != expected_blocks:
        raise ValueError("Test feature family does not match persisted run")
    if test_features.feature_names != expected_names:
        raise ValueError("Test feature names or channel order do not match persisted run")
    if test_features.sample_keys != expected_keys:
        raise ValueError("Test sample keys do not match persisted run")

    spec = get_model_spec(run.config.model_id)
    n_rows = test_features.X.shape[0]
    n_targets = run.test_targets.shape[1]
    if spec.topology == "independent":
        if len(run.pipelines) != n_targets:
            raise ValueError("Independent replay requires one pipeline per target")
        if spec.task == "classifier":
            raw = np.column_stack(
                [
                    pipeline.decision_function(test_features.X)
                    for pipeline in run.pipelines
                ]
            ).astype(np.float64, copy=False)
        else:
            raw = np.column_stack(
                [pipeline.predict(test_features.X) for pipeline in run.pipelines]
            ).astype(np.float64, copy=False)
    else:
        if len(run.pipelines) != 1:
            raise ValueError("Multi-output replay requires exactly one pipeline")
        raw = np.asarray(run.pipelines[0].predict(test_features.X), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
    if raw.shape != (n_rows, n_targets) or not np.isfinite(raw).all():
        raise ValueError("Replayed raw model outputs have an invalid shape or values")

    if spec.task == "classifier":
        model_items = run.results.get("models")
        if not isinstance(model_items, list) or len(model_items) != n_targets:
            raise ValueError("Classifier replay metadata is incomplete")
        scores = np.empty_like(raw)
        for index, item in enumerate(model_items):
            calibration = item.get("calibration")
            if not isinstance(calibration, dict):
                raise ValueError("Classifier calibration metadata is missing")
            scores[:, index] = expit(
                float(calibration["coefficient"]) * raw[:, index]
                + float(calibration["intercept"])
            )
    else:
        scores = np.clip(raw, 0.0, 1.0)
    scores = scores.astype(np.float64, copy=False)
    predictions = (
        scores >= float(run.results["prediction_threshold"])
    ).astype(np.int8)
    return scores, predictions


def summarize_model_runs(run_dirs: list[Path] | tuple[Path, ...]) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("At least one schema-v3 run is required")
    runs = tuple(load_model_run(path) for path in run_dirs)
    payload: dict[str, Any] = {
        "runs": [_model_run_summary(run) for run in runs],
        "combined": None,
    }
    if len(runs) == 2:
        payload["combined"] = _combine_model_runs(runs)
    elif len(runs) > 1 and {
        run.evaluation["protocol"] for run in runs
    } == {"within-subject"}:
        raise ValueError(
            "Within-subject evaluation requires exactly two complementary directions"
        )
    return payload


def compare_runs(run_dirs: list[Path] | tuple[Path, ...]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("Comparison requires at least two runs")
    runs = tuple(_load_comparison_run(path) for path in run_dirs)
    protocols = {(run.protocol, run.direction) for run in runs}
    if len(protocols) != 1:
        raise ValueError("Compared runs must use the same protocol and direction")
    reference_keys = runs[0].test_sample_keys
    if any(run.test_sample_keys != reference_keys for run in runs[1:]):
        raise ValueError("Compared runs must use identical ordered test sample keys")
    summaries = [
        {
            "run_dir": str(run.run_dir),
            "artifact_schema_version": run.schema_version,
            "model_id": run.model_id,
            "score_semantics": run.evaluation.get("score_semantics"),
            "selected_feature_family": run.evaluation["selected_feature_family"],
            "model_metrics": run.evaluation["model_metrics"],
            "model_bootstrap": run.evaluation["model_bootstrap"],
        }
        for run in runs
    ]
    reference_accuracy = summaries[0]["model_metrics"]["mean_balanced_accuracy"]
    for summary in summaries:
        summary["balanced_accuracy_difference_vs_first"] = (
            summary["model_metrics"]["mean_balanced_accuracy"]
            - reference_accuracy
        )
    return {
        "protocol": runs[0].protocol,
        "direction": runs[0].direction,
        "n_test_rows": len(reference_keys),
        "runs": summaries,
    }


def _load_comparison_run(run_dir: Path) -> ComparisonRun:
    manifest = _load_json(Path(run_dir) / "manifest.json")
    schema_version = manifest.get("schema_version")
    if schema_version == SCHEMA_VERSION:
        run = load_model_run(run_dir)
        return ComparisonRun(
            run_dir=run.run_dir,
            schema_version=SCHEMA_VERSION,
            model_id=run.config.model_id,
            protocol=str(run.evaluation["protocol"]),
            direction=str(run.evaluation["direction"]["name"]),
            test_sample_keys=tuple(
                tuple(key) for key in run.split["test_sample_keys"]
            ),
            evaluation=run.evaluation,
        )
    run = load_evaluation_run(run_dir)
    evaluation = dict(run.evaluation)
    evaluation.setdefault("score_semantics", "native_probability")
    return ComparisonRun(
        run_dir=run.run_dir,
        schema_version=int(run.manifest["schema_version"]),
        model_id="logistic-regression-independent",
        protocol=str(run.evaluation["protocol"]),
        direction=str(run.evaluation["direction"]["name"]),
        test_sample_keys=tuple(tuple(key) for key in run.split["test_sample_keys"]),
        evaluation=evaluation,
    )


def _write_fitted_payload(
    pipelines_dir: Path,
    *,
    result: DirectionEvaluationResult,
) -> tuple[list[str], list[dict[str, Any]]]:
    payload = result.fitted_model.payload
    if isinstance(payload, FittedCalibratedPixelModels):
        pipeline_files = []
        models = []
        for model in payload.models:
            filename = f"pipelines/pixel_{model.pixel_index:02d}.joblib"
            _write_pipeline(pipelines_dir.parent / filename, model.pipeline)
            pipeline_files.append(filename)
            models.append(
                {
                    "pixel_index": model.pixel_index,
                    "pixel_name": model.pixel_name,
                    "best_hyperparameters": _serialize_dataclass_like(
                        model.best_hyperparameters
                    ),
                    "best_cv_balanced_accuracy": model.best_cv_score,
                    "selected_feature_indices": (
                        model.selected_feature_indices.tolist()
                    ),
                    "selected_feature_names": list(model.selected_feature_names),
                    "calibration": {
                        "coefficient": model.calibration.coefficient,
                        "intercept": model.calibration.intercept,
                        "oof_fold_indices": (
                            model.calibration.oof_fold_indices.tolist()
                        ),
                    },
                    "candidate_scores": [
                        _serialize_dataclass_like(candidate)
                        for candidate in model.candidate_scores
                    ],
                }
            )
        return pipeline_files, models
    if isinstance(payload, FittedIndependentRegressionModels):
        pipeline_files = []
        models = []
        for model in payload.models:
            filename = f"pipelines/pixel_{model.pixel_index:02d}.joblib"
            _write_pipeline(pipelines_dir.parent / filename, model.pipeline)
            pipeline_files.append(filename)
            models.append(_serialize_regression_model(model))
        return pipeline_files, models
    if isinstance(payload, FittedMultiOutputRegressionModel):
        filename = "pipelines/multioutput.joblib"
        _write_pipeline(pipelines_dir.parent / filename, payload.pipeline)
        return [filename], [_serialize_regression_model(payload)]
    raise TypeError(f"Unsupported fitted payload type: {type(payload).__name__}")


def _serialize_regression_model(model: Any) -> dict[str, Any]:
    payload = {
        "best_hyperparameters": _serialize_dataclass_like(
            model.best_hyperparameters
        ),
        "best_cv_balanced_accuracy": model.best_cv_balanced_accuracy,
        "best_cv_clipped_mse": model.best_cv_clipped_mse,
        "selected_feature_indices": model.selected_feature_indices.tolist(),
        "selected_feature_names": list(model.selected_feature_names),
        "candidate_scores": [
            _serialize_dataclass_like(candidate)
            for candidate in model.candidate_scores
        ],
    }
    if hasattr(model, "pixel_index"):
        payload["pixel_index"] = model.pixel_index
        payload["pixel_name"] = model.pixel_name
    return payload


def _serialize_screening(screening: Any) -> dict[str, Any]:
    candidates = []
    for candidate in screening.candidates:
        payload = {"block_names": list(candidate.block_names)}
        for name in (
            "mean_score",
            "mean_balanced_accuracy",
            "mean_clipped_mse",
        ):
            if hasattr(candidate, name):
                payload[name] = float(getattr(candidate, name))
        for name in (
            "mean_pixel_scores",
            "fold_scores",
            "fold_balanced_accuracy",
            "fold_clipped_mse",
            "selected_feature_counts",
        ):
            if hasattr(candidate, name):
                payload[name] = getattr(candidate, name).tolist()
        candidates.append(payload)
    return {
        "selected_block_names": list(screening.selected_block_names),
        "candidates": candidates,
    }


def _serialize_dataclass_like(value: Any) -> dict[str, Any]:
    fields = getattr(value, "__dataclass_fields__", None)
    if not isinstance(fields, dict):
        raise TypeError(f"Expected dataclass-like metadata, got {type(value).__name__}")
    return {
        name: _json_value(getattr(value, name))
        for name in fields
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(fields, dict):
        return _serialize_dataclass_like(value)
    raise TypeError(f"Unsupported JSON metadata value: {value!r}")


def _build_split_payload(
    result: DirectionEvaluationResult,
    *,
    targets: PixelTargetDataset,
) -> dict[str, Any]:
    direction = result.direction
    payload = result.fitted_model.payload
    cross_validation = payload.cross_validation
    folds = []
    for fold in cross_validation.folds:
        fold_payload = {
            "fold_index": fold.fold_index,
            "train_indices": fold.train_indices.tolist(),
            "validation_indices": fold.validation_indices.tolist(),
            "train_subjects": list(fold.train_subjects),
            "validation_subjects": list(fold.validation_subjects),
        }
        if hasattr(fold, "pixel_index"):
            fold_payload["pixel_index"] = fold.pixel_index
        folds.append(fold_payload)
    return {
        "protocol": direction.protocol,
        "direction": direction.name,
        "train_target_indices": direction.train_indices.tolist(),
        "test_target_indices": direction.test_indices.tolist(),
        "train_subjects": list(direction.train_subjects),
        "test_subjects": list(direction.test_subjects),
        "train_sample_keys": [
            list(key) for key in result.fitted_model.training_sample_keys
        ],
        "test_sample_keys": [
            list(key) for key in result.prediction.test_sample_keys
        ],
        "test_image_fingerprints": [
            targets.image_fingerprints[int(index)]
            for index in direction.test_indices
        ],
        "cross_validation": {
            "n_samples": cross_validation.n_samples,
            "n_targets": getattr(
                cross_validation,
                "n_targets",
                getattr(cross_validation, "n_pixels", None),
            ),
            "n_splits": cross_validation.n_splits,
            "random_state": cross_validation.random_state,
            "folds": folds,
        },
    }


def _build_evaluation_payload(
    result: DirectionEvaluationResult,
    *,
    protocol_label: str,
) -> dict[str, Any]:
    direction = result.direction
    audit = result.audit
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": result.fitted_model.spec.model_id,
        "topology": result.fitted_model.spec.topology,
        "task": result.fitted_model.spec.task,
        "score_semantics": result.fitted_model.spec.score_semantics,
        "score_diagnostics": _diagnostics_payload(result.prediction),
        "protocol": direction.protocol,
        "protocol_label": protocol_label,
        "direction": {
            "name": direction.name,
            "label": direction.label,
            "train_trial": direction.train_trial,
            "test_trial": direction.test_trial,
        },
        "eligible_subjects": list(direction.eligible_subjects),
        "excluded_subjects": list(direction.excluded_subjects),
        "split": {
            "n_train_rows": int(direction.train_indices.size),
            "n_test_rows": int(direction.test_indices.size),
            "train_subjects": list(direction.train_subjects),
            "test_subjects": list(direction.test_subjects),
        },
        "split_audit": {
            "overlapping_subjects": list(audit.overlapping_subjects),
            "overlapping_sample_keys": [
                list(key) for key in audit.overlapping_sample_keys
            ],
            "overlapping_seeds": list(audit.overlapping_seeds),
            "overlapping_image_fingerprints": list(
                audit.overlapping_image_fingerprints
            ),
            "overlapping_trial_numbers": list(audit.overlapping_trial_numbers),
            "train_positive_counts": audit.train_positive_counts.tolist(),
            "test_positive_counts": audit.test_positive_counts.tolist(),
            "all_tasks_have_both_classes": audit.all_tasks_have_both_classes,
            "subject_contract_satisfied": audit.subject_contract_satisfied,
            "trial_contract_satisfied": audit.trial_contract_satisfied,
            "has_forbidden_leakage": audit.has_forbidden_leakage,
        },
        "selected_feature_family": list(
            result.fitted_model.selected_block_names
        ),
        "model_metrics": _metrics_payload(result.model_metrics),
        "model_bootstrap": _bootstrap_payload(result.model_bootstrap),
        "baselines": [
            {
                "name": baseline.prediction.name,
                "metrics": _metrics_payload(baseline.metrics),
                "array_files": {
                    "scores": (
                        f"arrays/baseline_{baseline.prediction.name}_scores.npy"
                    ),
                    "predictions": (
                        f"arrays/baseline_{baseline.prediction.name}_predictions.npy"
                    ),
                },
            }
            for baseline in result.baselines
        ],
    }


def _diagnostics_payload(prediction: ModelPrediction) -> dict[str, Any]:
    diagnostics = prediction.diagnostics
    return {
        "score_semantics": diagnostics.score_semantics,
        "clipped_below_zero_fraction": diagnostics.clipped_below_zero_fraction,
        "clipped_above_one_fraction": diagnostics.clipped_above_one_fraction,
    }


def _metrics_payload(metrics: PredictionMetrics) -> dict[str, Any]:
    return {
        "per_pixel_balanced_accuracy": (
            metrics.per_pixel_balanced_accuracy.tolist()
        ),
        "per_pixel_macro_f1": metrics.per_pixel_macro_f1.tolist(),
        "per_pixel_score_mse": metrics.per_pixel_score_mse.tolist(),
        "per_sample_iou": metrics.per_sample_iou.tolist(),
        "mean_balanced_accuracy": metrics.mean_balanced_accuracy,
        "mean_macro_f1": metrics.mean_macro_f1,
        "mean_score_mse": metrics.mean_score_mse,
        "mean_sample_iou": metrics.mean_sample_iou,
        "micro_iou": metrics.micro_iou,
        "bit_accuracy": metrics.bit_accuracy,
        "exact_match_accuracy": metrics.exact_match_accuracy,
        "mean_hamming_distance": metrics.mean_hamming_distance,
        "hamming_loss": metrics.hamming_loss,
    }


def _bootstrap_payload(interval: Any) -> dict[str, Any]:
    return {
        "metric": "mean_balanced_accuracy",
        "unit": "subject",
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "confidence_level": interval.confidence_level,
        "n_resamples": interval.n_resamples,
        "n_attempts": interval.n_attempts,
    }


def _load_baselines(
    run_dir: Path,
    *,
    evaluation: dict[str, Any],
) -> tuple[
    dict[str, NDArray[np.float64]],
    dict[str, NDArray[np.int8]],
]:
    items = evaluation.get("baselines")
    if not isinstance(items, list) or not items:
        raise ValueError("Evaluation metadata must contain baseline results")
    scores: dict[str, NDArray[np.float64]] = {}
    predictions: dict[str, NDArray[np.int8]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Evaluation baseline metadata is invalid")
        name = item["name"]
        files = item.get("array_files")
        if not isinstance(files, dict):
            raise ValueError(f"Baseline array metadata is missing: {name}")
        scores[name] = _load_array(
            _resolve_array_path(run_dir, files.get("scores")),
            np.float64,
        )
        predictions[name] = _load_array(
            _resolve_array_path(run_dir, files.get("predictions")),
            np.int8,
        )
    return scores, predictions


def _validate_evaluation_consistency(
    evaluation: dict[str, Any],
    *,
    split: dict[str, Any],
    features: dict[str, Any],
    results: dict[str, Any],
    scores: NDArray[np.float64],
    predictions: NDArray[np.int8],
    test_balanced_accuracy: NDArray[np.float64],
    test_targets: NDArray[np.int8],
    test_subject_ids: NDArray[np.int64],
    baseline_scores: dict[str, NDArray[np.float64]],
    baseline_predictions: dict[str, NDArray[np.int8]],
    config: RandomImageryModelConfig,
) -> None:
    if evaluation.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Evaluation payload does not declare schema version 3")
    if evaluation.get("model_id") != config.model_id:
        raise ValueError("Evaluation model does not match configuration")
    if evaluation.get("selected_feature_family") != features.get("block_names"):
        raise ValueError("Evaluation feature family does not match feature metadata")
    direction = evaluation.get("direction")
    if not isinstance(direction, dict) or direction.get("name") != split.get("direction"):
        raise ValueError("Evaluation direction does not match split metadata")
    if evaluation.get("protocol") != split.get("protocol"):
        raise ValueError("Evaluation protocol does not match split metadata")
    audit = evaluation.get("split_audit")
    if not isinstance(audit, dict) or audit.get("has_forbidden_leakage") is not False:
        raise ValueError("Evaluation split audit is invalid or reports forbidden leakage")
    expected_predictions = (
        scores >= float(results["prediction_threshold"])
    ).astype(np.int8)
    if not np.array_equal(predictions, expected_predictions):
        raise ValueError("Stored predictions differ from thresholded scores")

    metrics = evaluate_prediction_matrix(test_targets, predictions, scores)
    _assert_metrics_payload(
        evaluation.get("model_metrics"),
        expected=metrics,
        label="model",
    )
    if not np.allclose(
        test_balanced_accuracy,
        metrics.per_pixel_balanced_accuracy,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Stored balanced accuracy differs from model arrays")
    bootstrap = bootstrap_subject_mean_balanced_accuracy(
        test_targets,
        predictions,
        test_subject_ids,
        n_resamples=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    if evaluation.get("model_bootstrap") != _bootstrap_payload(bootstrap):
        raise ValueError("Stored model bootstrap differs from evaluation arrays")
    for item in evaluation["baselines"]:
        name = item["name"]
        _assert_metrics_payload(
            item.get("metrics"),
            expected=evaluate_prediction_matrix(
                test_targets,
                baseline_predictions[name],
                baseline_scores[name],
            ),
            label=f"baseline {name}",
        )


def _assert_metrics_payload(
    payload: object,
    *,
    expected: PredictionMetrics,
    label: str,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Stored {label} metrics are invalid")
    expected_payload = _metrics_payload(expected)
    for key, expected_value in expected_payload.items():
        actual_value = payload.get(key)
        if isinstance(expected_value, list):
            if not isinstance(actual_value, list) or not np.allclose(
                actual_value,
                expected_value,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"Stored {label} metric differs from arrays: {key}")
        elif actual_value != expected_value:
            raise ValueError(f"Stored {label} metric differs from arrays: {key}")


def _model_run_summary(run: LoadedModelRun) -> dict[str, Any]:
    return {
        "run_dir": str(run.run_dir),
        "artifact_schema_version": SCHEMA_VERSION,
        **run.evaluation,
    }


def _combine_model_runs(
    runs: tuple[LoadedModelRun, LoadedModelRun],
) -> dict[str, Any]:
    by_direction = {
        run.evaluation["direction"]["name"]: run
        for run in runs
    }
    expected = ("trial-1-to-trial-2", "trial-2-to-trial-1")
    if set(by_direction) != set(expected):
        raise ValueError(
            "Two-run aggregation requires complementary within-subject directions"
        )
    ordered = tuple(by_direction[name] for name in expected)
    if any(run.evaluation["protocol"] != "within-subject" for run in ordered):
        raise ValueError("Only within-subject directions can be combined")
    if ordered[0].config != ordered[1].config:
        raise ValueError("Combined runs must use identical configurations")
    first_keys = {tuple(key) for key in ordered[0].split["test_sample_keys"]}
    second_keys = {tuple(key) for key in ordered[1].split["test_sample_keys"]}
    if first_keys & second_keys:
        raise ValueError("Combined directions overlap test sample keys")

    test_targets = np.concatenate([run.test_targets for run in ordered], axis=0)
    scores = np.concatenate([run.scores for run in ordered], axis=0)
    predictions = np.concatenate([run.predictions for run in ordered], axis=0)
    subject_ids = np.concatenate([run.test_subject_ids for run in ordered])
    metrics = evaluate_prediction_matrix(test_targets, predictions, scores)
    bootstrap = bootstrap_subject_mean_balanced_accuracy(
        test_targets,
        predictions,
        subject_ids,
        n_resamples=ordered[0].config.bootstrap_iterations,
        random_state=ordered[0].config.random_state,
    )
    baselines = []
    for name in ordered[0].baseline_scores:
        baseline_scores = np.concatenate(
            [run.baseline_scores[name] for run in ordered],
            axis=0,
        )
        baseline_predictions = np.concatenate(
            [run.baseline_predictions[name] for run in ordered],
            axis=0,
        )
        baselines.append(
            {
                "name": name,
                "metrics": _metrics_payload(
                    evaluate_prediction_matrix(
                        test_targets,
                        baseline_predictions,
                        baseline_scores,
                    )
                ),
            }
        )
    return {
        "model_id": ordered[0].config.model_id,
        "protocol": "within-subject",
        "direction_names": list(expected),
        "run_dirs": [str(run.run_dir) for run in ordered],
        "n_test_rows": int(test_targets.shape[0]),
        "n_subjects": int(np.unique(subject_ids).size),
        "selected_feature_families": [
            run.evaluation["selected_feature_family"] for run in ordered
        ],
        "model_metrics": _metrics_payload(metrics),
        "model_bootstrap": _bootstrap_payload(bootstrap),
        "baselines": baselines,
    }


def _validate_direction_result(
    result: DirectionEvaluationResult,
    *,
    targets: PixelTargetDataset,
    config: RandomImageryModelConfig,
) -> None:
    if result.fitted_model.spec.model_id != config.model_id:
        raise ValueError("Fitted model does not match artifact configuration")
    if not np.array_equal(
        result.fitted_model.training_target_indices,
        result.direction.train_indices,
    ):
        raise ValueError("Fitted training rows do not match direction split")
    if not np.array_equal(
        result.prediction.test_target_indices,
        result.direction.test_indices,
    ):
        raise ValueError("Predicted test rows do not match direction split")
    expected_train_keys = tuple(
        targets.sample_keys[int(index)] for index in result.direction.train_indices
    )
    expected_test_keys = tuple(
        targets.sample_keys[int(index)] for index in result.direction.test_indices
    )
    if result.fitted_model.training_sample_keys != expected_train_keys:
        raise ValueError("Fitted training sample keys do not match targets")
    if result.prediction.test_sample_keys != expected_test_keys:
        raise ValueError("Prediction sample keys do not match targets")


def _validate_loaded_arrays(
    *,
    scores: NDArray[np.float64],
    predictions: NDArray[np.int8],
    test_balanced_accuracy: NDArray[np.float64],
    train_targets: NDArray[np.int8],
    test_targets: NDArray[np.int8],
    test_subject_ids: NDArray[np.int64],
) -> None:
    if scores.ndim != 2 or scores.dtype != np.dtype(np.float64):
        raise TypeError("Loaded scores must be a float64 matrix")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Loaded scores must be finite and in [0, 1]")
    n_test, n_targets = scores.shape
    if predictions.shape != scores.shape or predictions.dtype != np.dtype(np.int8):
        raise TypeError("Loaded predictions must be an int8 matrix matching scores")
    if not np.isin(predictions, (0, 1)).all():
        raise ValueError("Loaded predictions must be binary")
    if (
        test_balanced_accuracy.shape != (n_targets,)
        or test_balanced_accuracy.dtype != np.dtype(np.float64)
    ):
        raise TypeError("Loaded balanced accuracy must match target columns")
    if train_targets.ndim != 2 or train_targets.dtype != np.dtype(np.int8):
        raise TypeError("Loaded train targets must be a two-dimensional int8 array")
    if test_targets.shape != (n_test, n_targets) or test_targets.dtype != np.dtype(np.int8):
        raise TypeError("Loaded test targets must match scores")
    if test_subject_ids.shape != (n_test,) or test_subject_ids.dtype != np.dtype(np.int64):
        raise TypeError("Loaded test subject IDs must match test rows")


def _resolve_array_path(run_dir: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("Experiment array filename must be a string")
    path = Path(relative_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "arrays"
        or path.suffix != ".npy"
    ):
        raise ValueError(f"Unsafe experiment array filename: {relative_path!r}")
    return run_dir / path


def _resolve_pipeline_path(
    run_dir: Path,
    relative_path: object,
    *,
    manifest_files: set[str],
) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("Experiment pipeline filename must be a string")
    path = Path(relative_path)
    normalized = path.as_posix()
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "pipelines"
        or path.suffix != ".joblib"
    ):
        raise ValueError(f"Unsafe experiment pipeline filename: {relative_path!r}")
    if normalized not in manifest_files:
        raise ValueError(f"Pipeline is not present in manifest: {relative_path}")
    return run_dir / path


def _validate_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("file_count") != len(files):
        raise ValueError("Experiment manifest file inventory is invalid")
    expected_files = set(files) | {"manifest.json"}
    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            f"Experiment file inventory mismatch; missing={missing}, unexpected={unexpected}"
        )
    for relative_path, metadata in files.items():
        path = run_dir / relative_path
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid manifest metadata for {relative_path}")
        if path.stat().st_size != metadata.get("size"):
            raise ValueError(f"Experiment file size mismatch: {relative_path}")
        if _sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Experiment file hash mismatch: {relative_path}")


def _build_environment_payload() -> dict[str, Any]:
    package_names = (
        "joblib",
        "mne",
        "numpy",
        "omegaconf",
        "pydantic",
        "scikit-learn",
        "scipy",
    )
    git_commit, git_dirty = _git_state()
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            package_name: importlib.metadata.version(package_name)
            for package_name in package_names
        },
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _publish_run(temporary_dir: Path, *, run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Experiment run already exists and is immutable: {run_dir}")
    os.replace(temporary_dir, run_dir)
    _fsync_directory(run_dir.parent)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid experiment JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment JSON file must contain an object: {path}")
    return payload


def _write_array(path: Path, array: NDArray[Any]) -> None:
    with path.open("wb") as file:
        np.save(file, array, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())


def _load_array(path: Path, dtype: np.dtype[Any] | type[Any]) -> NDArray[Any]:
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"Experiment array {path.name} has unexpected dtype {array.dtype}")
    array.setflags(write=False)
    return array


def _write_pipeline(path: Path, pipeline: Pipeline) -> None:
    with path.open("wb") as file:
        joblib.dump(pipeline, file, compress=3)
        file.flush()
        os.fsync(file.fileno())


def _load_pipeline(path: Path) -> Pipeline:
    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Experiment model is not an sklearn Pipeline: {path}")
    return pipeline


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ComparisonRun",
    "LoadedModelRun",
    "compare_runs",
    "load_model_run",
    "model_run_dir",
    "replay_model_predictions",
    "summarize_model_runs",
    "write_model_protocol_runs",
]
