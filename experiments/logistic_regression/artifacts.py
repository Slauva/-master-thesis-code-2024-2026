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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from experiments.logistic_regression.baselines import build_non_eeg_baselines
from experiments.logistic_regression.config import (
    LogisticRegressionExperimentConfig,
    build_evaluation_config_hash,
    build_experiment_config_hash,
)
from experiments.logistic_regression.metrics import (
    PredictionMetrics,
    bootstrap_subject_mean_balanced_accuracy,
    evaluate_prediction_matrix,
)
from experiments.logistic_regression.runner import (
    DirectionEvaluationResult,
    ProtocolEvaluationResult,
)
from experiments.logistic_regression.schemas import (
    AlignedFeaturePartition,
    EvaluationDirection,
    FeatureScreeningResult,
    PixelGridSearchResult,
    PixelTargetDataset,
    SubjectSplit,
)


@dataclass(frozen=True, slots=True)
class LoadedExperimentRun:
    run_dir: Path
    manifest: dict[str, Any]
    config: LogisticRegressionExperimentConfig
    environment: dict[str, Any]
    split: dict[str, Any]
    features: dict[str, Any]
    screening: dict[str, Any] | None
    results: dict[str, Any]
    pipelines: tuple[Pipeline, ...]
    probabilities: NDArray[np.float64]
    predictions: NDArray[np.int8]
    test_balanced_accuracy: NDArray[np.float64]
    train_targets: NDArray[np.int8]
    test_targets: NDArray[np.int8]
    test_subject_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        n_test, n_pixels = self.probabilities.shape
        if self.probabilities.dtype != np.dtype(np.float64) or not np.isfinite(
            self.probabilities
        ).all():
            raise TypeError("Loaded probabilities must be a finite float64 matrix")
        if self.predictions.shape != (n_test, n_pixels) or self.predictions.dtype != np.dtype(np.int8):
            raise TypeError("Loaded predictions must be an int8 matrix matching probabilities")
        if not np.isin(self.predictions, (0, 1)).all():
            raise ValueError("Loaded predictions must be binary")
        if self.test_balanced_accuracy.shape != (n_pixels,) or self.test_balanced_accuracy.dtype != np.dtype(
            np.float64
        ):
            raise TypeError("Loaded test balanced accuracy must match pixels")
        if self.train_targets.ndim != 2 or self.train_targets.dtype != np.dtype(np.int8):
            raise TypeError("Loaded train targets must be a two-dimensional int8 array")
        if self.test_targets.shape != (n_test, n_pixels) or self.test_targets.dtype != np.dtype(np.int8):
            raise TypeError("Loaded test targets must match recorded predictions")
        if self.test_subject_ids.shape != (n_test,) or self.test_subject_ids.dtype != np.dtype(np.int64):
            raise TypeError("Loaded test subject IDs must match recorded test rows")
        if len(self.pipelines) != n_pixels:
            raise ValueError("Loaded run must contain one pipeline per pixel")
        if any(not isinstance(pipeline, Pipeline) for pipeline in self.pipelines):
            raise TypeError("Loaded models must be sklearn Pipelines")


@dataclass(frozen=True, slots=True)
class LoadedEvaluationRun:
    run_dir: Path
    manifest: dict[str, Any]
    config: LogisticRegressionExperimentConfig
    environment: dict[str, Any]
    split: dict[str, Any]
    features: dict[str, Any]
    screening: dict[str, Any] | None
    results: dict[str, Any]
    evaluation: dict[str, Any]
    probabilities: NDArray[np.float64]
    predictions: NDArray[np.int8]
    test_balanced_accuracy: NDArray[np.float64]
    train_targets: NDArray[np.int8]
    test_targets: NDArray[np.int8]
    test_subject_ids: NDArray[np.int64]
    baseline_probabilities: dict[str, NDArray[np.float64]]
    baseline_predictions: dict[str, NDArray[np.int8]]

    def __post_init__(self) -> None:
        _validate_loaded_arrays(
            probabilities=self.probabilities,
            predictions=self.predictions,
            test_balanced_accuracy=self.test_balanced_accuracy,
            train_targets=self.train_targets,
            test_targets=self.test_targets,
            test_subject_ids=self.test_subject_ids,
        )
        baseline_names = tuple(item["name"] for item in self.evaluation["baselines"])
        if set(self.baseline_probabilities) != set(baseline_names):
            raise ValueError("Loaded baseline probability arrays do not match evaluation metadata")
        if set(self.baseline_predictions) != set(baseline_names):
            raise ValueError("Loaded baseline prediction arrays do not match evaluation metadata")
        for name in baseline_names:
            if self.baseline_probabilities[name].shape != self.test_targets.shape:
                raise ValueError(f"Baseline probabilities do not match test targets: {name}")
            if self.baseline_predictions[name].shape != self.test_targets.shape:
                raise ValueError(f"Baseline predictions do not match test targets: {name}")


def write_experiment_run(
    result: PixelGridSearchResult,
    *,
    targets: PixelTargetDataset,
    split: SubjectSplit,
    config: LogisticRegressionExperimentConfig,
    feature_config_hash: str,
    screening: FeatureScreeningResult | None = None,
) -> Path:
    if config.artifacts.schema_version != 1:
        raise ValueError(
            "Legacy `write_experiment_run` only writes schema-v1 artifacts; "
            "use `write_protocol_evaluation_runs` for schema-v2 evaluations"
        )
    _validate_write_inputs(result, targets=targets, split=split)
    config_hash = build_experiment_config_hash(config)
    root = Path(config.artifacts.root)
    run_dir = root / config_hash
    root.mkdir(parents=True, exist_ok=True)
    if run_dir.exists() and not config.artifacts.overwrite:
        raise FileExistsError(f"Experiment run already exists and is immutable: {run_dir}")

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{config_hash}-", dir=root))
    try:
        _write_run_payload(
            temporary_dir,
            result=result,
            targets=targets,
            split=split,
            config=config,
            config_hash=config_hash,
            feature_config_hash=feature_config_hash,
            screening=screening,
        )
        _publish_run(
            temporary_dir,
            run_dir=run_dir,
            overwrite=config.artifacts.overwrite,
        )
        return run_dir
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def evaluation_run_dir(
    config: LogisticRegressionExperimentConfig,
    *,
    protocol: str,
    direction: str,
) -> Path:
    return Path(config.artifacts.root) / build_evaluation_config_hash(
        config,
        protocol=protocol,
        direction=direction,
    )


def write_protocol_evaluation_runs(
    result: ProtocolEvaluationResult,
    *,
    targets: PixelTargetDataset,
    config: LogisticRegressionExperimentConfig,
    feature_config_hash: str,
) -> tuple[Path, ...]:
    if config.artifacts.schema_version != 2:
        raise ValueError("Protocol-aware evaluation artifacts require schema version 2")
    if config.artifacts.overwrite:
        raise ValueError("Schema-v2 evaluation runs are immutable and cannot enable overwrite")
    return tuple(
        _write_direction_evaluation_run(
            direction_result,
            protocol_label=result.definition.label,
            targets=targets,
            config=config,
            feature_config_hash=feature_config_hash,
        )
        for direction_result in result.directions
    )


def _write_direction_evaluation_run(
    result: DirectionEvaluationResult,
    *,
    protocol_label: str,
    targets: PixelTargetDataset,
    config: LogisticRegressionExperimentConfig,
    feature_config_hash: str,
) -> Path:
    _validate_write_inputs(
        result.grid_search,
        targets=targets,
        split=result.direction,
    )
    config_hash = build_evaluation_config_hash(
        config,
        protocol=result.direction.protocol,
        direction=result.direction.name,
    )
    root = Path(config.artifacts.root)
    run_dir = root / config_hash
    root.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"Experiment run already exists and is immutable: {run_dir}")

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{config_hash}-", dir=root))
    try:
        evaluation_payload = _build_evaluation_payload(
            result,
            protocol_label=protocol_label,
        )
        _write_run_payload(
            temporary_dir,
            result=result.grid_search,
            targets=targets,
            split=result.direction,
            config=config,
            config_hash=config_hash,
            feature_config_hash=feature_config_hash,
            screening=result.screening,
            evaluation_payload=evaluation_payload,
            baseline_arrays={
                baseline.prediction.name: (
                    baseline.prediction.probabilities,
                    baseline.prediction.predictions,
                )
                for baseline in result.baselines
            },
            writer=(
                "experiments.logistic_regression.artifacts."
                "write_protocol_evaluation_runs"
            ),
        )
        _publish_run(temporary_dir, run_dir=run_dir, overwrite=False)
        return run_dir
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def load_experiment_run(
    run_dir: Path,
    *,
    trusted: bool = False,
) -> LoadedExperimentRun:
    resolved_dir = Path(run_dir)
    manifest_path = resolved_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    _validate_manifest(resolved_dir, manifest)

    config_payload = _load_json(resolved_dir / "config.json")
    config = LogisticRegressionExperimentConfig.model_validate(config_payload)
    schema_version = manifest.get("schema_version")
    evaluation_path = resolved_dir / "evaluation.json"
    evaluation = _load_json(evaluation_path) if evaluation_path.is_file() else None
    expected_hash = _expected_config_hash(
        config,
        schema_version=schema_version,
        evaluation=evaluation,
    )
    if manifest.get("config_hash") != expected_hash or resolved_dir.name != expected_hash:
        raise ValueError("Experiment config hash does not match the manifest and run directory")
    if schema_version != config.artifacts.schema_version:
        raise ValueError("Experiment artifact schema does not match the resolved configuration")

    environment = _load_json(resolved_dir / "environment.json")
    split = _load_json(resolved_dir / "split.json")
    features = _load_json(resolved_dir / "features.json")
    results = _load_json(resolved_dir / "results.json")
    screening_path = resolved_dir / "screening.json"
    screening = _load_json(screening_path) if screening_path.is_file() else None

    arrays_dir = resolved_dir / "arrays"
    probabilities = _load_array(arrays_dir / "probabilities.npy", np.float64)
    predictions = _load_array(arrays_dir / "predictions.npy", np.int8)
    test_balanced_accuracy = _load_array(
        arrays_dir / "test_balanced_accuracy.npy",
        np.float64,
    )
    train_targets = _load_array(arrays_dir / "train_targets.npy", np.int8)
    test_targets = _load_array(arrays_dir / "test_targets.npy", np.int8)
    test_subject_ids = _load_array(arrays_dir / "test_subject_ids.npy", np.int64)

    if not trusted:
        raise PermissionError(
            "Refusing to load joblib pipelines from an untrusted run; pass `trusted=True` "
            "only for locally produced, hash-validated artifacts"
        )
    pipeline_files = results.get("pipeline_files")
    if not isinstance(pipeline_files, list) or not pipeline_files:
        raise ValueError("Experiment results do not contain pipeline filenames")
    pipelines = tuple(
        _load_pipeline(
            _resolve_pipeline_path(
                resolved_dir,
                filename,
                manifest_files=set(manifest["files"]),
            )
        )
        for filename in pipeline_files
    )
    return LoadedExperimentRun(
        run_dir=resolved_dir,
        manifest=manifest,
        config=config,
        environment=environment,
        split=split,
        features=features,
        screening=screening,
        results=results,
        pipelines=pipelines,
        probabilities=probabilities,
        predictions=predictions,
        test_balanced_accuracy=test_balanced_accuracy,
        train_targets=train_targets,
        test_targets=test_targets,
        test_subject_ids=test_subject_ids,
    )


def load_evaluation_run(run_dir: Path) -> LoadedEvaluationRun:
    resolved_dir = Path(run_dir)
    manifest_path = resolved_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    _validate_manifest(resolved_dir, manifest)

    config = LogisticRegressionExperimentConfig.model_validate(
        _load_json(resolved_dir / "config.json")
    )
    schema_version = manifest.get("schema_version")
    if schema_version != config.artifacts.schema_version:
        raise ValueError("Experiment artifact schema does not match the resolved configuration")

    environment = _load_json(resolved_dir / "environment.json")
    split = _load_json(resolved_dir / "split.json")
    features = _load_json(resolved_dir / "features.json")
    results = _load_json(resolved_dir / "results.json")
    screening_path = resolved_dir / "screening.json"
    screening = _load_json(screening_path) if screening_path.is_file() else None

    arrays_dir = resolved_dir / "arrays"
    probabilities = _load_array(arrays_dir / "probabilities.npy", np.float64)
    predictions = _load_array(arrays_dir / "predictions.npy", np.int8)
    test_balanced_accuracy = _load_array(
        arrays_dir / "test_balanced_accuracy.npy",
        np.float64,
    )
    train_targets = _load_array(arrays_dir / "train_targets.npy", np.int8)
    test_targets = _load_array(arrays_dir / "test_targets.npy", np.int8)
    test_subject_ids = _load_array(arrays_dir / "test_subject_ids.npy", np.int64)
    _validate_loaded_arrays(
        probabilities=probabilities,
        predictions=predictions,
        test_balanced_accuracy=test_balanced_accuracy,
        train_targets=train_targets,
        test_targets=test_targets,
        test_subject_ids=test_subject_ids,
    )

    if schema_version == 1:
        baseline_probabilities, baseline_predictions = _build_legacy_baseline_arrays(
            train_targets,
            n_test_samples=test_targets.shape[0],
            config=config,
        )
        evaluation = _build_legacy_evaluation_payload(
            split=split,
            features=features,
            test_targets=test_targets,
            probabilities=probabilities,
            predictions=predictions,
            test_subject_ids=test_subject_ids,
            baseline_probabilities=baseline_probabilities,
            baseline_predictions=baseline_predictions,
            config=config,
        )
    elif schema_version == 2:
        evaluation = _load_json(resolved_dir / "evaluation.json")
        baseline_probabilities, baseline_predictions = _load_baseline_arrays(
            resolved_dir,
            evaluation=evaluation,
        )
        _validate_evaluation_consistency(
            evaluation,
            split=split,
            features=features,
            test_targets=test_targets,
            probabilities=probabilities,
            predictions=predictions,
            test_balanced_accuracy=test_balanced_accuracy,
            test_subject_ids=test_subject_ids,
            baseline_probabilities=baseline_probabilities,
            baseline_predictions=baseline_predictions,
            config=config,
        )
    else:
        raise ValueError(f"Unsupported experiment artifact schema: {schema_version!r}")

    expected_hash = _expected_config_hash(
        config,
        schema_version=schema_version,
        evaluation=evaluation,
    )
    if manifest.get("config_hash") != expected_hash or resolved_dir.name != expected_hash:
        raise ValueError("Experiment config hash does not match the manifest and run directory")
    return LoadedEvaluationRun(
        run_dir=resolved_dir,
        manifest=manifest,
        config=config,
        environment=environment,
        split=split,
        features=features,
        screening=screening,
        results=results,
        evaluation=evaluation,
        probabilities=probabilities,
        predictions=predictions,
        test_balanced_accuracy=test_balanced_accuracy,
        train_targets=train_targets,
        test_targets=test_targets,
        test_subject_ids=test_subject_ids,
        baseline_probabilities=baseline_probabilities,
        baseline_predictions=baseline_predictions,
    )


def summarize_evaluation_runs(run_dirs: tuple[Path, ...] | list[Path]) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("At least one experiment run is required")
    runs = tuple(load_evaluation_run(Path(run_dir)) for run_dir in run_dirs)
    payload: dict[str, Any] = {
        "runs": [_build_loaded_run_summary(run) for run in runs],
        "combined": None,
    }
    if len(runs) == 2:
        payload["combined"] = _combine_loaded_within_subject_runs(runs)
    elif len(runs) > 1:
        protocols = {run.evaluation["protocol"] for run in runs}
        if protocols == {"within-subject"}:
            raise ValueError(
                "Within-subject evaluation requires exactly two complementary directions"
            )
    return payload


def reproduce_experiment_predictions(
    run: LoadedExperimentRun,
    *,
    test_features: AlignedFeaturePartition,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    expected_block_names = tuple(run.features["block_names"])
    expected_feature_names = tuple(run.features["feature_names"])
    expected_test_keys = tuple(tuple(key) for key in run.split["test_sample_keys"])
    if test_features.block_names != expected_block_names:
        raise ValueError("Test feature family does not match the persisted run")
    if test_features.feature_names != expected_feature_names:
        raise ValueError("Test feature names or channel order do not match the persisted run")
    if test_features.sample_keys != expected_test_keys:
        raise ValueError("Test sample keys do not match the persisted run")

    probabilities = np.empty_like(run.probabilities)
    for pixel_index, pipeline in enumerate(run.pipelines):
        classifier = pipeline.named_steps.get("model")
        if not isinstance(classifier, LogisticRegression):
            raise TypeError("Persisted pipeline contains an unexpected classifier")
        positive_columns = np.flatnonzero(classifier.classes_ == 1)
        if positive_columns.size != 1:
            raise ValueError(f"Persisted pixel {pixel_index} pipeline lacks one positive class")
        probabilities[:, pixel_index] = pipeline.predict_proba(test_features.X)[
            :,
            int(positive_columns[0]),
        ]
    threshold = float(run.results["prediction_threshold"])
    predictions = (probabilities >= threshold).astype(np.int8)
    return probabilities, predictions


def _write_run_payload(
    run_dir: Path,
    *,
    result: PixelGridSearchResult,
    targets: PixelTargetDataset,
    split: SubjectSplit | EvaluationDirection,
    config: LogisticRegressionExperimentConfig,
    config_hash: str,
    feature_config_hash: str,
    screening: FeatureScreeningResult | None,
    evaluation_payload: dict[str, Any] | None = None,
    baseline_arrays: dict[
        str,
        tuple[NDArray[np.float64], NDArray[np.int8]],
    ]
    | None = None,
    writer: str = "experiments.logistic_regression.artifacts.write_experiment_run",
) -> None:
    (run_dir / "arrays").mkdir()
    (run_dir / "pipelines").mkdir()

    _write_json(run_dir / "config.json", config.model_dump(mode="json"))
    _write_json(run_dir / "environment.json", _build_environment_payload())
    _write_json(
        run_dir / "split.json",
        _build_split_payload(
            result,
            targets=targets,
            split=split,
            config=config,
        ),
    )
    _write_json(
        run_dir / "features.json",
        {
            "feature_config_hash": feature_config_hash,
            "block_names": list(result.fitted_models.block_names),
            "feature_names": list(result.fitted_models.feature_names),
        },
    )
    if screening is not None:
        _write_json(run_dir / "screening.json", _build_screening_payload(screening))
    if evaluation_payload is not None:
        _write_json(run_dir / "evaluation.json", evaluation_payload)

    pipeline_files: list[str] = []
    for model in result.fitted_models.models:
        relative_path = f"pipelines/pixel_{model.pixel_index:02d}.joblib"
        _write_pipeline(run_dir / relative_path, model.pipeline)
        pipeline_files.append(relative_path)
    _write_json(
        run_dir / "results.json",
        _build_results_payload(result, pipeline_files=pipeline_files),
    )

    train_indices = result.fitted_models.training_target_indices
    test_indices = result.test_target_indices
    arrays_dir = run_dir / "arrays"
    _write_array(arrays_dir / "probabilities.npy", result.probabilities)
    _write_array(arrays_dir / "predictions.npy", result.predictions)
    _write_array(arrays_dir / "test_balanced_accuracy.npy", result.test_balanced_accuracy)
    _write_array(arrays_dir / "train_targets.npy", targets.y[train_indices])
    _write_array(arrays_dir / "test_targets.npy", targets.y[test_indices])
    _write_array(arrays_dir / "test_subject_ids.npy", targets.subject_ids[test_indices])
    for name, (baseline_probabilities, baseline_predictions) in (
        baseline_arrays or {}
    ).items():
        _write_array(
            arrays_dir / f"baseline_{name}_probabilities.npy",
            baseline_probabilities,
        )
        _write_array(
            arrays_dir / f"baseline_{name}_predictions.npy",
            baseline_predictions,
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
            "schema_version": config.artifacts.schema_version,
            "config_hash": config_hash,
            "generated_at": datetime.now(UTC).isoformat(),
            "writer": writer,
            "file_count": len(files),
            "files": files,
        },
    )
    _fsync_directory(run_dir)


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


def _build_split_payload(
    result: PixelGridSearchResult,
    *,
    targets: PixelTargetDataset,
    split: SubjectSplit | EvaluationDirection,
    config: LogisticRegressionExperimentConfig,
) -> dict[str, Any]:
    cross_validation = result.fitted_models.cross_validation
    return {
        "random_state": config.split.random_state,
        "test_size": config.split.test_size,
        "protocol": getattr(split, "protocol", "cross-subject"),
        "direction": getattr(split, "name", "cross-subject"),
        "train_target_indices": split.train_indices.tolist(),
        "test_target_indices": split.test_indices.tolist(),
        "train_subjects": list(split.train_subjects),
        "test_subjects": list(split.test_subjects),
        "train_sample_keys": [list(key) for key in result.fitted_models.training_sample_keys],
        "test_sample_keys": [list(key) for key in result.test_sample_keys],
        "test_image_fingerprints": [
            targets.image_fingerprints[int(index)]
            for index in result.test_target_indices
        ],
        "cross_validation": {
            "n_samples": cross_validation.n_samples,
            "n_pixels": cross_validation.n_pixels,
            "n_splits": cross_validation.n_splits,
            "random_state": cross_validation.random_state,
            "folds": [
                {
                    "pixel_index": fold.pixel_index,
                    "fold_index": fold.fold_index,
                    "train_indices": fold.train_indices.tolist(),
                    "validation_indices": fold.validation_indices.tolist(),
                    "train_subjects": list(fold.train_subjects),
                    "validation_subjects": list(fold.validation_subjects),
                }
                for fold in cross_validation.folds
            ],
        },
    }


def _build_screening_payload(screening: FeatureScreeningResult) -> dict[str, Any]:
    return {
        "selected_block_names": list(screening.selected_block_names),
        "candidates": [
            {
                "block_names": list(candidate.block_names),
                "mean_score": candidate.mean_score,
                "mean_pixel_scores": candidate.mean_pixel_scores.tolist(),
                "fold_scores": candidate.fold_scores.tolist(),
                "selected_feature_counts": candidate.selected_feature_counts.tolist(),
            }
            for candidate in screening.candidates
        ],
    }


def _build_results_payload(
    result: PixelGridSearchResult,
    *,
    pipeline_files: list[str],
) -> dict[str, Any]:
    return {
        "prediction_threshold": result.threshold,
        "pipeline_files": pipeline_files,
        "probability_shape": list(result.probabilities.shape),
        "prediction_shape": list(result.predictions.shape),
        "mean_test_balanced_accuracy": float(result.test_balanced_accuracy.mean()),
        "models": [
            {
                "pixel_index": model.pixel_index,
                "pixel_name": model.pixel_name,
                "best_hyperparameters": {
                    "select_k": model.best_hyperparameters.select_k,
                    "c": model.best_hyperparameters.c,
                    "penalty": model.best_hyperparameters.penalty,
                    "class_weight": model.best_hyperparameters.class_weight,
                },
                "best_cv_score": model.best_cv_score,
                "selected_feature_indices": model.selected_feature_indices.tolist(),
                "selected_feature_names": list(model.selected_feature_names),
                "coefficients": model.coefficients.tolist(),
                "intercept": model.intercept,
                "n_iter": model.n_iter,
                "candidate_scores": [
                    {
                        "hyperparameters": {
                            "select_k": candidate.hyperparameters.select_k,
                            "c": candidate.hyperparameters.c,
                            "penalty": candidate.hyperparameters.penalty,
                            "class_weight": candidate.hyperparameters.class_weight,
                        },
                        "mean_score": candidate.mean_score,
                        "std_score": candidate.std_score,
                        "rank": candidate.rank,
                    }
                    for candidate in model.candidate_scores
                ],
            }
            for model in result.fitted_models.models
        ],
    }


def _build_evaluation_payload(
    result: DirectionEvaluationResult,
    *,
    protocol_label: str,
) -> dict[str, Any]:
    direction = result.direction
    audit = result.audit
    return {
        "schema_version": 2,
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
        "selected_feature_family": list(result.screening.selected_block_names),
        "model_metrics": _metrics_payload(result.model_metrics),
        "model_bootstrap": _bootstrap_payload(result.model_bootstrap),
        "baselines": [
            {
                "name": baseline.prediction.name,
                "metrics": _metrics_payload(baseline.metrics),
                "array_files": {
                    "probabilities": (
                        f"arrays/baseline_{baseline.prediction.name}_probabilities.npy"
                    ),
                    "predictions": (
                        f"arrays/baseline_{baseline.prediction.name}_predictions.npy"
                    ),
                },
            }
            for baseline in result.baselines
        ],
    }


def _metrics_payload(metrics: PredictionMetrics) -> dict[str, Any]:
    return {
        "per_pixel_balanced_accuracy": (
            metrics.per_pixel_balanced_accuracy.tolist()
        ),
        "per_pixel_macro_f1": metrics.per_pixel_macro_f1.tolist(),
        "per_pixel_brier_score": metrics.per_pixel_brier_score.tolist(),
        "per_sample_iou": metrics.per_sample_iou.tolist(),
        "mean_balanced_accuracy": metrics.mean_balanced_accuracy,
        "mean_macro_f1": metrics.mean_macro_f1,
        "mean_brier_score": metrics.mean_brier_score,
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


def _build_legacy_evaluation_payload(
    *,
    split: dict[str, Any],
    features: dict[str, Any],
    test_targets: NDArray[np.int8],
    probabilities: NDArray[np.float64],
    predictions: NDArray[np.int8],
    test_subject_ids: NDArray[np.int64],
    baseline_probabilities: dict[str, NDArray[np.float64]],
    baseline_predictions: dict[str, NDArray[np.int8]],
    config: LogisticRegressionExperimentConfig,
) -> dict[str, Any]:
    model_metrics = evaluate_prediction_matrix(
        test_targets,
        predictions,
        probabilities,
    )
    model_bootstrap = bootstrap_subject_mean_balanced_accuracy(
        test_targets,
        predictions,
        test_subject_ids,
        n_resamples=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    train_subjects = tuple(int(value) for value in split["train_subjects"])
    test_subjects = tuple(int(value) for value in split["test_subjects"])
    return {
        "schema_version": 1,
        "protocol": "cross-subject",
        "protocol_label": "cross-subject generalization",
        "direction": {
            "name": "cross-subject",
            "label": "train subjects -> held-out subjects",
            "train_trial": None,
            "test_trial": None,
        },
        "eligible_subjects": sorted(set(train_subjects) | set(test_subjects)),
        "excluded_subjects": [],
        "split": {
            "n_train_rows": len(split["train_target_indices"]),
            "n_test_rows": len(split["test_target_indices"]),
            "train_subjects": list(train_subjects),
            "test_subjects": list(test_subjects),
        },
        "split_audit": {
            "overlapping_subjects": sorted(set(train_subjects) & set(test_subjects)),
            "has_forbidden_leakage": bool(
                set(train_subjects) & set(test_subjects)
            ),
            "inferred_from_schema_v1": True,
        },
        "selected_feature_family": list(features["block_names"]),
        "model_metrics": _metrics_payload(model_metrics),
        "model_bootstrap": _bootstrap_payload(model_bootstrap),
        "baselines": [
            {
                "name": name,
                "metrics": _metrics_payload(
                    evaluate_prediction_matrix(
                        test_targets,
                        baseline_predictions[name],
                        baseline_probabilities[name],
                    )
                ),
            }
            for name in baseline_probabilities
        ],
    }


def _build_legacy_baseline_arrays(
    train_targets: NDArray[np.int8],
    *,
    n_test_samples: int,
    config: LogisticRegressionExperimentConfig,
) -> tuple[
    dict[str, NDArray[np.float64]],
    dict[str, NDArray[np.int8]],
]:
    predictions = build_non_eeg_baselines(
        train_targets,
        n_test_samples=n_test_samples,
        threshold=config.prediction_threshold,
        random_state=config.random_state,
    )
    return (
        {item.name: item.probabilities for item in predictions},
        {item.name: item.predictions for item in predictions},
    )


def _load_baseline_arrays(
    run_dir: Path,
    *,
    evaluation: dict[str, Any],
) -> tuple[
    dict[str, NDArray[np.float64]],
    dict[str, NDArray[np.int8]],
]:
    baseline_items = evaluation.get("baselines")
    if not isinstance(baseline_items, list) or not baseline_items:
        raise ValueError("Evaluation metadata must contain baseline results")
    probabilities: dict[str, NDArray[np.float64]] = {}
    predictions: dict[str, NDArray[np.int8]] = {}
    for item in baseline_items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Evaluation baseline metadata is invalid")
        name = item["name"]
        files = item.get("array_files")
        if not isinstance(files, dict):
            raise ValueError(f"Evaluation baseline array metadata is missing: {name}")
        probabilities[name] = _load_array(
            _resolve_array_path(run_dir, files.get("probabilities")),
            np.float64,
        )
        predictions[name] = _load_array(
            _resolve_array_path(run_dir, files.get("predictions")),
            np.int8,
        )
    return probabilities, predictions


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


def _validate_evaluation_consistency(
    evaluation: dict[str, Any],
    *,
    split: dict[str, Any],
    features: dict[str, Any],
    test_targets: NDArray[np.int8],
    probabilities: NDArray[np.float64],
    predictions: NDArray[np.int8],
    test_balanced_accuracy: NDArray[np.float64],
    test_subject_ids: NDArray[np.int64],
    baseline_probabilities: dict[str, NDArray[np.float64]],
    baseline_predictions: dict[str, NDArray[np.int8]],
    config: LogisticRegressionExperimentConfig,
) -> None:
    if evaluation.get("schema_version") != 2:
        raise ValueError("Evaluation payload does not declare schema version 2")
    direction = evaluation.get("direction")
    if not isinstance(direction, dict) or direction.get("name") != split.get("direction"):
        raise ValueError("Evaluation direction does not match split metadata")
    if evaluation.get("protocol") != split.get("protocol"):
        raise ValueError("Evaluation protocol does not match split metadata")
    if evaluation.get("selected_feature_family") != features.get("block_names"):
        raise ValueError("Evaluation feature family does not match feature metadata")
    split_summary = evaluation.get("split")
    if not isinstance(split_summary, dict):
        raise ValueError("Evaluation split summary is invalid")
    expected_split_values = {
        "n_train_rows": len(split["train_target_indices"]),
        "n_test_rows": len(split["test_target_indices"]),
        "train_subjects": split["train_subjects"],
        "test_subjects": split["test_subjects"],
    }
    if any(
        split_summary.get(key) != value
        for key, value in expected_split_values.items()
    ):
        raise ValueError("Evaluation split summary does not match split metadata")
    audit = evaluation.get("split_audit")
    if not isinstance(audit, dict) or audit.get("has_forbidden_leakage") is not False:
        raise ValueError("Evaluation split audit is invalid or reports forbidden leakage")

    model_metrics = evaluate_prediction_matrix(
        test_targets,
        predictions,
        probabilities,
    )
    _assert_metrics_payload(
        evaluation.get("model_metrics"),
        expected=model_metrics,
        label="model",
    )
    if not np.allclose(
        test_balanced_accuracy,
        model_metrics.per_pixel_balanced_accuracy,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Stored test balanced accuracy differs from model predictions")
    expected_bootstrap = bootstrap_subject_mean_balanced_accuracy(
        test_targets,
        predictions,
        test_subject_ids,
        n_resamples=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    stored_bootstrap = evaluation.get("model_bootstrap")
    if not isinstance(stored_bootstrap, dict):
        raise ValueError("Evaluation model bootstrap metadata is invalid")
    expected_bootstrap_payload = _bootstrap_payload(expected_bootstrap)
    if any(
        stored_bootstrap.get(key) != value
        for key, value in expected_bootstrap_payload.items()
    ):
        raise ValueError("Stored model bootstrap differs from evaluation arrays")

    baseline_items = evaluation.get("baselines")
    if not isinstance(baseline_items, list):
        raise ValueError("Evaluation baseline metadata is invalid")
    for item in baseline_items:
        name = item["name"]
        _assert_metrics_payload(
            item.get("metrics"),
            expected=evaluate_prediction_matrix(
                test_targets,
                baseline_predictions[name],
                baseline_probabilities[name],
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


def _build_loaded_run_summary(run: LoadedEvaluationRun) -> dict[str, Any]:
    return {
        "run_dir": str(run.run_dir),
        "artifact_schema_version": run.manifest["schema_version"],
        **run.evaluation,
    }


def _combine_loaded_within_subject_runs(
    runs: tuple[LoadedEvaluationRun, LoadedEvaluationRun],
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
        raise ValueError("Combined within-subject runs must use identical configurations")
    first_keys = {tuple(key) for key in ordered[0].split["test_sample_keys"]}
    second_keys = {tuple(key) for key in ordered[1].split["test_sample_keys"]}
    if first_keys & second_keys:
        raise ValueError("Combined within-subject directions overlap test sample keys")

    test_targets = np.concatenate([run.test_targets for run in ordered], axis=0)
    probabilities = np.concatenate([run.probabilities for run in ordered], axis=0)
    predictions = np.concatenate([run.predictions for run in ordered], axis=0)
    subject_ids = np.concatenate([run.test_subject_ids for run in ordered])
    metrics = evaluate_prediction_matrix(test_targets, predictions, probabilities)
    bootstrap = bootstrap_subject_mean_balanced_accuracy(
        test_targets,
        predictions,
        subject_ids,
        n_resamples=ordered[0].config.bootstrap_iterations,
        random_state=ordered[0].config.random_state,
    )

    baseline_names = tuple(ordered[0].baseline_probabilities)
    if tuple(ordered[1].baseline_probabilities) != baseline_names:
        raise ValueError("Combined within-subject baseline order differs between runs")
    baselines = []
    for name in baseline_names:
        baseline_probabilities = np.concatenate(
            [run.baseline_probabilities[name] for run in ordered],
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
                        baseline_probabilities,
                    )
                ),
            }
        )
    return {
        "protocol": "within-subject",
        "protocol_label": ordered[0].evaluation["protocol_label"],
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


def _expected_config_hash(
    config: LogisticRegressionExperimentConfig,
    *,
    schema_version: object,
    evaluation: dict[str, Any] | None,
) -> str:
    if schema_version == 1:
        return build_experiment_config_hash(config)
    if schema_version == 2 and evaluation is not None:
        direction = evaluation.get("direction")
        if not isinstance(direction, dict):
            raise ValueError("Evaluation direction metadata is invalid")
        return build_evaluation_config_hash(
            config,
            protocol=str(evaluation.get("protocol")),
            direction=str(direction.get("name")),
        )
    raise ValueError(f"Unsupported experiment artifact schema: {schema_version!r}")


def _validate_loaded_arrays(
    *,
    probabilities: NDArray[np.float64],
    predictions: NDArray[np.int8],
    test_balanced_accuracy: NDArray[np.float64],
    train_targets: NDArray[np.int8],
    test_targets: NDArray[np.int8],
    test_subject_ids: NDArray[np.int64],
) -> None:
    n_test, n_pixels = probabilities.shape
    if probabilities.dtype != np.dtype(np.float64) or not np.isfinite(
        probabilities
    ).all():
        raise TypeError("Loaded probabilities must be a finite float64 matrix")
    if predictions.shape != (n_test, n_pixels) or predictions.dtype != np.dtype(np.int8):
        raise TypeError("Loaded predictions must be an int8 matrix matching probabilities")
    if not np.isin(predictions, (0, 1)).all():
        raise ValueError("Loaded predictions must be binary")
    if test_balanced_accuracy.shape != (n_pixels,) or test_balanced_accuracy.dtype != np.dtype(
        np.float64
    ):
        raise TypeError("Loaded test balanced accuracy must match pixels")
    if train_targets.ndim != 2 or train_targets.dtype != np.dtype(np.int8):
        raise TypeError("Loaded train targets must be a two-dimensional int8 array")
    if test_targets.shape != (n_test, n_pixels) or test_targets.dtype != np.dtype(np.int8):
        raise TypeError("Loaded test targets must match recorded predictions")
    if test_subject_ids.shape != (n_test,) or test_subject_ids.dtype != np.dtype(np.int64):
        raise TypeError("Loaded test subject IDs must match recorded test rows")


def _validate_write_inputs(
    result: PixelGridSearchResult,
    *,
    targets: PixelTargetDataset,
    split: SubjectSplit | EvaluationDirection,
) -> None:
    if not np.array_equal(result.fitted_models.training_target_indices, split.train_indices):
        raise ValueError("Fitted training rows do not match the configured outer split")
    if not np.array_equal(result.test_target_indices, split.test_indices):
        raise ValueError("Predicted test rows do not match the configured outer split")
    expected_train_keys = tuple(targets.sample_keys[int(index)] for index in split.train_indices)
    expected_test_keys = tuple(targets.sample_keys[int(index)] for index in split.test_indices)
    if result.fitted_models.training_sample_keys != expected_train_keys:
        raise ValueError("Fitted training sample keys do not match target metadata")
    if result.test_sample_keys != expected_test_keys:
        raise ValueError("Predicted test sample keys do not match target metadata")


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


def _resolve_pipeline_path(
    run_dir: Path,
    relative_path: object,
    *,
    manifest_files: set[str],
) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("Experiment pipeline filename must be a string")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe experiment pipeline filename: {relative_path!r}")
    normalized = path.as_posix()
    if len(path.parts) != 2 or path.parts[0] != "pipelines" or path.suffix != ".joblib":
        raise ValueError(f"Invalid experiment pipeline location: {relative_path!r}")
    if normalized not in manifest_files:
        raise ValueError(f"Experiment pipeline is not present in the manifest: {relative_path}")
    return run_dir / path


def _publish_run(temporary_dir: Path, *, run_dir: Path, overwrite: bool) -> None:
    if not run_dir.exists():
        os.replace(temporary_dir, run_dir)
        _fsync_directory(run_dir.parent)
        return
    if not overwrite:
        raise FileExistsError(f"Experiment run already exists and is immutable: {run_dir}")

    backup_dir = run_dir.with_name(f".{run_dir.name}-backup")
    if backup_dir.exists():
        raise FileExistsError(f"Experiment overwrite backup already exists: {backup_dir}")
    os.replace(run_dir, backup_dir)
    try:
        os.replace(temporary_dir, run_dir)
        _fsync_directory(run_dir.parent)
    except BaseException:
        os.replace(backup_dir, run_dir)
        raise
    shutil.rmtree(backup_dir)


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
