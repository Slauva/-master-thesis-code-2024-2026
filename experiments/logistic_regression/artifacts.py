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

from experiments.logistic_regression.config import (
    LogisticRegressionExperimentConfig,
    build_experiment_config_hash,
)
from experiments.logistic_regression.schemas import (
    AlignedFeaturePartition,
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


def write_experiment_run(
    result: PixelGridSearchResult,
    *,
    targets: PixelTargetDataset,
    split: SubjectSplit,
    config: LogisticRegressionExperimentConfig,
    feature_config_hash: str,
    screening: FeatureScreeningResult | None = None,
) -> Path:
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
    expected_hash = build_experiment_config_hash(config)
    if manifest.get("config_hash") != expected_hash or resolved_dir.name != expected_hash:
        raise ValueError("Experiment config hash does not match the manifest and run directory")
    if manifest.get("schema_version") != config.artifacts.schema_version:
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
    split: SubjectSplit,
    config: LogisticRegressionExperimentConfig,
    config_hash: str,
    feature_config_hash: str,
    screening: FeatureScreeningResult | None,
) -> None:
    (run_dir / "arrays").mkdir()
    (run_dir / "pipelines").mkdir()

    _write_json(run_dir / "config.json", config.model_dump(mode="json"))
    _write_json(run_dir / "environment.json", _build_environment_payload())
    _write_json(run_dir / "split.json", _build_split_payload(result, targets=targets, split=split))
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
            "writer": "experiments.logistic_regression.artifacts.write_experiment_run",
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
    split: SubjectSplit,
) -> dict[str, Any]:
    cross_validation = result.fitted_models.cross_validation
    return {
        "random_state": split.random_state,
        "test_size": split.test_size,
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


def _validate_write_inputs(
    result: PixelGridSearchResult,
    *,
    targets: PixelTargetDataset,
    split: SubjectSplit,
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
