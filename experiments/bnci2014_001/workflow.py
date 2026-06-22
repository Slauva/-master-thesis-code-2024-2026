from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.bnci2014_001.baselines import (
    CSP_LDA_BASELINE_VERSION,
    fit_predict_csp_lda,
)
from experiments.bnci2014_001.config import BNCI2014001Config
from experiments.bnci2014_001.data import (
    BNCIEpochDataset,
    BNCISplit,
    create_leave_one_subject_splits,
    load_bnci_epochs,
)
from experiments.bnci2014_001.metrics import (
    ClassificationMetrics,
    evaluate_multiclass_predictions,
    summarize_fold_metrics,
)
from experiments.bnci2014_001.project_features import (
    FEATURE_LOGREG_VERSION,
    BNCIProjectFeatureMatrix,
    build_project_feature_matrix,
    fit_predict_feature_logreg,
    resolve_feature_benchmark_config,
)
from experiments.bnci2014_001.torch_pilot import (
    TORCH_FFT_PILOT_VERSION,
    TorchPilotFoldPrediction,
    fit_predict_torch_fft_pilot,
)
from features.config import build_feature_config_hash


@dataclass(frozen=True, slots=True)
class FoldBaselineResult:
    split: BNCISplit
    metrics: ClassificationMetrics
    y_true: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    probabilities: NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    config_hash: str
    class_names: tuple[str, ...]
    folds: tuple[FoldBaselineResult, ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BaselineExecutionResult:
    run_dir: Path
    config_hash: str
    reused: bool
    evaluation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FoldFeatureBenchmarkResult:
    split: BNCISplit
    metrics: ClassificationMetrics
    y_true: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    probabilities: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FeatureBenchmarkRunResult:
    config_hash: str
    class_names: tuple[str, ...]
    feature_matrix: BNCIProjectFeatureMatrix
    folds: tuple[FoldFeatureBenchmarkResult, ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FeatureBenchmarkExecutionResult:
    run_dir: Path
    config_hash: str
    reused: bool
    evaluation: dict[str, Any]
    comparison: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FoldTorchPilotResult:
    split: BNCISplit
    prediction: TorchPilotFoldPrediction
    metrics: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class TorchPilotRunResult:
    config_hash: str
    class_names: tuple[str, ...]
    folds: tuple[FoldTorchPilotResult, ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TorchPilotExecutionResult:
    run_dir: Path
    config_hash: str
    reused: bool
    evaluation: dict[str, Any]
    comparison: dict[str, Any]


def run_csp_lda_loso(
    config: BNCI2014001Config,
    *,
    dataset: BNCIEpochDataset | None = None,
) -> BaselineRunResult:
    epochs = dataset if dataset is not None else load_bnci_epochs(config)
    splits = create_leave_one_subject_splits(epochs)
    folds: list[FoldBaselineResult] = []
    for split in splits:
        prediction = fit_predict_csp_lda(
            epochs,
            split,
            baseline_config=config.baseline,
            split_config=config.split,
        )
        metrics = evaluate_multiclass_predictions(
            prediction.y_true,
            prediction.y_pred,
            class_names=epochs.class_names,
        )
        folds.append(
            FoldBaselineResult(
                split=split,
                metrics=metrics,
                y_true=prediction.y_true,
                y_pred=prediction.y_pred,
                test_indices=prediction.test_indices,
                probabilities=prediction.probabilities,
            )
        )
    return BaselineRunResult(
        config_hash=build_csp_lda_config_hash(config),
        class_names=epochs.class_names,
        folds=tuple(folds),
        summary=summarize_fold_metrics(tuple(fold.metrics for fold in folds)),
    )


def run_feature_logreg_loso(
    config: BNCI2014001Config,
    *,
    dataset: BNCIEpochDataset | None = None,
    feature_matrix: BNCIProjectFeatureMatrix | None = None,
) -> FeatureBenchmarkRunResult:
    epochs = dataset if dataset is not None else load_bnci_epochs(config)
    matrix = feature_matrix or build_project_feature_matrix(
        epochs,
        benchmark_config=config.project_features,
        source_sfreq=config.dataset.source_sfreq,
    )
    splits = create_leave_one_subject_splits(epochs)
    folds: list[FoldFeatureBenchmarkResult] = []
    for split in splits:
        prediction = fit_predict_feature_logreg(
            matrix,
            split,
            dataset=epochs,
            benchmark_config=config.project_features,
            split_config=config.split,
        )
        metrics = evaluate_multiclass_predictions(
            prediction.y_true,
            prediction.y_pred,
            class_names=epochs.class_names,
        )
        folds.append(
            FoldFeatureBenchmarkResult(
                split=split,
                metrics=metrics,
                y_true=prediction.y_true,
                y_pred=prediction.y_pred,
                test_indices=prediction.test_indices,
                probabilities=prediction.probabilities,
            )
        )
    return FeatureBenchmarkRunResult(
        config_hash=build_feature_logreg_config_hash(config),
        class_names=epochs.class_names,
        feature_matrix=matrix,
        folds=tuple(folds),
        summary=summarize_fold_metrics(tuple(fold.metrics for fold in folds)),
    )


def run_torch_fft_pilot_loso(
    config: BNCI2014001Config,
    *,
    dataset: BNCIEpochDataset | None = None,
) -> TorchPilotRunResult:
    epochs = dataset if dataset is not None else load_bnci_epochs(config)
    splits = create_leave_one_subject_splits(epochs)
    folds: list[FoldTorchPilotResult] = []
    for split in splits:
        prediction = fit_predict_torch_fft_pilot(
            epochs,
            split,
            pilot_config=config.torch_pilot,
            split_config=config.split,
            source_sfreq=config.dataset.source_sfreq,
        )
        metrics = evaluate_multiclass_predictions(
            prediction.y_true,
            prediction.y_pred,
            class_names=epochs.class_names,
        )
        folds.append(
            FoldTorchPilotResult(
                split=split,
                prediction=prediction,
                metrics=metrics,
            )
        )
    return TorchPilotRunResult(
        config_hash=build_torch_fft_pilot_config_hash(config),
        class_names=epochs.class_names,
        folds=tuple(folds),
        summary=summarize_fold_metrics(tuple(fold.metrics for fold in folds)),
    )


def execute_csp_lda_baseline(
    config: BNCI2014001Config,
    *,
    reuse_existing: bool = False,
) -> BaselineExecutionResult:
    run_dir = get_csp_lda_run_dir(config)
    if run_dir.exists():
        if not reuse_existing:
            raise FileExistsError(f"BNCI2014_001 CSP+LDA run already exists: {run_dir}")
        validate_baseline_manifest(run_dir)
        return BaselineExecutionResult(
            run_dir=run_dir,
            config_hash=build_csp_lda_config_hash(config),
            reused=True,
            evaluation=_load_json(run_dir / "evaluation.json"),
        )

    result = run_csp_lda_loso(config)
    written_dir = write_csp_lda_baseline_run(config, result)
    return BaselineExecutionResult(
        run_dir=written_dir,
        config_hash=result.config_hash,
        reused=False,
        evaluation=_load_json(written_dir / "evaluation.json"),
    )


def execute_feature_logreg_benchmark(
    config: BNCI2014001Config,
    *,
    reuse_existing: bool = False,
    require_reference: bool = True,
) -> FeatureBenchmarkExecutionResult:
    run_dir = get_feature_logreg_run_dir(config)
    reference = _load_stage4_reference(config, required=require_reference)
    if run_dir.exists():
        if not reuse_existing:
            raise FileExistsError(f"BNCI2014_001 feature Logistic Regression run already exists: {run_dir}")
        validate_baseline_manifest(run_dir)
        return FeatureBenchmarkExecutionResult(
            run_dir=run_dir,
            config_hash=build_feature_logreg_config_hash(config),
            reused=True,
            evaluation=_load_json(run_dir / "evaluation.json"),
            comparison=_load_json(run_dir / "comparison.json"),
        )

    result = run_feature_logreg_loso(config)
    written_dir = write_feature_logreg_run(config, result, reference=reference)
    return FeatureBenchmarkExecutionResult(
        run_dir=written_dir,
        config_hash=result.config_hash,
        reused=False,
        evaluation=_load_json(written_dir / "evaluation.json"),
        comparison=_load_json(written_dir / "comparison.json"),
    )


def execute_torch_fft_pilot(
    config: BNCI2014001Config,
    *,
    reuse_existing: bool = False,
) -> TorchPilotExecutionResult:
    run_dir = get_torch_fft_pilot_run_dir(config)
    references = _load_torch_references(config)
    if run_dir.exists():
        if not reuse_existing:
            raise FileExistsError(f"BNCI2014_001 Torch FFT pilot run already exists: {run_dir}")
        validate_baseline_manifest(run_dir)
        return TorchPilotExecutionResult(
            run_dir=run_dir,
            config_hash=build_torch_fft_pilot_config_hash(config),
            reused=True,
            evaluation=_load_json(run_dir / "evaluation.json"),
            comparison=_load_json(run_dir / "comparison.json"),
        )

    result = run_torch_fft_pilot_loso(config)
    written_dir = write_torch_fft_pilot_run(config, result, references=references)
    return TorchPilotExecutionResult(
        run_dir=written_dir,
        config_hash=result.config_hash,
        reused=False,
        evaluation=_load_json(written_dir / "evaluation.json"),
        comparison=_load_json(written_dir / "comparison.json"),
    )


def get_csp_lda_run_dir(config: BNCI2014001Config) -> Path:
    return Path(config.artifacts.root) / config.baseline.model_id / build_csp_lda_config_hash(config)


def get_feature_logreg_run_dir(config: BNCI2014001Config) -> Path:
    return Path(config.artifacts.root) / config.project_features.model_id / build_feature_logreg_config_hash(config)


def get_torch_fft_pilot_run_dir(config: BNCI2014001Config) -> Path:
    return Path(config.artifacts.root) / config.torch_pilot.model_id / build_torch_fft_pilot_config_hash(config)


def build_csp_lda_config_hash(config: BNCI2014001Config) -> str:
    payload = {
        "baseline_version": CSP_LDA_BASELINE_VERSION,
        "protocol": config.split.primary_protocol,
        "artifact_schema_version": config.artifacts.schema_version,
        "config": {
            "dataset": config.dataset.model_dump(mode="json"),
            "split": config.split.model_dump(mode="json"),
            "baseline": config.baseline.model_dump(mode="json"),
            "artifacts": config.artifacts.model_dump(mode="json"),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def build_feature_logreg_config_hash(config: BNCI2014001Config) -> str:
    feature_config = resolve_feature_benchmark_config(config.project_features)
    payload = {
        "benchmark_version": FEATURE_LOGREG_VERSION,
        "protocol": config.split.primary_protocol,
        "artifact_schema_version": config.artifacts.schema_version,
        "feature_config": feature_config.model_dump(mode="json"),
        "feature_config_hash": build_feature_config_hash(feature_config),
        "config": {
            "dataset": config.dataset.model_dump(mode="json"),
            "split": config.split.model_dump(mode="json"),
            "project_features": config.project_features.model_dump(mode="json"),
            "artifacts": config.artifacts.model_dump(mode="json"),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def build_torch_fft_pilot_config_hash(config: BNCI2014001Config) -> str:
    payload = {
        "pilot_version": TORCH_FFT_PILOT_VERSION,
        "protocol": config.split.primary_protocol,
        "artifact_schema_version": config.artifacts.schema_version,
        "config": {
            "dataset": config.dataset.model_dump(mode="json"),
            "split": config.split.model_dump(mode="json"),
            "torch_pilot": config.torch_pilot.model_dump(mode="json"),
            "artifacts": config.artifacts.model_dump(mode="json"),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def write_csp_lda_baseline_run(
    config: BNCI2014001Config,
    result: BaselineRunResult,
) -> Path:
    expected_hash = build_csp_lda_config_hash(config)
    if result.config_hash != expected_hash:
        raise ValueError("Baseline result hash does not match the resolved configuration")
    run_dir = get_csp_lda_run_dir(config)
    if run_dir.exists():
        raise FileExistsError(f"BNCI2014_001 CSP+LDA run already exists: {run_dir}")

    root = run_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / f".{run_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_run_payload(tmp_dir, config, result)
        tmp_dir.rename(run_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    _fsync_directory(root)
    validate_baseline_manifest(run_dir)
    return run_dir


def write_feature_logreg_run(
    config: BNCI2014001Config,
    result: FeatureBenchmarkRunResult,
    *,
    reference: dict[str, Any] | None = None,
) -> Path:
    expected_hash = build_feature_logreg_config_hash(config)
    if result.config_hash != expected_hash:
        raise ValueError("Feature benchmark result hash does not match the resolved configuration")
    run_dir = get_feature_logreg_run_dir(config)
    if run_dir.exists():
        raise FileExistsError(f"BNCI2014_001 feature Logistic Regression run already exists: {run_dir}")

    root = run_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / f".{run_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_feature_run_payload(tmp_dir, config, result, reference=reference)
        tmp_dir.rename(run_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    _fsync_directory(root)
    validate_baseline_manifest(run_dir)
    return run_dir


def write_torch_fft_pilot_run(
    config: BNCI2014001Config,
    result: TorchPilotRunResult,
    *,
    references: dict[str, dict[str, Any]],
) -> Path:
    expected_hash = build_torch_fft_pilot_config_hash(config)
    if result.config_hash != expected_hash:
        raise ValueError("Torch pilot result hash does not match the resolved configuration")
    run_dir = get_torch_fft_pilot_run_dir(config)
    if run_dir.exists():
        raise FileExistsError(f"BNCI2014_001 Torch FFT pilot run already exists: {run_dir}")

    root = run_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / f".{run_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_torch_run_payload(tmp_dir, config, result, references=references)
        tmp_dir.rename(run_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    _fsync_directory(root)
    validate_baseline_manifest(run_dir)
    return run_dir


def validate_baseline_manifest(run_dir: Path) -> None:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BNCI2014_001 manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("file_count") != len(files):
        raise ValueError("BNCI2014_001 file inventory is invalid")
    expected_files = set(files) | {"manifest.json"}
    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in Path(run_dir).rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(f"BNCI2014_001 inventory mismatch; missing={missing}, unexpected={unexpected}")
    for relative_path, metadata in files.items():
        path = Path(run_dir) / relative_path
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid inventory metadata for {relative_path}")
        if path.stat().st_size != metadata.get("size"):
            raise ValueError(f"BNCI2014_001 file size mismatch: {relative_path}")
        if _sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"BNCI2014_001 file hash mismatch: {relative_path}")


def _write_run_payload(
    run_dir: Path,
    config: BNCI2014001Config,
    result: BaselineRunResult,
) -> None:
    (run_dir / "arrays").mkdir(parents=True)
    _write_json(run_dir / "config.json", config.model_dump(mode="json"))
    _write_json(run_dir / "environment.json", _build_environment_payload())
    _write_json(run_dir / "split.json", _build_split_payload(result))
    _write_json(run_dir / "evaluation.json", _build_evaluation_payload(result))
    arrays_dir = run_dir / "arrays"
    _write_array(arrays_dir / "test_indices.npy", np.concatenate([fold.test_indices for fold in result.folds]))
    _write_array(arrays_dir / "y_true.npy", np.concatenate([fold.y_true for fold in result.folds]))
    _write_array(arrays_dir / "y_pred.npy", np.concatenate([fold.y_pred for fold in result.folds]))
    _write_array(arrays_dir / "fold_index.npy", _fold_index_array(result.folds))
    probabilities = [fold.probabilities for fold in result.folds if fold.probabilities is not None]
    if len(probabilities) == len(result.folds):
        _write_array(arrays_dir / "probabilities.npy", np.concatenate(probabilities))

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
            "config_hash": result.config_hash,
            "baseline_version": CSP_LDA_BASELINE_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "writer": "experiments.bnci2014_001.workflow.write_csp_lda_baseline_run",
            "file_count": len(files),
            "files": files,
        },
    )
    _fsync_directory(run_dir)


def _write_feature_run_payload(
    run_dir: Path,
    config: BNCI2014001Config,
    result: FeatureBenchmarkRunResult,
    *,
    reference: dict[str, Any] | None,
) -> None:
    (run_dir / "arrays").mkdir(parents=True)
    _write_json(run_dir / "config.json", config.model_dump(mode="json"))
    _write_json(run_dir / "environment.json", _build_environment_payload())
    _write_json(run_dir / "split.json", _build_split_payload(result))
    _write_json(run_dir / "features.json", _build_feature_payload(result))
    _write_json(run_dir / "comparison.json", _build_feature_comparison_payload(result, reference=reference))
    _write_json(run_dir / "evaluation.json", _build_feature_evaluation_payload(result))
    arrays_dir = run_dir / "arrays"
    _write_array(arrays_dir / "features.npy", result.feature_matrix.X)
    _write_array(arrays_dir / "test_indices.npy", np.concatenate([fold.test_indices for fold in result.folds]))
    _write_array(arrays_dir / "y_true.npy", np.concatenate([fold.y_true for fold in result.folds]))
    _write_array(arrays_dir / "y_pred.npy", np.concatenate([fold.y_pred for fold in result.folds]))
    _write_array(arrays_dir / "fold_index.npy", _feature_fold_index_array(result.folds))
    _write_array(arrays_dir / "probabilities.npy", np.concatenate([fold.probabilities for fold in result.folds]))

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
            "config_hash": result.config_hash,
            "benchmark_version": FEATURE_LOGREG_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "writer": "experiments.bnci2014_001.workflow.write_feature_logreg_run",
            "file_count": len(files),
            "files": files,
        },
    )
    _fsync_directory(run_dir)


def _write_torch_run_payload(
    run_dir: Path,
    config: BNCI2014001Config,
    result: TorchPilotRunResult,
    *,
    references: dict[str, dict[str, Any]],
) -> None:
    (run_dir / "arrays").mkdir(parents=True)
    _write_json(run_dir / "config.json", config.model_dump(mode="json"))
    _write_json(run_dir / "environment.json", _build_environment_payload())
    _write_json(run_dir / "split.json", _build_torch_split_payload(result))
    _write_json(run_dir / "training.json", _build_torch_training_payload(result))
    _write_json(run_dir / "comparison.json", _build_torch_comparison_payload(result, references=references))
    _write_json(run_dir / "evaluation.json", _build_torch_evaluation_payload(result))
    arrays_dir = run_dir / "arrays"
    _write_array(
        arrays_dir / "test_indices.npy",
        np.concatenate([fold.prediction.test_indices for fold in result.folds]),
    )
    _write_array(arrays_dir / "y_true.npy", np.concatenate([fold.prediction.y_true for fold in result.folds]))
    _write_array(arrays_dir / "y_pred.npy", np.concatenate([fold.prediction.y_pred for fold in result.folds]))
    _write_array(arrays_dir / "fold_index.npy", _torch_fold_index_array(result.folds))
    _write_array(
        arrays_dir / "probabilities.npy",
        np.concatenate([fold.prediction.probabilities for fold in result.folds]),
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
            "config_hash": result.config_hash,
            "pilot_version": TORCH_FFT_PILOT_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "writer": "experiments.bnci2014_001.workflow.write_torch_fft_pilot_run",
            "file_count": len(files),
            "files": files,
        },
    )
    _fsync_directory(run_dir)


def _build_split_payload(result: BaselineRunResult) -> dict[str, Any]:
    folds = []
    for fold in result.folds:
        split = fold.split
        folds.append(
            {
                "name": split.name,
                "train_subjects": list(split.train_subjects),
                "test_subjects": list(split.test_subjects),
                "n_train": int(split.train_indices.size),
                "n_test": int(split.test_indices.size),
                "train_indices_sha256": _sha256_array(split.train_indices),
                "test_indices_sha256": _sha256_array(split.test_indices),
            }
        )
    return {
        "protocol": "leave-one-subject-out",
        "n_folds": len(result.folds),
        "folds": folds,
    }


def _build_feature_payload(result: FeatureBenchmarkRunResult) -> dict[str, Any]:
    return {
        "feature_config_hash": result.feature_matrix.feature_config_hash,
        "feature_config": result.feature_matrix.feature_config.model_dump(mode="json"),
        "feature_shape": list(result.feature_matrix.X.shape),
        "feature_dtype": str(result.feature_matrix.X.dtype),
        "feature_names": list(result.feature_matrix.feature_names),
        "sample_keys_sha256": _sha256_sample_keys(result.feature_matrix.sample_keys),
        "target_sha256": _sha256_array(result.feature_matrix.y),
    }


def _build_torch_split_payload(result: TorchPilotRunResult) -> dict[str, Any]:
    folds = []
    for fold in result.folds:
        split = fold.split
        prediction = fold.prediction
        folds.append(
            {
                "name": split.name,
                "train_subjects": list(split.train_subjects),
                "test_subjects": list(split.test_subjects),
                "validation_subject": prediction.validation_subject,
                "n_train_fit": int(prediction.train_fit_indices.size),
                "n_validation": int(prediction.validation_indices.size),
                "n_test": int(prediction.test_indices.size),
                "train_fit_indices_sha256": _sha256_array(prediction.train_fit_indices),
                "validation_indices_sha256": _sha256_array(prediction.validation_indices),
                "train_indices_sha256": _sha256_array(split.train_indices),
                "test_indices_sha256": _sha256_array(prediction.test_indices),
            }
        )
    return {
        "protocol": "leave-one-subject-out",
        "validation_strategy": "lowest-train-subject",
        "n_folds": len(result.folds),
        "folds": folds,
    }


def _build_torch_training_payload(result: TorchPilotRunResult) -> dict[str, Any]:
    return {
        "model_id": "fft-cnn-pilot",
        "input": {
            "method": "fft",
            "shape": [1, 22, 39],
            "transform": "log1p_nonnegative_power",
            "normalization": "train_fit_tensor_mean_std_per_outer_fold",
        },
        "folds": [
            {
                "name": fold.split.name,
                "validation_subject": fold.prediction.validation_subject,
                "best_epoch": fold.prediction.best_epoch,
                "epochs_ran": len(fold.prediction.history),
                "history": list(fold.prediction.history),
            }
            for fold in result.folds
        ],
    }


def _build_feature_comparison_payload(
    result: FeatureBenchmarkRunResult,
    *,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "comparison_reference": "csp-lda",
        "stage4_reference_available": reference is not None,
        "split_alignment": "not_checked",
    }
    if reference is None:
        return payload

    split_payload = _build_split_payload(result)
    reference_split = reference["split"]
    _require_matching_split_payload(split_payload, reference_split)
    stage4_score = reference["evaluation"]["summary"]["balanced_accuracy_mean"]
    stage5_score = result.summary["balanced_accuracy_mean"]
    payload.update(
        {
            "stage4_reference_run_dir": reference["run_dir"],
            "stage4_reference_config_hash": reference["manifest"]["config_hash"],
            "stage4_reference_balanced_accuracy_mean": stage4_score,
            "stage5_balanced_accuracy_mean": stage5_score,
            "balanced_accuracy_delta_vs_stage4": stage5_score - stage4_score,
            "split_alignment": "matched_by_fold_name_and_index_hash",
        }
    )
    return payload


def _build_torch_comparison_payload(
    result: TorchPilotRunResult,
    *,
    references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "comparison_references": sorted(references),
        "stage6_balanced_accuracy_mean": result.summary["balanced_accuracy_mean"],
    }
    torch_split = _torch_reference_split_payload(result)
    for name, reference in references.items():
        _require_matching_split_payload(torch_split, reference["split"])
        reference_score = reference["evaluation"]["summary"]["balanced_accuracy_mean"]
        payload[f"{name}_run_dir"] = reference["run_dir"]
        payload[f"{name}_config_hash"] = reference["manifest"]["config_hash"]
        payload[f"{name}_balanced_accuracy_mean"] = reference_score
        payload[f"balanced_accuracy_delta_vs_{name}"] = result.summary["balanced_accuracy_mean"] - reference_score
    payload["split_alignment"] = "matched_by_fold_name_and_index_hash"
    return payload


def _build_evaluation_payload(result: BaselineRunResult) -> dict[str, Any]:
    return {
        "model_id": "csp-lda",
        "class_names": list(result.class_names),
        "summary": result.summary,
        "folds": [
            {
                "name": fold.split.name,
                "test_subjects": list(fold.split.test_subjects),
                "metrics": fold.metrics.to_payload(),
            }
            for fold in result.folds
        ],
    }


def _build_feature_evaluation_payload(result: FeatureBenchmarkRunResult) -> dict[str, Any]:
    return {
        "model_id": "feature-logreg",
        "class_names": list(result.class_names),
        "summary": result.summary,
        "folds": [
            {
                "name": fold.split.name,
                "test_subjects": list(fold.split.test_subjects),
                "metrics": fold.metrics.to_payload(),
            }
            for fold in result.folds
        ],
    }


def _build_torch_evaluation_payload(result: TorchPilotRunResult) -> dict[str, Any]:
    return {
        "model_id": "fft-cnn-pilot",
        "class_names": list(result.class_names),
        "summary": result.summary,
        "folds": [
            {
                "name": fold.split.name,
                "test_subjects": list(fold.split.test_subjects),
                "validation_subject": fold.prediction.validation_subject,
                "best_epoch": fold.prediction.best_epoch,
                "metrics": fold.metrics.to_payload(),
            }
            for fold in result.folds
        ],
    }


def _fold_index_array(folds: tuple[FoldBaselineResult, ...]) -> NDArray[np.int64]:
    values = [
        np.full(fold.test_indices.shape, fold_index, dtype=np.int64)
        for fold_index, fold in enumerate(folds)
    ]
    return np.concatenate(values)


def _torch_fold_index_array(folds: tuple[FoldTorchPilotResult, ...]) -> NDArray[np.int64]:
    values = [
        np.full(fold.prediction.test_indices.shape, fold_index, dtype=np.int64)
        for fold_index, fold in enumerate(folds)
    ]
    return np.concatenate(values)


def _torch_reference_split_payload(result: TorchPilotRunResult) -> dict[str, Any]:
    folds = []
    for fold in result.folds:
        split = fold.split
        folds.append(
            {
                "name": split.name,
                "train_indices_sha256": _sha256_array(split.train_indices),
                "test_indices_sha256": _sha256_array(fold.prediction.test_indices),
            }
        )
    return {
        "protocol": "leave-one-subject-out",
        "n_folds": len(result.folds),
        "folds": folds,
    }


def _feature_fold_index_array(folds: tuple[FoldFeatureBenchmarkResult, ...]) -> NDArray[np.int64]:
    values = [
        np.full(fold.test_indices.shape, fold_index, dtype=np.int64)
        for fold_index, fold in enumerate(folds)
    ]
    return np.concatenate(values)


def _load_stage4_reference(config: BNCI2014001Config, *, required: bool) -> dict[str, Any] | None:
    run_dir = get_csp_lda_run_dir(config)
    if not run_dir.exists():
        if required:
            raise FileNotFoundError(f"Stage 4 CSP+LDA reference artifact does not exist: {run_dir}")
        return None
    validate_baseline_manifest(run_dir)
    return {
        "run_dir": run_dir.as_posix(),
        "manifest": _load_json(run_dir / "manifest.json"),
        "split": _load_json(run_dir / "split.json"),
        "evaluation": _load_json(run_dir / "evaluation.json"),
    }


def _load_torch_references(config: BNCI2014001Config) -> dict[str, dict[str, Any]]:
    references = {
        "csp_lda": get_csp_lda_run_dir(config),
        "feature_logreg": get_feature_logreg_run_dir(config),
    }
    loaded: dict[str, dict[str, Any]] = {}
    for name, run_dir in references.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Stage 6 reference artifact {name!r} does not exist: {run_dir}")
        validate_baseline_manifest(run_dir)
        loaded[name] = {
            "run_dir": run_dir.as_posix(),
            "manifest": _load_json(run_dir / "manifest.json"),
            "split": _load_json(run_dir / "split.json"),
            "evaluation": _load_json(run_dir / "evaluation.json"),
        }
    return loaded


def _require_matching_split_payload(payload: dict[str, Any], reference: dict[str, Any]) -> None:
    if payload["protocol"] != reference.get("protocol") or payload["n_folds"] != reference.get("n_folds"):
        raise ValueError("Feature benchmark split protocol does not match the Stage 4 reference")
    for fold, reference_fold in zip(payload["folds"], reference["folds"], strict=True):
        for key in ("name", "train_indices_sha256", "test_indices_sha256"):
            if fold[key] != reference_fold.get(key):
                raise ValueError(f"Feature benchmark fold {fold['name']!r} does not match Stage 4 by {key}")


def _build_environment_payload() -> dict[str, Any]:
    package_names = (
        "mne",
        "moabb",
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


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return commit, bool(status.strip())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _fsync_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"BNCI2014_001 JSON file must contain an object: {path}")
    return payload


def _write_array(path: Path, array: NDArray[Any]) -> None:
    with path.open("wb") as file:
        np.save(file, array, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _sha256_sample_keys(sample_keys: tuple[Any, ...]) -> str:
    canonical = json.dumps(sample_keys, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
