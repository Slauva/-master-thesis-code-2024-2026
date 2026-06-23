from __future__ import annotations

import hashlib
import importlib.metadata
import json
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

from experiments.bnci2014_009.baselines import (
    CLASSICAL_BASELINE_VERSION,
    ClassicalFoldPrediction,
    fit_predict_classical_variant,
)
from experiments.bnci2014_009.config import BNCI2014009Config
from experiments.bnci2014_009.data import (
    BNCI009EpochDataset,
    BNCI009Split,
    create_leave_one_subject_splits,
    load_bnci009_epochs,
)
from experiments.bnci2014_009.features import (
    ERP_FEATURE_VERSION,
    XDAWN_RIEMANNIAN_FEATURE_VERSION,
    build_erp_feature_matrix,
)
from experiments.bnci2014_009.metrics import (
    BinaryP300Metrics,
    evaluate_binary_p300_predictions,
    summarize_fold_metrics,
)


@dataclass(frozen=True, slots=True)
class FoldClassicalResult:
    split: BNCI009Split
    prediction: ClassicalFoldPrediction
    metrics: BinaryP300Metrics


@dataclass(frozen=True, slots=True)
class ClassicalVariantResult:
    model_id: str
    folds: tuple[FoldClassicalResult, ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ClassicalSweepResult:
    config_hash: str
    class_names: tuple[str, ...]
    variants: tuple[ClassicalVariantResult, ...]


@dataclass(frozen=True, slots=True)
class ClassicalSweepExecutionResult:
    run_dir: Path
    config_hash: str
    reused: bool
    evaluation: dict[str, Any]


def execute_classical_benchmark(
    config: BNCI2014009Config,
    *,
    reuse_existing: bool = False,
) -> ClassicalSweepExecutionResult:
    run_dir = get_classical_run_dir(config)
    if run_dir.exists():
        if not reuse_existing:
            raise FileExistsError(f"BNCI2014_009 classical benchmark already exists: {run_dir}")
        validate_classical_manifest(run_dir)
        return ClassicalSweepExecutionResult(
            run_dir=run_dir,
            config_hash=build_classical_config_hash(config),
            reused=True,
            evaluation=_load_json(run_dir / "evaluation.json"),
        )
    result = run_classical_benchmark(config)
    written_dir = write_classical_benchmark(config, result)
    return ClassicalSweepExecutionResult(
        run_dir=written_dir,
        config_hash=result.config_hash,
        reused=False,
        evaluation=_load_json(written_dir / "evaluation.json"),
    )


def run_classical_benchmark(
    config: BNCI2014009Config,
    *,
    dataset: BNCI009EpochDataset | None = None,
) -> ClassicalSweepResult:
    epochs = dataset if dataset is not None else load_bnci009_epochs(config)
    erp_features = build_erp_feature_matrix(
        epochs,
        source_sfreq=config.dataset.source_sfreq,
        waveform_stride=config.classical.erp_waveform_stride,
    )
    splits = create_leave_one_subject_splits(epochs)
    variants: list[ClassicalVariantResult] = []
    for model_id in config.classical.variants:
        print(f"[bnci009 classical] running {model_id}", flush=True)
        folds: list[FoldClassicalResult] = []
        for split in splits:
            prediction = fit_predict_classical_variant(
                model_id,
                epochs,
                erp_features,
                split,
                classical_config=config.classical,
                split_config=config.split,
            )
            metrics = evaluate_binary_p300_predictions(
                prediction.y_true,
                prediction.y_pred,
                target_score=prediction.target_score,
                class_names=epochs.class_names,
            )
            folds.append(FoldClassicalResult(split=split, prediction=prediction, metrics=metrics))
        variants.append(
            ClassicalVariantResult(
                model_id=model_id,
                folds=tuple(folds),
                summary=summarize_fold_metrics(tuple(fold.metrics for fold in folds)),
            )
        )
    return ClassicalSweepResult(
        config_hash=build_classical_config_hash(config),
        class_names=epochs.class_names,
        variants=tuple(variants),
    )


def get_classical_run_dir(config: BNCI2014009Config) -> Path:
    return config.artifacts.root / config.classical.model_id / build_classical_config_hash(config)


def build_classical_config_hash(config: BNCI2014009Config) -> str:
    payload = {
        "dataset": config.dataset.model_dump(mode="json"),
        "split": config.split.model_dump(mode="json"),
        "classical": config.classical.model_dump(mode="json"),
        "artifacts": config.artifacts.model_dump(mode="json"),
        "baseline_version": CLASSICAL_BASELINE_VERSION,
        "erp_feature_version": ERP_FEATURE_VERSION,
        "xdawn_riemannian_feature_version": XDAWN_RIEMANNIAN_FEATURE_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_classical_benchmark(
    config: BNCI2014009Config,
    result: ClassicalSweepResult,
) -> Path:
    run_dir = get_classical_run_dir(config)
    if run_dir.exists():
        raise FileExistsError(f"BNCI2014_009 classical benchmark already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir.parent / f".{run_dir.name}.{uuid.uuid4().hex}.tmp"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(tmp_dir / "config.json", _config_payload(config, result.config_hash))
        _write_json(tmp_dir / "environment.json", _environment_payload())
        _write_json(tmp_dir / "split.json", _split_payload(result))
        _write_json(tmp_dir / "evaluation.json", _evaluation_payload(result))
        _write_json(tmp_dir / "predictions.json", _predictions_payload(result))
        _write_json(tmp_dir / "manifest.json", _manifest_payload(tmp_dir))
        tmp_dir.replace(run_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return run_dir


def validate_classical_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ValueError("Manifest does not contain a file inventory")
    actual = _file_inventory(run_dir)
    actual.pop("manifest.json", None)
    if actual != expected:
        raise ValueError("BNCI2014_009 classical manifest inventory mismatch")


def _config_payload(config: BNCI2014009Config, config_hash: str) -> dict[str, Any]:
    return {
        "config_hash": config_hash,
        "baseline_version": CLASSICAL_BASELINE_VERSION,
        "erp_feature_version": ERP_FEATURE_VERSION,
        "xdawn_riemannian_feature_version": XDAWN_RIEMANNIAN_FEATURE_VERSION,
        "config": config.model_dump(mode="json"),
    }


def _split_payload(result: ClassicalSweepResult) -> dict[str, Any]:
    reference = result.variants[0]
    return {
        "class_names": list(result.class_names),
        "splits": [
            {
                "name": fold.split.name,
                "train_subjects": list(fold.split.train_subjects),
                "test_subjects": list(fold.split.test_subjects),
                "train_indices_sha256": _array_hash(fold.split.train_indices),
                "test_indices_sha256": _array_hash(fold.split.test_indices),
                "n_train": int(fold.split.train_indices.size),
                "n_test": int(fold.split.test_indices.size),
            }
            for fold in reference.folds
        ],
    }


def _evaluation_payload(result: ClassicalSweepResult) -> dict[str, Any]:
    return {
        "config_hash": result.config_hash,
        "class_names": list(result.class_names),
        "variants": [
            {
                "model_id": variant.model_id,
                "summary": variant.summary,
                "folds": [
                    {
                        "split": fold.split.name,
                        "test_subjects": list(fold.split.test_subjects),
                        "metrics": fold.metrics.to_payload(),
                    }
                    for fold in variant.folds
                ],
            }
            for variant in result.variants
        ],
    }


def _predictions_payload(result: ClassicalSweepResult) -> dict[str, Any]:
    return {
        "config_hash": result.config_hash,
        "variants": [
            {
                "model_id": variant.model_id,
                "folds": [
                    {
                        "split": fold.split.name,
                        "test_indices": fold.prediction.test_indices.astype(int).tolist(),
                        "y_true": fold.prediction.y_true.astype(int).tolist(),
                        "y_pred": fold.prediction.y_pred.astype(int).tolist(),
                        "target_score": (
                            fold.prediction.target_score.astype(float).tolist()
                            if fold.prediction.target_score is not None
                            else None
                        ),
                    }
                    for fold in variant.folds
                ],
            }
            for variant in result.variants
        ],
    }


def _environment_payload() -> dict[str, Any]:
    packages = {}
    for name in ("mne", "moabb", "numpy", "pandas", "pyriemann", "scikit-learn", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_revision": _git_revision(),
    }


def _manifest_payload(run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "files": _file_inventory(run_dir),
    }


def _file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        inventory[relative] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return inventory


def _array_hash(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
