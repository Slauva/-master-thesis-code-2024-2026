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
from numpy.typing import NDArray

from experiments.bnci2014_009.config import (
    BNCI009SpectralTorchArchitecture,
    BNCI009SpectralTorchBenchmarkConfig,
    BNCI009SplitConfig,
    BNCI2014009Config,
)
from experiments.bnci2014_009.data import (
    BNCI009EpochDataset,
    create_leave_one_subject_splits,
    load_bnci009_epochs,
)
from experiments.bnci2014_009.metrics import evaluate_binary_p300_predictions, summarize_fold_metrics
from experiments.bnci2014_009.torch_raw import (
    RawTorchFoldResult,
    apply_tensor_standardizer,
    fit_class_weights,
    fit_tensor_standardizer,
    predict_raw_torch_probabilities,
    resolve_torch_device,
    select_validation_indices,
    set_torch_seed,
    train_raw_torch_model,
)
from experiments.random_imagery_torch.models import SpectralModelShape, build_spectral_model
from preprocessors.config import FFTConfig, load_preprocessing_config
from preprocessors.fft import compute_fft_psd

SPECTRAL_TORCH_BENCHMARK_VERSION = 1


@dataclass(frozen=True, slots=True)
class BNCI009SpectralTensorDataset:
    method: str
    X: NDArray[np.float32]
    y: NDArray[np.int64]
    sample_keys: tuple[tuple[int, str, str, int], ...]
    input_shape: SpectralModelShape
    frequencies: NDArray[np.float32]
    transform: str

    def __post_init__(self) -> None:
        if self.method != "fft":
            raise ValueError("BNCI2014_009 Stage 6 currently supports only FFT tensors")
        if self.X.ndim != 4:
            raise ValueError("Spectral tensors must have shape (epoch, plane, channel, frequency)")
        if self.X.shape[0] != self.y.shape[0] or self.X.shape[0] != len(self.sample_keys):
            raise ValueError("Spectral tensors, targets, and sample keys must align")
        if tuple(self.X.shape[1:]) != self.input_shape.tensor_shape:
            raise ValueError("Spectral tensor shape does not match the model input shape")
        if self.X.shape[-1] != self.frequencies.shape[0]:
            raise ValueError("Spectral tensor width must match the frequency grid")
        if not np.isfinite(self.X).all() or not np.isfinite(self.frequencies).all():
            raise ValueError("Spectral tensors and frequencies must be finite")


@dataclass(frozen=True, slots=True)
class SpectralTorchVariantResult:
    architecture: BNCI009SpectralTorchArchitecture
    method: str
    model_id: str
    input_shape: SpectralModelShape
    folds: tuple[RawTorchFoldResult, ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SpectralTorchBenchmarkResult:
    config_hash: str
    class_names: tuple[str, ...]
    preprocessing: dict[str, Any]
    variants: tuple[SpectralTorchVariantResult, ...]


@dataclass(frozen=True, slots=True)
class SpectralTorchExecutionResult:
    run_dir: Path
    config_hash: str
    reused: bool
    evaluation: dict[str, Any]


def execute_spectral_torch_benchmark(
    config: BNCI2014009Config,
    *,
    reuse_existing: bool = False,
) -> SpectralTorchExecutionResult:
    run_dir = get_spectral_torch_run_dir(config)
    if run_dir.exists():
        if not reuse_existing:
            raise FileExistsError(f"BNCI2014_009 spectral Torch benchmark already exists: {run_dir}")
        validate_spectral_torch_manifest(run_dir)
        return SpectralTorchExecutionResult(
            run_dir=run_dir,
            config_hash=build_spectral_torch_config_hash(config),
            reused=True,
            evaluation=_load_json(run_dir / "evaluation.json"),
        )
    result = run_spectral_torch_benchmark(config)
    written_dir = write_spectral_torch_benchmark(config, result)
    return SpectralTorchExecutionResult(
        run_dir=written_dir,
        config_hash=result.config_hash,
        reused=False,
        evaluation=_load_json(written_dir / "evaluation.json"),
    )


def run_spectral_torch_benchmark(
    config: BNCI2014009Config,
    *,
    dataset: BNCI009EpochDataset | None = None,
) -> SpectralTorchBenchmarkResult:
    epochs = dataset if dataset is not None else load_bnci009_epochs(config)
    preprocessing_config = load_preprocessing_config("fft")
    spectral_by_method = {
        method: materialize_fft_tensor_dataset(
            epochs,
            source_sfreq=config.dataset.source_sfreq,
            preprocessing_config=preprocessing_config,
        )
        for method in config.spectral_torch.spectral_methods
    }
    variants: list[SpectralTorchVariantResult] = []
    for method in config.spectral_torch.spectral_methods:
        spectral = spectral_by_method[method]
        for architecture in config.spectral_torch.architectures:
            print(f"[bnci009 spectral-torch] training {architecture}-{method}", flush=True)
            variants.append(
                run_spectral_torch_variant(
                    epochs,
                    spectral,
                    architecture=architecture,
                    config=config.spectral_torch,
                    split_config=config.split,
                )
            )
    return SpectralTorchBenchmarkResult(
        config_hash=build_spectral_torch_config_hash(config),
        class_names=epochs.class_names,
        preprocessing=_preprocessing_payload(preprocessing_config),
        variants=tuple(variants),
    )


def materialize_fft_tensor_dataset(
    dataset: BNCI009EpochDataset,
    *,
    source_sfreq: float,
    preprocessing_config: FFTConfig | None = None,
) -> BNCI009SpectralTensorDataset:
    config = preprocessing_config or load_preprocessing_config("fft")
    if not isinstance(config, FFTConfig):
        raise TypeError("FFT tensor materialization requires an FFTConfig")
    powers: list[NDArray[np.float32]] = []
    frequencies: NDArray[np.float32] | None = None
    for epoch in dataset.X:
        transformed = compute_fft_psd(epoch, source_sfreq=source_sfreq, config=config)
        power = np.log1p(np.maximum(np.asarray(transformed.eeg_power, dtype=np.float32), np.float32(0.0)))
        powers.append(power[np.newaxis, :, :])
        current_frequencies = np.asarray(transformed.frequencies, dtype=np.float32)
        if frequencies is None:
            frequencies = current_frequencies
        elif not np.array_equal(frequencies, current_frequencies):
            raise ValueError("FFT tensor frequency grids are inconsistent")
    if frequencies is None:
        raise ValueError("Cannot materialize an empty FFT tensor dataset")
    X = np.stack(powers).astype(np.float32, copy=False)
    y = np.asarray(dataset.y, dtype=np.int64)
    input_shape = SpectralModelShape(
        input_planes=int(X.shape[1]),
        electrodes=int(X.shape[2]),
        width=int(X.shape[3]),
    )
    X.setflags(write=False)
    y.setflags(write=False)
    frequencies.setflags(write=False)
    return BNCI009SpectralTensorDataset(
        method="fft",
        X=X,
        y=y,
        sample_keys=dataset.sample_keys,
        input_shape=input_shape,
        frequencies=frequencies,
        transform="log1p_nonnegative_fft_psd",
    )


def run_spectral_torch_variant(
    dataset: BNCI009EpochDataset,
    spectral: BNCI009SpectralTensorDataset,
    *,
    architecture: BNCI009SpectralTorchArchitecture,
    config: BNCI009SpectralTorchBenchmarkConfig,
    split_config: BNCI009SplitConfig,
) -> SpectralTorchVariantResult:
    if spectral.sample_keys != dataset.sample_keys:
        raise ValueError("Spectral tensor sample keys do not match BNCI2014_009 epoch order")
    folds = tuple(
        _fit_predict_spectral_torch_fold(
            dataset,
            spectral,
            split,
            architecture=architecture,
            config=config,
            split_config=split_config,
        )
        for split in create_leave_one_subject_splits(dataset)
    )
    return SpectralTorchVariantResult(
        architecture=architecture,
        method=spectral.method,
        model_id=f"{architecture}-{spectral.method}-spectral",
        input_shape=spectral.input_shape,
        folds=folds,
        summary=summarize_fold_metrics(tuple(fold.metrics for fold in folds)),
    )


def get_spectral_torch_run_dir(config: BNCI2014009Config) -> Path:
    return config.artifacts.root / config.spectral_torch.model_id / build_spectral_torch_config_hash(config)


def build_spectral_torch_config_hash(config: BNCI2014009Config) -> str:
    preprocessing_config = load_preprocessing_config("fft")
    payload = {
        "benchmark_version": SPECTRAL_TORCH_BENCHMARK_VERSION,
        "dataset": config.dataset.model_dump(mode="json"),
        "split": config.split.model_dump(mode="json"),
        "spectral_torch": config.spectral_torch.model_dump(mode="json"),
        "artifacts": config.artifacts.model_dump(mode="json"),
        "preprocessing": _preprocessing_payload(preprocessing_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_spectral_torch_benchmark(
    config: BNCI2014009Config,
    result: SpectralTorchBenchmarkResult,
) -> Path:
    if result.config_hash != build_spectral_torch_config_hash(config):
        raise ValueError("Spectral Torch result hash does not match the resolved configuration")
    run_dir = get_spectral_torch_run_dir(config)
    if run_dir.exists():
        raise FileExistsError(f"BNCI2014_009 spectral Torch benchmark already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir.parent / f".{run_dir.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_spectral_torch_payload(tmp_dir, config, result)
        tmp_dir.replace(run_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    validate_spectral_torch_manifest(run_dir)
    return run_dir


def validate_spectral_torch_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing spectral Torch manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("file_count") != len(files):
        raise ValueError("Spectral Torch manifest inventory is invalid")
    expected_files = set(files) | {"manifest.json"}
    actual_files = {path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(f"Spectral Torch inventory mismatch; missing={missing}, unexpected={unexpected}")
    for relative_path, metadata in files.items():
        path = run_dir / relative_path
        if path.stat().st_size != metadata.get("bytes"):
            raise ValueError(f"Spectral Torch file size mismatch: {relative_path}")
        if _sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Spectral Torch file hash mismatch: {relative_path}")


def _fit_predict_spectral_torch_fold(
    dataset: BNCI009EpochDataset,
    spectral: BNCI009SpectralTensorDataset,
    split: Any,
    *,
    architecture: BNCI009SpectralTorchArchitecture,
    config: BNCI009SpectralTorchBenchmarkConfig,
    split_config: BNCI009SplitConfig,
) -> RawTorchFoldResult:
    train_fit_indices, validation_indices, validation_subject = select_validation_indices(
        dataset,
        split,
        split_config=split_config,
    )
    set_torch_seed(config.seed)
    device = resolve_torch_device(config.device)
    X_train_raw = spectral.X[train_fit_indices]
    y_train = np.asarray(spectral.y[train_fit_indices], dtype=np.int64)
    X_val_raw = spectral.X[validation_indices]
    y_val = np.asarray(spectral.y[validation_indices], dtype=np.int64)
    mean, std = fit_tensor_standardizer(X_train_raw)
    X_train = apply_tensor_standardizer(X_train_raw, mean=mean, std=std)
    X_val = apply_tensor_standardizer(X_val_raw, mean=mean, std=std)
    model = build_spectral_model(
        architecture,
        input_shape=spectral.input_shape,
        n_outputs=len(dataset.class_names),
        dropout_rate=config.dropout_rate,
    ).to(device)
    class_weights = fit_class_weights(y_train, n_classes=len(dataset.class_names), weighting=config.class_weighting)
    history, best_epoch = train_raw_torch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        config=config,  # type: ignore[arg-type]
        class_weights=class_weights,
        device=device,
    )
    test_indices = np.asarray(split.test_indices, dtype=np.int64)
    X_test = apply_tensor_standardizer(spectral.X[test_indices], mean=mean, std=std)
    probabilities = predict_raw_torch_probabilities(
        model,
        X_test,
        batch_size=config.batch_size,
        device=device,
    )
    y_pred = np.asarray(np.argmax(probabilities, axis=1), dtype=np.int64)
    y_true = np.asarray(spectral.y[test_indices], dtype=np.int64)
    metrics = evaluate_binary_p300_predictions(
        y_true,
        y_pred,
        target_score=probabilities[:, 0],
        class_names=dataset.class_names,
    )
    for array in (train_fit_indices, validation_indices, test_indices, y_true, y_pred, probabilities):
        array.setflags(write=False)
    return RawTorchFoldResult(
        split=split,
        validation_subject=validation_subject,
        train_fit_indices=train_fit_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        best_epoch=best_epoch,
        history=tuple(history),
        metrics=metrics,
    )


def _write_spectral_torch_payload(
    run_dir: Path,
    config: BNCI2014009Config,
    result: SpectralTorchBenchmarkResult,
) -> None:
    arrays_dir = run_dir / "arrays"
    arrays_dir.mkdir(parents=True)
    _write_json(run_dir / "config.json", _config_payload(config, result))
    _write_json(run_dir / "environment.json", _environment_payload())
    _write_json(run_dir / "evaluation.json", _evaluation_payload(result))
    _write_json(run_dir / "training.json", _training_payload(result))
    _write_json(run_dir / "comparison.json", _comparison_payload(config, result))
    for variant in result.variants:
        prefix = variant.model_id.replace("-", "_")
        _write_array(
            arrays_dir / f"{prefix}_test_indices.npy",
            np.concatenate([fold.test_indices for fold in variant.folds]),
        )
        _write_array(arrays_dir / f"{prefix}_y_true.npy", np.concatenate([fold.y_true for fold in variant.folds]))
        _write_array(arrays_dir / f"{prefix}_y_pred.npy", np.concatenate([fold.y_pred for fold in variant.folds]))
        _write_array(
            arrays_dir / f"{prefix}_probabilities.npy",
            np.concatenate([fold.probabilities for fold in variant.folds]),
        )
    files = _file_inventory(run_dir)
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": config.artifacts.schema_version,
            "config_hash": result.config_hash,
            "benchmark_version": SPECTRAL_TORCH_BENCHMARK_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "file_count": len(files),
            "files": files,
        },
    )


def _config_payload(config: BNCI2014009Config, result: SpectralTorchBenchmarkResult) -> dict[str, Any]:
    return {
        "config_hash": result.config_hash,
        "benchmark_version": SPECTRAL_TORCH_BENCHMARK_VERSION,
        "preprocessing": result.preprocessing,
        "config": config.model_dump(mode="json"),
    }


def _evaluation_payload(result: SpectralTorchBenchmarkResult) -> dict[str, Any]:
    return {
        "model_id": "spectral-torch",
        "config_hash": result.config_hash,
        "class_names": list(result.class_names),
        "preprocessing": result.preprocessing,
        "variants": [
            {
                "model_id": variant.model_id,
                "architecture": variant.architecture,
                "method": variant.method,
                "input_shape": list(variant.input_shape.tensor_shape),
                "summary": variant.summary,
                "folds": [
                    {
                        "name": fold.split.name,
                        "test_subjects": list(fold.split.test_subjects),
                        "validation_subject": fold.validation_subject,
                        "best_epoch": fold.best_epoch,
                        "metrics": fold.metrics.to_payload(),
                    }
                    for fold in variant.folds
                ],
            }
            for variant in result.variants
        ],
    }


def _training_payload(result: SpectralTorchBenchmarkResult) -> dict[str, Any]:
    return {
        "variants": [
            {
                "model_id": variant.model_id,
                "architecture": variant.architecture,
                "method": variant.method,
                "folds": [
                    {
                        "name": fold.split.name,
                        "validation_subject": fold.validation_subject,
                        "n_train_fit": int(fold.train_fit_indices.size),
                        "n_validation": int(fold.validation_indices.size),
                        "n_test": int(fold.test_indices.size),
                        "best_epoch": fold.best_epoch,
                        "epochs_ran": len(fold.history),
                        "history": list(fold.history),
                    }
                    for fold in variant.folds
                ],
            }
            for variant in result.variants
        ],
    }


def _comparison_payload(config: BNCI2014009Config, result: SpectralTorchBenchmarkResult) -> dict[str, Any]:
    classical = _best_reference(config.artifacts.root / config.classical.model_id)
    raw = _best_reference(config.artifacts.root / config.raw_torch.model_id)
    raw_alignment = _raw_split_alignment(config, result)
    return {
        "classical_reference": classical,
        "raw_torch_reference": raw,
        "raw_split_alignment": raw_alignment,
        "deferred_methods": [
            {
                "method": method,
                "reason": (
                    "Deferred from Stage 6 because the default project time-frequency contracts "
                    "were designed for longer epochs; P300-specific windows should be introduced "
                    "only with a separate validated tensor contract."
                ),
            }
            for method in config.spectral_torch.deferred_methods
        ],
        "variants": [
            {
                "model_id": variant.model_id,
                "balanced_accuracy_mean": variant.summary["balanced_accuracy_mean"],
                "delta_vs_best_classical": (
                    variant.summary["balanced_accuracy_mean"] - classical["best_balanced_accuracy_mean"]
                    if classical
                    else None
                ),
                "delta_vs_best_raw_torch": (
                    variant.summary["balanced_accuracy_mean"] - raw["best_balanced_accuracy_mean"] if raw else None
                ),
            }
            for variant in result.variants
        ],
    }


def _best_reference(root: Path) -> dict[str, Any] | None:
    candidates = (
        sorted(path for path in root.iterdir() if (path / "evaluation.json").is_file())
        if root.exists()
        else []
    )
    if not candidates:
        return None
    run_dir = candidates[0]
    evaluation = _load_json(run_dir / "evaluation.json")
    variants = evaluation.get("variants", [])
    if not isinstance(variants, list) or not variants:
        return None
    best = max(variants, key=lambda row: row["summary"]["balanced_accuracy_mean"])
    return {
        "run_dir": run_dir.as_posix(),
        "best_model_id": best["model_id"],
        "best_balanced_accuracy_mean": best["summary"]["balanced_accuracy_mean"],
    }


def _raw_split_alignment(config: BNCI2014009Config, result: SpectralTorchBenchmarkResult) -> dict[str, Any] | None:
    root = config.artifacts.root / config.raw_torch.model_id
    candidates = sorted(path for path in root.iterdir() if (path / "arrays").is_dir()) if root.exists() else []
    if not candidates or not result.variants:
        return None
    raw_arrays = candidates[0] / "arrays"
    raw_candidates = sorted(raw_arrays.glob("*_test_indices.npy"))
    if not raw_candidates:
        return None
    reference_indices = np.load(raw_candidates[0], allow_pickle=False).astype(np.int64, copy=False)
    spectral_indices = np.concatenate([fold.test_indices for fold in result.variants[0].folds]).astype(
        np.int64,
        copy=False,
    )
    return {
        "reference_run_dir": candidates[0].as_posix(),
        "reference_file": raw_candidates[0].relative_to(candidates[0]).as_posix(),
        "matches_raw_test_indices": bool(np.array_equal(reference_indices, spectral_indices)),
        "raw_test_indices_sha256": _array_hash(reference_indices),
        "spectral_test_indices_sha256": _array_hash(spectral_indices),
    }


def _preprocessing_payload(config: FFTConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def _environment_payload() -> dict[str, Any]:
    packages = {}
    for name in ("mne", "moabb", "numpy", "pydantic", "scikit-learn", "scipy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git_commit, git_dirty = _git_state()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        inventory[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return inventory


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _write_array(path: Path, array: NDArray[Any]) -> None:
    with path.open("wb") as file:
        np.save(file, array, allow_pickle=False)


def _array_hash(values: NDArray[np.integer[Any]]) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return commit, bool(status.strip())
