from __future__ import annotations

import copy
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
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.bnci2014_001.config import BNCI2014001Config, BNCITorchFullBenchmarkConfig
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
from experiments.bnci2014_001.spectral import compute_bnci_spectral_sample
from experiments.bnci2014_001.torch_pilot import (
    apply_tensor_standardizer,
    fit_tensor_standardizer,
    resolve_torch_device,
    select_validation_indices,
    set_torch_seed,
)
from experiments.random_imagery_torch.models import (
    ArchitectureName,
    SpectralModelShape,
    build_spectral_model,
)
from preprocessors.config import PreprocessingMethod

TORCH_FULL_BENCHMARK_VERSION = 2
LEGACY_TORCH_FULL_BENCHMARK_VERSIONS = (1,)
TENSOR_CACHE_CHUNK_SIZE = 64
TENSOR_TRANSFORM = "log1p_nonnegative_power_flat_tf_v2"
LEGACY_TENSOR_TRANSFORM = "log1p_nonnegative_power"


@dataclass(frozen=True, slots=True)
class BNCISpectralTensorDataset:
    method: PreprocessingMethod
    X: NDArray[np.float32]
    y: NDArray[np.int64]
    sample_keys: tuple[tuple[int, str, str, int], ...]
    input_shape: SpectralModelShape
    tensor_transform: str

    def __post_init__(self) -> None:
        if self.X.ndim != 4:
            raise ValueError("Spectral tensors must have shape (epoch, plane, electrode, width)")
        if self.X.shape[0] != self.y.shape[0] or self.X.shape[0] != len(self.sample_keys):
            raise ValueError("Spectral tensors, targets, and sample keys must align")
        if tuple(self.X.shape[1:]) != self.input_shape.tensor_shape:
            raise ValueError("Spectral tensor shape does not match the model input shape")
        if not np.isfinite(self.X).all():
            raise ValueError("Spectral tensors must be finite")


@dataclass(frozen=True, slots=True)
class FullTorchFoldResult:
    split: BNCISplit
    validation_subject: int
    train_fit_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    y_true: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    best_epoch: int
    history: tuple[dict[str, float], ...]
    metrics: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class FullTorchVariantResult:
    architecture: ArchitectureName
    method: PreprocessingMethod
    model_id: str
    input_shape: SpectralModelShape
    folds: tuple[FullTorchFoldResult, ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FullTorchBenchmarkResult:
    config_hash: str
    class_names: tuple[str, ...]
    variants: tuple[FullTorchVariantResult, ...]


@dataclass(frozen=True, slots=True)
class FullTorchExecutionResult:
    run_dir: Path
    config_hash: str
    reused: bool
    evaluation: dict[str, Any]


def execute_full_torch_benchmark(
    config: BNCI2014001Config,
    *,
    reuse_existing: bool = False,
) -> FullTorchExecutionResult:
    run_dir = get_full_torch_run_dir(config)
    if run_dir.exists():
        if not reuse_existing:
            raise FileExistsError(f"BNCI2014_001 full Torch benchmark already exists: {run_dir}")
        validate_full_torch_manifest(run_dir)
        return FullTorchExecutionResult(
            run_dir=run_dir,
            config_hash=build_full_torch_config_hash(config),
            reused=True,
            evaluation=_load_json(run_dir / "evaluation.json"),
        )
    result = run_full_torch_benchmark(config)
    written_dir = write_full_torch_benchmark(config, result)
    return FullTorchExecutionResult(
        run_dir=written_dir,
        config_hash=result.config_hash,
        reused=False,
        evaluation=_load_json(written_dir / "evaluation.json"),
    )


def run_full_torch_benchmark(
    config: BNCI2014001Config,
    *,
    dataset: BNCIEpochDataset | None = None,
) -> FullTorchBenchmarkResult:
    epochs = dataset if dataset is not None else load_bnci_epochs(config)
    spectral_by_method = {
        method: load_or_materialize_spectral_tensor_dataset(
            config,
            epochs,
            method=method,
            source_sfreq=config.dataset.source_sfreq,
        )
        for method in config.torch_full.spectral_methods
    }
    variants: list[FullTorchVariantResult] = []
    for method in config.torch_full.spectral_methods:
        spectral = spectral_by_method[method]
        for architecture in config.torch_full.architectures:
            print(f"[bnci torch-full] training {architecture}-{method}", flush=True)
            variants.append(
                run_full_torch_variant(
                    epochs,
                    spectral,
                    architecture=architecture,
                    config=config.torch_full,
                    split_config=config.split,
                )
            )
    return FullTorchBenchmarkResult(
        config_hash=build_full_torch_config_hash(config),
        class_names=epochs.class_names,
        variants=tuple(variants),
    )


def run_full_torch_variant(
    dataset: BNCIEpochDataset,
    spectral: BNCISpectralTensorDataset,
    *,
    architecture: ArchitectureName,
    config: BNCITorchFullBenchmarkConfig,
    split_config: Any,
) -> FullTorchVariantResult:
    splits = create_leave_one_subject_splits(dataset)
    folds: list[FullTorchFoldResult] = []
    for split in splits:
        fold = fit_predict_full_torch_fold(
            dataset,
            spectral,
            split,
            architecture=architecture,
            config=config,
            split_config=split_config,
        )
        folds.append(fold)
    return FullTorchVariantResult(
        architecture=architecture,
        method=spectral.method,
        model_id=f"{architecture}-{spectral.method}-bnci",
        input_shape=spectral.input_shape,
        folds=tuple(folds),
        summary=summarize_fold_metrics(tuple(fold.metrics for fold in folds)),
    )


def fit_predict_full_torch_fold(
    dataset: BNCIEpochDataset,
    spectral: BNCISpectralTensorDataset,
    split: BNCISplit,
    *,
    architecture: ArchitectureName,
    config: BNCITorchFullBenchmarkConfig,
    split_config: Any,
) -> FullTorchFoldResult:
    train_fit_indices, validation_indices, validation_subject = select_validation_indices(
        dataset,
        split,
        split_config=split_config,
    )
    set_torch_seed(config.seed)
    device = resolve_torch_device(config.device)
    X_train_raw = spectral.X[train_fit_indices]
    y_train = spectral.y[train_fit_indices]
    X_val_raw = spectral.X[validation_indices]
    y_val = spectral.y[validation_indices]
    mean, std = fit_tensor_standardizer(X_train_raw)
    X_train = apply_tensor_standardizer(X_train_raw, mean=mean, std=std)
    X_val = apply_tensor_standardizer(X_val_raw, mean=mean, std=std)
    model = build_spectral_model(
        architecture,
        input_shape=spectral.input_shape,
        n_outputs=len(dataset.class_names),
        dropout_rate=config.dropout_rate,
    ).to(device)
    history, best_epoch = train_full_torch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        config=config,
        device=device,
    )
    test_indices = np.asarray(split.test_indices, dtype=np.int64)
    X_test = apply_tensor_standardizer(spectral.X[test_indices], mean=mean, std=std)
    probabilities = predict_full_torch_probabilities(
        model,
        X_test,
        batch_size=config.batch_size,
        device=device,
    )
    y_pred = np.asarray(np.argmax(probabilities, axis=1), dtype=np.int64)
    y_true = np.asarray(spectral.y[test_indices], dtype=np.int64)
    metrics = evaluate_multiclass_predictions(
        y_true,
        y_pred,
        class_names=dataset.class_names,
    )
    for array in (train_fit_indices, validation_indices, test_indices, y_true, y_pred, probabilities):
        array.setflags(write=False)
    return FullTorchFoldResult(
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


def materialize_spectral_tensor_dataset(
    dataset: BNCIEpochDataset,
    *,
    method: PreprocessingMethod,
    source_sfreq: float,
    chunk_dir: Path | None = None,
    chunk_size: int = TENSOR_CACHE_CHUNK_SIZE,
) -> BNCISpectralTensorDataset:
    if chunk_dir is not None:
        tensors = _materialize_spectral_tensor_chunks(
            dataset,
            method=method,
            source_sfreq=source_sfreq,
            chunk_dir=chunk_dir,
            chunk_size=chunk_size,
        )
    else:
        tensors = [
            _compute_bnci_spectral_tensor(epoch, metadata, method=method, source_sfreq=source_sfreq)
            for epoch, metadata in zip(dataset.X, dataset.metadata, strict=True)
        ]
    if not tensors:
        raise ValueError("Cannot materialize an empty spectral tensor dataset")

    input_shape: SpectralModelShape | None = None
    for tensor in tensors:
        if input_shape is None:
            input_shape = SpectralModelShape(
                input_planes=int(tensor.shape[0]),
                electrodes=int(tensor.shape[1]),
                width=int(tensor.shape[2]),
            )
        elif tensor.shape != input_shape.tensor_shape:
            raise ValueError(f"Inconsistent {method} tensor shape: {tensor.shape} != {input_shape.tensor_shape}")
    X = np.stack(tensors).astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError(f"{method} tensors must be finite")
    y = np.asarray(dataset.y, dtype=np.int64)
    X.setflags(write=False)
    y.setflags(write=False)
    if input_shape is None:  # pragma: no cover - guarded by the empty tensors check above.
        raise ValueError("Cannot materialize an empty spectral tensor dataset")
    return BNCISpectralTensorDataset(
        method=method,
        X=X,
        y=y,
        sample_keys=dataset.sample_keys,
        input_shape=input_shape,
        tensor_transform=TENSOR_TRANSFORM,
    )


def _materialize_spectral_tensor_chunks(
    dataset: BNCIEpochDataset,
    *,
    method: PreprocessingMethod,
    source_sfreq: float,
    chunk_dir: Path,
    chunk_size: int,
) -> list[NDArray[np.float32]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    tensors: list[NDArray[np.float32]] = []
    total = len(dataset.X)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk_path = chunk_dir / f"{start:05d}_{stop:05d}.npy"
        if chunk_path.is_file():
            chunk = np.load(chunk_path, allow_pickle=False).astype(np.float32, copy=False)
            print(f"[bnci torch-full] reused {method} tensor chunk {stop}/{total}", flush=True)
        else:
            chunk_rows = [
                _compute_bnci_spectral_tensor(epoch, metadata, method=method, source_sfreq=source_sfreq)
                for epoch, metadata in zip(
                    dataset.X[start:stop],
                    dataset.metadata[start:stop],
                    strict=True,
                )
            ]
            chunk = np.stack(chunk_rows).astype(np.float32, copy=False)
            _write_array(chunk_path, chunk)
            print(f"[bnci torch-full] cached {method} tensor chunk {stop}/{total}", flush=True)
        if chunk.ndim != 4 or chunk.shape[0] != stop - start:
            raise ValueError(f"Invalid cached {method} tensor chunk shape: {chunk_path} {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError(f"Cached {method} tensor chunk must be finite: {chunk_path}")
        tensors.extend(np.asarray(row, dtype=np.float32) for row in chunk)
    return tensors


def _compute_bnci_spectral_tensor(
    epoch: NDArray[np.float32],
    metadata: Any,
    *,
    method: PreprocessingMethod,
    source_sfreq: float,
) -> NDArray[np.float32]:
    sample = compute_bnci_spectral_sample(
        epoch,
        metadata,
        method=method,
        source_sfreq=source_sfreq,
    )
    power = np.log1p(np.maximum(np.asarray(sample.eeg_power, dtype=np.float32), np.float32(0.0)))
    if method == "fft":
        tensor = power[np.newaxis, :, :]
    else:
        tensor = power.reshape(power.shape[0], -1)[np.newaxis, :, :]
    return np.asarray(tensor, dtype=np.float32)


def load_or_materialize_spectral_tensor_dataset(
    config: BNCI2014001Config,
    dataset: BNCIEpochDataset,
    *,
    method: PreprocessingMethod,
    source_sfreq: float,
) -> BNCISpectralTensorDataset:
    cache_dir = _tensor_cache_dir(config, method)
    manifest_path = cache_dir / "manifest.json"
    tensor_path = cache_dir / "tensors.npy"
    if manifest_path.is_file() and tensor_path.is_file():
        cached = _load_cached_spectral_tensor_dataset(
            dataset,
            method=method,
            manifest_path=manifest_path,
            tensor_path=tensor_path,
        )
        if cached.tensor_transform == TENSOR_TRANSFORM:
            print(f"[bnci torch-full] reused {method} tensors {cached.X.shape}", flush=True)
            return cached

    migrated = _migrate_legacy_spectral_tensor_cache(config, dataset, method=method, cache_dir=cache_dir)
    if migrated is not None:
        print(f"[bnci torch-full] migrated {method} tensors {migrated.X.shape}", flush=True)
        return migrated

    print(f"[bnci torch-full] materializing {method} tensors", flush=True)
    spectral = materialize_spectral_tensor_dataset(
        dataset,
        method=method,
        source_sfreq=source_sfreq,
        chunk_dir=cache_dir / "chunks",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_array(tensor_path, spectral.X)
    _write_tensor_cache_manifest(config, spectral, tensor_path=tensor_path, manifest_path=manifest_path)
    print(f"[bnci torch-full] cached {method} tensors {spectral.X.shape}", flush=True)
    return spectral


def _load_cached_spectral_tensor_dataset(
    dataset: BNCIEpochDataset,
    *,
    method: PreprocessingMethod,
    manifest_path: Path,
    tensor_path: Path,
) -> BNCISpectralTensorDataset:
    manifest = _load_json(manifest_path)
    if _sha256_file(tensor_path) != manifest.get("tensor_sha256"):
        raise ValueError(f"Cached {method} tensor hash mismatch: {tensor_path}")
    X = np.load(tensor_path, allow_pickle=False)
    y = np.asarray(dataset.y, dtype=np.int64)
    input_shape = SpectralModelShape(
        input_planes=int(manifest["input_shape"][0]),
        electrodes=int(manifest["input_shape"][1]),
        width=int(manifest["input_shape"][2]),
    )
    return BNCISpectralTensorDataset(
        method=method,
        X=np.asarray(X, dtype=np.float32),
        y=y,
        sample_keys=dataset.sample_keys,
        input_shape=input_shape,
        tensor_transform=str(manifest["tensor_transform"]),
    )


def _migrate_legacy_spectral_tensor_cache(
    config: BNCI2014001Config,
    dataset: BNCIEpochDataset,
    *,
    method: PreprocessingMethod,
    cache_dir: Path,
) -> BNCISpectralTensorDataset | None:
    for benchmark_version in LEGACY_TORCH_FULL_BENCHMARK_VERSIONS:
        legacy_dir = _tensor_cache_dir_for_version(config, method, benchmark_version=benchmark_version)
        legacy_manifest_path = legacy_dir / "manifest.json"
        legacy_tensor_path = legacy_dir / "tensors.npy"
        if not legacy_manifest_path.is_file() or not legacy_tensor_path.is_file():
            continue
        legacy = _load_cached_spectral_tensor_dataset(
            dataset,
            method=method,
            manifest_path=legacy_manifest_path,
            tensor_path=legacy_tensor_path,
        )
        if legacy.tensor_transform != LEGACY_TENSOR_TRANSFORM:
            continue
        migrated = _flatten_legacy_spectral_tensor_dataset(legacy)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = cache_dir / "tensors.npy"
        manifest_path = cache_dir / "manifest.json"
        _write_array(tensor_path, migrated.X)
        _write_tensor_cache_manifest(config, migrated, tensor_path=tensor_path, manifest_path=manifest_path)
        return migrated
    return None


def _flatten_legacy_spectral_tensor_dataset(
    spectral: BNCISpectralTensorDataset,
) -> BNCISpectralTensorDataset:
    if spectral.method == "fft":
        X = spectral.X
    else:
        if spectral.X.ndim != 4:
            raise ValueError(f"Legacy {spectral.method} tensors must be four-dimensional")
        # Legacy time-frequency tensors were (epoch, frequency, channel, time).
        X = np.transpose(spectral.X, (0, 2, 1, 3)).reshape(
            spectral.X.shape[0],
            1,
            spectral.X.shape[2],
            spectral.X.shape[1] * spectral.X.shape[3],
        )
    X = np.asarray(X, dtype=np.float32)
    X.setflags(write=False)
    input_shape = SpectralModelShape(
        input_planes=int(X.shape[1]),
        electrodes=int(X.shape[2]),
        width=int(X.shape[3]),
    )
    return BNCISpectralTensorDataset(
        method=spectral.method,
        X=X,
        y=spectral.y,
        sample_keys=spectral.sample_keys,
        input_shape=input_shape,
        tensor_transform=TENSOR_TRANSFORM,
    )


def _write_tensor_cache_manifest(
    config: BNCI2014001Config,
    spectral: BNCISpectralTensorDataset,
    *,
    tensor_path: Path,
    manifest_path: Path,
) -> None:
    _write_json(
        manifest_path,
        {
            "method": spectral.method,
            "input_shape": list(spectral.input_shape.tensor_shape),
            "tensor_shape": list(spectral.X.shape),
            "tensor_dtype": str(spectral.X.dtype),
            "tensor_transform": spectral.tensor_transform,
            "tensor_sha256": _sha256_file(tensor_path),
            "config_hash": build_full_torch_config_hash(config),
            "benchmark_version": TORCH_FULL_BENCHMARK_VERSION,
        },
    )


def train_full_torch_model(
    model: nn.Module,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.int64],
    *,
    config: BNCITorchFullBenchmarkConfig,
    device: torch.device,
) -> tuple[list[dict[str, float]], int]:
    train_loader = _make_loader(X_train, y_train, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    val_loader = _make_loader(X_val, y_val, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_score = -np.inf
    best_loss = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            if hasattr(model, "project_max_norm_"):
                model.project_max_norm_()
            batch_size = int(y_batch.shape[0])
            train_loss += float(loss.detach().cpu()) * batch_size
            n_train += batch_size
        val_loss, val_balanced_accuracy = _evaluate_validation(model, val_loader, criterion, device=device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss / max(n_train, 1),
                "validation_loss": val_loss,
                "validation_balanced_accuracy": val_balanced_accuracy,
            }
        )
        improved = (
            val_balanced_accuracy > best_score
            or np.isclose(val_balanced_accuracy, best_score) and val_loss < best_loss
        )
        if improved:
            best_score = val_balanced_accuracy
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break
    model.load_state_dict(best_state)
    return history, best_epoch


def predict_full_torch_probabilities(
    model: nn.Module,
    X: NDArray[np.float32],
    *,
    batch_size: int,
    device: torch.device,
) -> NDArray[np.float64]:
    loader = _make_loader(X, np.zeros(X.shape[0], dtype=np.int64), batch_size=batch_size, shuffle=False, seed=0)
    model.eval()
    probabilities: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for X_batch, _ in loader:
            logits = model(X_batch.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64))
    return np.concatenate(probabilities, axis=0)


def get_full_torch_run_dir(config: BNCI2014001Config) -> Path:
    return Path(config.artifacts.root) / config.torch_full.model_id / build_full_torch_config_hash(config)


def _tensor_cache_dir(config: BNCI2014001Config, method: PreprocessingMethod) -> Path:
    return _tensor_cache_dir_for_version(config, method, benchmark_version=TORCH_FULL_BENCHMARK_VERSION)


def _tensor_cache_dir_for_version(
    config: BNCI2014001Config,
    method: PreprocessingMethod,
    *,
    benchmark_version: int,
) -> Path:
    return (
        Path(config.artifacts.root)
        / "torch-full-tensors"
        / _build_full_torch_config_hash(config, benchmark_version=benchmark_version)
        / method
    )


def build_full_torch_config_hash(config: BNCI2014001Config) -> str:
    return _build_full_torch_config_hash(config, benchmark_version=TORCH_FULL_BENCHMARK_VERSION)


def _build_full_torch_config_hash(
    config: BNCI2014001Config,
    *,
    benchmark_version: int,
) -> str:
    payload = {
        "benchmark_version": benchmark_version,
        "protocol": config.split.primary_protocol,
        "artifact_schema_version": config.artifacts.schema_version,
        "config": {
            "dataset": config.dataset.model_dump(mode="json"),
            "split": config.split.model_dump(mode="json"),
            "torch_full": config.torch_full.model_dump(mode="json"),
            "artifacts": config.artifacts.model_dump(mode="json"),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def write_full_torch_benchmark(
    config: BNCI2014001Config,
    result: FullTorchBenchmarkResult,
) -> Path:
    if result.config_hash != build_full_torch_config_hash(config):
        raise ValueError("Full Torch result hash does not match the resolved configuration")
    run_dir = get_full_torch_run_dir(config)
    if run_dir.exists():
        raise FileExistsError(f"BNCI2014_001 full Torch benchmark already exists: {run_dir}")
    root = run_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / f".{run_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_full_torch_payload(tmp_dir, config, result)
        tmp_dir.rename(run_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
    _fsync_directory(root)
    validate_full_torch_manifest(run_dir)
    return run_dir


def validate_full_torch_manifest(run_dir: Path) -> None:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Full Torch manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("file_count") != len(files):
        raise ValueError("Full Torch file inventory is invalid")
    expected_files = set(files) | {"manifest.json"}
    actual_files = {path.relative_to(run_dir).as_posix() for path in Path(run_dir).rglob("*") if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(f"Full Torch inventory mismatch; missing={missing}, unexpected={unexpected}")
    for relative_path, metadata in files.items():
        path = Path(run_dir) / relative_path
        if path.stat().st_size != metadata.get("size"):
            raise ValueError(f"Full Torch file size mismatch: {relative_path}")
        if _sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Full Torch file hash mismatch: {relative_path}")


def _write_full_torch_payload(
    run_dir: Path,
    config: BNCI2014001Config,
    result: FullTorchBenchmarkResult,
) -> None:
    (run_dir / "arrays").mkdir(parents=True)
    _write_json(run_dir / "config.json", config.model_dump(mode="json"))
    _write_json(run_dir / "environment.json", _build_environment_payload())
    _write_json(run_dir / "evaluation.json", _build_evaluation_payload(result))
    _write_json(run_dir / "training.json", _build_training_payload(result))
    _write_json(run_dir / "comparison.json", _build_comparison_payload(config, result))
    arrays_dir = run_dir / "arrays"
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
    files = {
        path.relative_to(run_dir).as_posix(): {"sha256": _sha256_file(path), "size": path.stat().st_size}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": config.artifacts.schema_version,
            "config_hash": result.config_hash,
            "benchmark_version": TORCH_FULL_BENCHMARK_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "writer": "experiments.bnci2014_001.torch_full.write_full_torch_benchmark",
            "file_count": len(files),
            "files": files,
        },
    )
    _fsync_directory(run_dir)


def _build_evaluation_payload(result: FullTorchBenchmarkResult) -> dict[str, Any]:
    return {
        "model_id": "torch-full",
        "class_names": list(result.class_names),
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


def _build_training_payload(result: FullTorchBenchmarkResult) -> dict[str, Any]:
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


def _build_comparison_payload(config: BNCI2014001Config, result: FullTorchBenchmarkResult) -> dict[str, Any]:
    references = _load_reference_scores(config)
    return {
        "references": references,
        "variants": [
            {
                "model_id": variant.model_id,
                "balanced_accuracy_mean": variant.summary["balanced_accuracy_mean"],
                "delta_vs_csp_lda": variant.summary["balanced_accuracy_mean"]
                - references["csp_lda"]["balanced_accuracy_mean"],
                "delta_vs_feature_logreg": variant.summary["balanced_accuracy_mean"]
                - references["feature_logreg"]["balanced_accuracy_mean"],
            }
            for variant in result.variants
        ],
        "split_alignment": "same_leave_one_subject_out_constructor",
    }


def _load_reference_scores(config: BNCI2014001Config) -> dict[str, dict[str, Any]]:
    roots = {
        "csp_lda": Path(config.artifacts.root) / config.baseline.model_id,
        "feature_logreg": Path(config.artifacts.root) / config.project_features.model_id,
    }
    references: dict[str, dict[str, Any]] = {}
    for name, root in roots.items():
        candidates = (
            sorted(path for path in root.iterdir() if (path / "evaluation.json").is_file())
            if root.exists()
            else []
        )
        if not candidates:
            continue
        run_dir = candidates[0]
        evaluation = _load_json(run_dir / "evaluation.json")
        references[name] = {
            "run_dir": run_dir.as_posix(),
            "balanced_accuracy_mean": evaluation["summary"]["balanced_accuracy_mean"],
        }
    return references


def _make_loader(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            losses.extend([float(loss.cpu())] * int(y_batch.shape[0]))
            y_true.extend(int(value) for value in y_batch.cpu().numpy())
            y_pred.extend(int(value) for value in torch.argmax(logits, dim=1).cpu().numpy())
    return float(np.mean(losses)), _balanced_accuracy(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
    )


def _balanced_accuracy(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    recalls = []
    for label in np.unique(y_true):
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _fsync_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Full Torch JSON file must contain an object: {path}")
    return payload


def _write_array(path: Path, array: NDArray[Any]) -> None:
    with path.open("wb") as file:
        np.save(file, array, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())


def _build_environment_payload() -> dict[str, Any]:
    package_names = ("mne", "moabb", "numpy", "pydantic", "scikit-learn", "scipy", "torch")
    git_commit, git_dirty = _git_state()
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {package_name: importlib.metadata.version(package_name) for package_name in package_names},
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
        status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return commit, bool(status.strip())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
