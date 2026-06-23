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

from experiments.bnci2014_009.baselines import validate_training_split
from experiments.bnci2014_009.config import (
    BNCI009RawTorchArchitecture,
    BNCI009RawTorchBenchmarkConfig,
    BNCI009SplitConfig,
    BNCI2014009Config,
)
from experiments.bnci2014_009.data import (
    BNCI009EpochDataset,
    BNCI009Split,
    create_leave_one_subject_splits,
    load_bnci009_epochs,
)
from experiments.bnci2014_009.metrics import (
    BinaryP300Metrics,
    evaluate_binary_p300_predictions,
    summarize_fold_metrics,
)
from experiments.random_imagery_torch.models import SpectralModelShape, build_spectral_model

RAW_TORCH_BENCHMARK_VERSION = 1


@dataclass(frozen=True, slots=True)
class BNCI009RawTensorDataset:
    X: NDArray[np.float32]
    y: NDArray[np.int64]
    sample_keys: tuple[tuple[int, str, str, int], ...]
    input_shape: SpectralModelShape

    def __post_init__(self) -> None:
        if self.X.ndim != 4:
            raise ValueError("Raw ERP tensors must have shape (epoch, plane, channel, time)")
        if self.X.shape[0] != self.y.shape[0] or self.X.shape[0] != len(self.sample_keys):
            raise ValueError("Raw ERP tensors, targets, and sample keys must align")
        if tuple(self.X.shape[1:]) != self.input_shape.tensor_shape:
            raise ValueError("Raw ERP tensor shape does not match the model input shape")
        if not np.isfinite(self.X).all():
            raise ValueError("Raw ERP tensors must be finite")


@dataclass(frozen=True, slots=True)
class RawTorchFoldResult:
    split: BNCI009Split
    validation_subject: int
    train_fit_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    y_true: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    best_epoch: int
    history: tuple[dict[str, float], ...]
    metrics: BinaryP300Metrics


@dataclass(frozen=True, slots=True)
class RawTorchVariantResult:
    architecture: BNCI009RawTorchArchitecture
    model_id: str
    input_shape: SpectralModelShape
    folds: tuple[RawTorchFoldResult, ...]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RawTorchBenchmarkResult:
    config_hash: str
    class_names: tuple[str, ...]
    variants: tuple[RawTorchVariantResult, ...]


@dataclass(frozen=True, slots=True)
class RawTorchExecutionResult:
    run_dir: Path
    config_hash: str
    reused: bool
    evaluation: dict[str, Any]


class BNCI009RawCNN(nn.Module):
    def __init__(
        self,
        *,
        input_shape: SpectralModelShape,
        n_classes: int,
        hidden_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(input_shape.input_planes, hidden_channels, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(input_shape.electrodes, 1), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(hidden_channels, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != self.input_shape.tensor_shape:
            raise ValueError(f"Raw CNN expected (batch, {self.input_shape.tensor_shape}), got {tuple(inputs.shape)}")
        return self.classifier(self.features(inputs))


def execute_raw_torch_benchmark(
    config: BNCI2014009Config,
    *,
    reuse_existing: bool = False,
) -> RawTorchExecutionResult:
    run_dir = get_raw_torch_run_dir(config)
    if run_dir.exists():
        if not reuse_existing:
            raise FileExistsError(f"BNCI2014_009 raw Torch benchmark already exists: {run_dir}")
        validate_raw_torch_manifest(run_dir)
        return RawTorchExecutionResult(
            run_dir=run_dir,
            config_hash=build_raw_torch_config_hash(config),
            reused=True,
            evaluation=_load_json(run_dir / "evaluation.json"),
        )
    result = run_raw_torch_benchmark(config)
    written_dir = write_raw_torch_benchmark(config, result)
    return RawTorchExecutionResult(
        run_dir=written_dir,
        config_hash=result.config_hash,
        reused=False,
        evaluation=_load_json(written_dir / "evaluation.json"),
    )


def run_raw_torch_benchmark(
    config: BNCI2014009Config,
    *,
    dataset: BNCI009EpochDataset | None = None,
) -> RawTorchBenchmarkResult:
    epochs = dataset if dataset is not None else load_bnci009_epochs(config)
    raw = materialize_raw_tensor_dataset(epochs)
    variants: list[RawTorchVariantResult] = []
    for architecture in config.raw_torch.architectures:
        print(f"[bnci009 raw-torch] training {architecture}", flush=True)
        variants.append(
            run_raw_torch_variant(
                epochs,
                raw,
                architecture=architecture,
                config=config.raw_torch,
                split_config=config.split,
            )
        )
    return RawTorchBenchmarkResult(
        config_hash=build_raw_torch_config_hash(config),
        class_names=epochs.class_names,
        variants=tuple(variants),
    )


def materialize_raw_tensor_dataset(dataset: BNCI009EpochDataset) -> BNCI009RawTensorDataset:
    X = np.asarray(dataset.X[:, np.newaxis, :, :], dtype=np.float32)
    y = np.asarray(dataset.y, dtype=np.int64)
    input_shape = SpectralModelShape(
        input_planes=int(X.shape[1]),
        electrodes=int(X.shape[2]),
        width=int(X.shape[3]),
    )
    X.setflags(write=False)
    y.setflags(write=False)
    return BNCI009RawTensorDataset(
        X=X,
        y=y,
        sample_keys=dataset.sample_keys,
        input_shape=input_shape,
    )


def run_raw_torch_variant(
    dataset: BNCI009EpochDataset,
    raw: BNCI009RawTensorDataset,
    *,
    architecture: BNCI009RawTorchArchitecture,
    config: BNCI009RawTorchBenchmarkConfig,
    split_config: BNCI009SplitConfig,
) -> RawTorchVariantResult:
    if raw.sample_keys != dataset.sample_keys:
        raise ValueError("Raw tensor sample keys do not match BNCI2014_009 epoch order")
    splits = create_leave_one_subject_splits(dataset)
    folds = tuple(
        fit_predict_raw_torch_fold(
            dataset,
            raw,
            split,
            architecture=architecture,
            config=config,
            split_config=split_config,
        )
        for split in splits
    )
    return RawTorchVariantResult(
        architecture=architecture,
        model_id=f"{architecture}-raw-erp",
        input_shape=raw.input_shape,
        folds=folds,
        summary=summarize_fold_metrics(tuple(fold.metrics for fold in folds)),
    )


def fit_predict_raw_torch_fold(
    dataset: BNCI009EpochDataset,
    raw: BNCI009RawTensorDataset,
    split: BNCI009Split,
    *,
    architecture: BNCI009RawTorchArchitecture,
    config: BNCI009RawTorchBenchmarkConfig,
    split_config: BNCI009SplitConfig,
) -> RawTorchFoldResult:
    train_fit_indices, validation_indices, validation_subject = select_validation_indices(
        dataset,
        split,
        split_config=split_config,
    )
    set_torch_seed(config.seed)
    device = resolve_torch_device(config.device)
    X_train_raw = raw.X[train_fit_indices]
    y_train = np.asarray(raw.y[train_fit_indices], dtype=np.int64)
    X_val_raw = raw.X[validation_indices]
    y_val = np.asarray(raw.y[validation_indices], dtype=np.int64)
    mean, std = fit_tensor_standardizer(X_train_raw)
    X_train = apply_tensor_standardizer(X_train_raw, mean=mean, std=std)
    X_val = apply_tensor_standardizer(X_val_raw, mean=mean, std=std)
    model = build_raw_torch_model(
        architecture,
        input_shape=raw.input_shape,
        n_classes=len(dataset.class_names),
        config=config,
    ).to(device)
    class_weights = fit_class_weights(y_train, n_classes=len(dataset.class_names), weighting=config.class_weighting)
    history, best_epoch = train_raw_torch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        config=config,
        class_weights=class_weights,
        device=device,
    )
    test_indices = np.asarray(split.test_indices, dtype=np.int64)
    X_test = apply_tensor_standardizer(raw.X[test_indices], mean=mean, std=std)
    probabilities = predict_raw_torch_probabilities(
        model,
        X_test,
        batch_size=config.batch_size,
        device=device,
    )
    y_pred = np.asarray(np.argmax(probabilities, axis=1), dtype=np.int64)
    y_true = np.asarray(raw.y[test_indices], dtype=np.int64)
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


def select_validation_indices(
    dataset: BNCI009EpochDataset,
    split: BNCI009Split,
    *,
    split_config: BNCI009SplitConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    validate_training_split(dataset, split, split_config=split_config)
    validation_subject = min(split.train_subjects)
    train_indices = np.asarray(split.train_indices, dtype=np.int64)
    train_subjects = dataset.subjects[train_indices]
    validation_mask = train_subjects == validation_subject
    validation_indices = train_indices[validation_mask].astype(np.int64, copy=False)
    train_fit_indices = train_indices[~validation_mask].astype(np.int64, copy=False)
    if validation_indices.size == 0 or train_fit_indices.size == 0:
        raise ValueError("Raw Torch validation split produced an empty partition")
    _require_all_classes(dataset.y[train_fit_indices], len(dataset.class_names), "train-fit")
    _require_all_classes(dataset.y[validation_indices], len(dataset.class_names), "validation")
    train_fit_indices.setflags(write=False)
    validation_indices.setflags(write=False)
    return train_fit_indices, validation_indices, validation_subject


def build_raw_torch_model(
    architecture: BNCI009RawTorchArchitecture,
    *,
    input_shape: SpectralModelShape,
    n_classes: int,
    config: BNCI009RawTorchBenchmarkConfig,
) -> nn.Module:
    if architecture == "raw-cnn":
        return BNCI009RawCNN(
            input_shape=input_shape,
            n_classes=n_classes,
            hidden_channels=config.hidden_channels,
            dropout_rate=config.dropout_rate,
        )
    return build_spectral_model(
        architecture,
        input_shape=input_shape,
        n_outputs=n_classes,
        dropout_rate=config.dropout_rate,
    )


def fit_tensor_standardizer(
    X_train: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    mean = np.mean(X_train, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.std(X_train, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1.0e-6))
    return mean, std


def apply_tensor_standardizer(
    X: NDArray[np.float32],
    *,
    mean: NDArray[np.float32],
    std: NDArray[np.float32],
) -> NDArray[np.float32]:
    normalized = ((X - mean) / std).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError("Standardized raw ERP tensors must be finite")
    return normalized


def fit_class_weights(
    y_train: NDArray[np.int64],
    *,
    n_classes: int,
    weighting: str | None,
) -> NDArray[np.float32] | None:
    if weighting is None:
        return None
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("Cannot compute balanced class weights when a class is absent")
    weights = counts.sum() / (n_classes * counts)
    return np.asarray(weights, dtype=np.float32)


def train_raw_torch_model(
    model: nn.Module,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.int64],
    *,
    config: BNCI009RawTorchBenchmarkConfig,
    class_weights: NDArray[np.float32] | None,
    device: torch.device,
) -> tuple[list[dict[str, float]], int]:
    train_loader = _make_loader(X_train, y_train, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    val_loader = _make_loader(X_val, y_val, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    weight_tensor = torch.from_numpy(class_weights).to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
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


def predict_raw_torch_probabilities(
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


def get_raw_torch_run_dir(config: BNCI2014009Config) -> Path:
    return config.artifacts.root / config.raw_torch.model_id / build_raw_torch_config_hash(config)


def build_raw_torch_config_hash(config: BNCI2014009Config) -> str:
    payload = {
        "benchmark_version": RAW_TORCH_BENCHMARK_VERSION,
        "dataset": config.dataset.model_dump(mode="json"),
        "split": config.split.model_dump(mode="json"),
        "raw_torch": config.raw_torch.model_dump(mode="json"),
        "artifacts": config.artifacts.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_raw_torch_benchmark(
    config: BNCI2014009Config,
    result: RawTorchBenchmarkResult,
) -> Path:
    if result.config_hash != build_raw_torch_config_hash(config):
        raise ValueError("Raw Torch result hash does not match the resolved configuration")
    run_dir = get_raw_torch_run_dir(config)
    if run_dir.exists():
        raise FileExistsError(f"BNCI2014_009 raw Torch benchmark already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir.parent / f".{run_dir.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_raw_torch_payload(tmp_dir, config, result)
        tmp_dir.replace(run_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    validate_raw_torch_manifest(run_dir)
    return run_dir


def validate_raw_torch_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing raw Torch manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("file_count") != len(files):
        raise ValueError("Raw Torch manifest inventory is invalid")
    expected_files = set(files) | {"manifest.json"}
    actual_files = {path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(f"Raw Torch inventory mismatch; missing={missing}, unexpected={unexpected}")
    for relative_path, metadata in files.items():
        path = run_dir / relative_path
        if path.stat().st_size != metadata.get("bytes"):
            raise ValueError(f"Raw Torch file size mismatch: {relative_path}")
        if _sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Raw Torch file hash mismatch: {relative_path}")


def _write_raw_torch_payload(
    run_dir: Path,
    config: BNCI2014009Config,
    result: RawTorchBenchmarkResult,
) -> None:
    arrays_dir = run_dir / "arrays"
    arrays_dir.mkdir(parents=True)
    _write_json(run_dir / "config.json", _config_payload(config, result.config_hash))
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
            "benchmark_version": RAW_TORCH_BENCHMARK_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "file_count": len(files),
            "files": files,
        },
    )


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for BNCI2014_009 raw Torch but is not available")
    return torch.device(device)


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _require_all_classes(y: NDArray[np.integer[Any]], n_classes: int, partition_name: str) -> None:
    observed = set(int(value) for value in np.unique(y))
    expected = set(range(n_classes))
    if observed != expected:
        raise ValueError(f"{partition_name} partition does not contain all classes: {sorted(observed)}")


def _config_payload(config: BNCI2014009Config, config_hash: str) -> dict[str, Any]:
    return {
        "config_hash": config_hash,
        "benchmark_version": RAW_TORCH_BENCHMARK_VERSION,
        "config": config.model_dump(mode="json"),
    }


def _evaluation_payload(result: RawTorchBenchmarkResult) -> dict[str, Any]:
    return {
        "model_id": "raw-erp-torch",
        "config_hash": result.config_hash,
        "class_names": list(result.class_names),
        "variants": [
            {
                "model_id": variant.model_id,
                "architecture": variant.architecture,
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


def _training_payload(result: RawTorchBenchmarkResult) -> dict[str, Any]:
    return {
        "variants": [
            {
                "model_id": variant.model_id,
                "architecture": variant.architecture,
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


def _comparison_payload(config: BNCI2014009Config, result: RawTorchBenchmarkResult) -> dict[str, Any]:
    classical = _load_classical_reference(config)
    return {
        "classical_reference": classical,
        "variants": [
            {
                "model_id": variant.model_id,
                "balanced_accuracy_mean": variant.summary["balanced_accuracy_mean"],
                "delta_vs_best_classical": (
                    variant.summary["balanced_accuracy_mean"] - classical["best_balanced_accuracy_mean"]
                    if classical
                    else None
                ),
            }
            for variant in result.variants
        ],
        "split_alignment": "same_leave_one_subject_out_constructor",
    }


def _load_classical_reference(config: BNCI2014009Config) -> dict[str, Any] | None:
    root = config.artifacts.root / config.classical.model_id
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


def _environment_payload() -> dict[str, Any]:
    packages = {}
    for name in ("mne", "moabb", "numpy", "pydantic", "scikit-learn", "torch"):
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
        file.flush()
        os.fsync(file.fileno())


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
