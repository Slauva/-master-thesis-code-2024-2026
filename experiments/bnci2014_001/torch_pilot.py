from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.bnci2014_001.baselines import validate_training_split
from experiments.bnci2014_001.config import BNCISplitConfig, BNCITorchPilotConfig
from experiments.bnci2014_001.data import BNCIEpochDataset, BNCISplit
from experiments.bnci2014_001.spectral import compute_bnci_spectral_sample

TORCH_FFT_PILOT_VERSION = 1


@dataclass(frozen=True, slots=True)
class TorchPilotFoldPrediction:
    split_name: str
    validation_subject: int
    train_fit_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    y_true: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    best_epoch: int
    history: tuple[dict[str, float], ...]


class BNCILightweightFFTNet(nn.Module):
    def __init__(
        self,
        *,
        n_classes: int,
        hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def select_validation_indices(
    dataset: BNCIEpochDataset,
    split: BNCISplit,
    *,
    split_config: BNCISplitConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    validate_training_split(dataset, split, split_config=split_config)
    validation_subject = min(split.train_subjects)
    train_indices = np.asarray(split.train_indices, dtype=np.int64)
    train_subjects = dataset.subjects[train_indices]
    validation_mask = train_subjects == validation_subject
    validation_indices = train_indices[validation_mask].astype(np.int64, copy=False)
    train_fit_indices = train_indices[~validation_mask].astype(np.int64, copy=False)
    if validation_indices.size == 0 or train_fit_indices.size == 0:
        raise ValueError("Torch pilot validation split produced an empty partition")
    _require_all_classes(dataset.y[train_fit_indices], len(dataset.class_names), "train-fit")
    _require_all_classes(dataset.y[validation_indices], len(dataset.class_names), "validation")
    train_fit_indices.setflags(write=False)
    validation_indices.setflags(write=False)
    return train_fit_indices, validation_indices, validation_subject


def fit_predict_torch_fft_pilot(
    dataset: BNCIEpochDataset,
    split: BNCISplit,
    *,
    pilot_config: BNCITorchPilotConfig,
    split_config: BNCISplitConfig,
    source_sfreq: float,
) -> TorchPilotFoldPrediction:
    train_fit_indices, validation_indices, validation_subject = select_validation_indices(
        dataset,
        split,
        split_config=split_config,
    )
    set_torch_seed(pilot_config.seed)
    device = resolve_torch_device(pilot_config.device)

    X_train_raw = materialize_fft_tensors(dataset, train_fit_indices, source_sfreq=source_sfreq)
    y_train = np.asarray(dataset.y[train_fit_indices], dtype=np.int64)
    X_val_raw = materialize_fft_tensors(dataset, validation_indices, source_sfreq=source_sfreq)
    y_val = np.asarray(dataset.y[validation_indices], dtype=np.int64)
    mean, std = fit_tensor_standardizer(X_train_raw)
    X_train = apply_tensor_standardizer(X_train_raw, mean=mean, std=std)
    X_val = apply_tensor_standardizer(X_val_raw, mean=mean, std=std)

    model = BNCILightweightFFTNet(
        n_classes=len(dataset.class_names),
        hidden_channels=pilot_config.hidden_channels,
        dropout=pilot_config.dropout,
    ).to(device)
    history, best_epoch = train_torch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        config=pilot_config,
        device=device,
    )

    test_indices = np.asarray(split.test_indices, dtype=np.int64)
    X_test_raw = materialize_fft_tensors(dataset, test_indices, source_sfreq=source_sfreq)
    X_test = apply_tensor_standardizer(X_test_raw, mean=mean, std=std)
    probabilities = predict_torch_probabilities(model, X_test, batch_size=pilot_config.batch_size, device=device)
    y_pred = np.asarray(np.argmax(probabilities, axis=1), dtype=np.int64)
    y_true = np.asarray(dataset.y[test_indices], dtype=np.int64)
    for array in (train_fit_indices, validation_indices, test_indices, y_true, y_pred, probabilities):
        array.setflags(write=False)
    return TorchPilotFoldPrediction(
        split_name=split.name,
        validation_subject=validation_subject,
        train_fit_indices=train_fit_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        best_epoch=best_epoch,
        history=tuple(history),
    )


def materialize_fft_tensors(
    dataset: BNCIEpochDataset,
    indices: NDArray[np.integer[Any]],
    *,
    source_sfreq: float,
) -> NDArray[np.float32]:
    tensors: list[NDArray[np.float32]] = []
    for index in np.asarray(indices, dtype=np.int64):
        spectral = compute_bnci_spectral_sample(
            dataset.X[int(index)],
            dataset.metadata[int(index)],
            method="fft",
            source_sfreq=source_sfreq,
        )
        power = np.asarray(spectral.eeg_power, dtype=np.float32)
        power = np.log1p(np.maximum(power, np.float32(0.0)))
        tensors.append(power[np.newaxis, :, :])
    X = np.stack(tensors).astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("FFT pilot tensors must be finite")
    return X


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
        raise ValueError("Standardized FFT tensors must be finite")
    return normalized


def train_torch_model(
    model: nn.Module,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.int64],
    *,
    config: BNCITorchPilotConfig,
    device: torch.device,
) -> tuple[list[dict[str, float]], int]:
    train_loader = _make_loader(
        X_train,
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = _make_loader(
        X_val,
        y_val,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            batch_size = int(y_batch.shape[0])
            train_loss += float(loss.detach().cpu()) * batch_size
            n_train += batch_size

        val_loss, val_balanced_accuracy = _evaluate_validation(model, val_loader, criterion, device=device)
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(n_train, 1),
            "validation_loss": val_loss,
            "validation_balanced_accuracy": val_balanced_accuracy,
        }
        history.append(record)
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


def predict_torch_probabilities(
    model: nn.Module,
    X: NDArray[np.float32],
    *,
    batch_size: int,
    device: torch.device,
) -> NDArray[np.float64]:
    loader = _make_loader(
        X,
        np.zeros(X.shape[0], dtype=np.int64),
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    model.eval()
    probabilities: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for X_batch, _ in loader:
            logits = model(X_batch.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64))
    return np.concatenate(probabilities, axis=0)


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for the BNCI Torch pilot but is not available")
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


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
    labels = np.unique(y_true)
    recalls = []
    for label in labels:
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls))


def _require_all_classes(
    y: NDArray[np.integer[Any]],
    n_classes: int,
    partition_name: str,
) -> None:
    observed = set(int(value) for value in np.unique(y))
    expected = set(range(n_classes))
    if observed != expected:
        raise ValueError(f"Torch pilot {partition_name} partition is missing class(es): {sorted(expected - observed)}")
