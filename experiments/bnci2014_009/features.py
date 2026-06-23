from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from experiments.bnci2014_009.data import BNCI009EpochDataset, BNCI009SampleKey

BNCI009_CHANNELS: tuple[str, ...] = (
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "P3",
    "P4",
    "PO7",
    "PO8",
    "F3",
    "F4",
    "FCz",
    "C3",
    "C4",
    "CP3",
    "CPz",
    "CP4",
)
ERP_FEATURE_VERSION = 1
XDAWN_RIEMANNIAN_FEATURE_VERSION = 1
DEFAULT_ERP_WINDOWS_SECONDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
)


@dataclass(frozen=True, slots=True)
class BNCI009ERPFeatureMatrix:
    X: NDArray[np.floating[Any]]
    y: NDArray[np.int64]
    sample_keys: tuple[BNCI009SampleKey, ...]
    feature_names: tuple[str, ...]
    waveform_time_indices: tuple[int, ...]
    waveform_times_seconds: tuple[float, ...]
    window_bounds_seconds: tuple[tuple[float, float], ...]
    version: int = ERP_FEATURE_VERSION

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("ERP feature matrix must be two-dimensional")
        if self.y.ndim != 1:
            raise ValueError("ERP feature targets must be one-dimensional")
        if self.X.shape[0] != self.y.shape[0] or self.X.shape[0] != len(self.sample_keys):
            raise ValueError("ERP feature rows, targets, and sample keys must align")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("ERP feature columns must align with feature names")
        if len(set(self.sample_keys)) != len(self.sample_keys):
            raise ValueError("ERP feature matrix must contain one row per BNCI009 epoch")
        if not np.isfinite(self.X).all():
            raise ValueError("ERP feature matrix must contain only finite values")


@dataclass(frozen=True, slots=True)
class BNCI009XdawnRiemannianFeatures:
    X_train: NDArray[np.float64]
    X_apply: NDArray[np.float64]
    train_covariances_shape: tuple[int, int, int]
    apply_covariances_shape: tuple[int, int, int]
    n_filters: int
    estimator: str
    tangent_metric: str
    version: int = XDAWN_RIEMANNIAN_FEATURE_VERSION

    def __post_init__(self) -> None:
        if self.X_train.ndim != 2 or self.X_apply.ndim != 2:
            raise ValueError("Tangent-space features must be two-dimensional")
        if self.X_train.shape[1] != self.X_apply.shape[1]:
            raise ValueError("Train and apply tangent-space features must have the same width")
        if not np.isfinite(self.X_train).all() or not np.isfinite(self.X_apply).all():
            raise ValueError("Tangent-space features must contain only finite values")
        if len(self.train_covariances_shape) != 3 or len(self.apply_covariances_shape) != 3:
            raise ValueError("Covariance shapes must describe three-dimensional arrays")


def build_erp_feature_matrix(
    dataset: BNCI009EpochDataset,
    *,
    source_sfreq: float,
    waveform_stride: int = 4,
    window_bounds_seconds: tuple[tuple[float, float], ...] = DEFAULT_ERP_WINDOWS_SECONDS,
    channel_names: tuple[str, ...] = BNCI009_CHANNELS,
    dtype: np.dtype[Any] | str = np.float32,
) -> BNCI009ERPFeatureMatrix:
    if source_sfreq <= 0:
        raise ValueError("source_sfreq must be positive")
    if waveform_stride < 1:
        raise ValueError("waveform_stride must be at least 1")
    if len(channel_names) != dataset.X.shape[1]:
        raise ValueError("channel_names length must match dataset channels")

    epochs = np.asarray(dataset.X, dtype=np.float64)
    waveform_indices = tuple(range(0, epochs.shape[2], waveform_stride))
    waveform = epochs[:, :, waveform_indices].reshape(epochs.shape[0], -1)
    feature_blocks: list[NDArray[np.float64]] = [waveform]
    feature_names = _waveform_feature_names(channel_names, waveform_indices, source_sfreq)

    if window_bounds_seconds:
        window_features, window_names = _window_mean_features(
            epochs,
            source_sfreq=source_sfreq,
            window_bounds_seconds=window_bounds_seconds,
            channel_names=channel_names,
        )
        feature_blocks.append(window_features)
        feature_names += window_names

    X = np.asarray(np.hstack(feature_blocks), dtype=dtype)
    X.setflags(write=False)
    y = np.asarray(dataset.y, dtype=np.int64)
    y.setflags(write=False)
    return BNCI009ERPFeatureMatrix(
        X=X,
        y=y,
        sample_keys=dataset.sample_keys,
        feature_names=feature_names,
        waveform_time_indices=waveform_indices,
        waveform_times_seconds=tuple(index / source_sfreq for index in waveform_indices),
        window_bounds_seconds=window_bounds_seconds,
    )


def fit_transform_xdawn_tangent_space(
    X_train: NDArray[np.floating[Any]],
    y_train: NDArray[np.integer[Any]],
    X_apply: NDArray[np.floating[Any]],
    *,
    n_filters: int = 2,
    estimator: str = "oas",
    tangent_metric: str = "riemann",
) -> BNCI009XdawnRiemannianFeatures:
    train = _as_epoch_tensor(X_train, name="X_train")
    apply = _as_epoch_tensor(X_apply, name="X_apply")
    targets = np.asarray(y_train, dtype=np.int64)
    if targets.ndim != 1 or targets.shape[0] != train.shape[0]:
        raise ValueError("y_train must be one-dimensional and align with X_train")
    if len(set(int(value) for value in np.unique(targets))) < 2:
        raise ValueError("xDAWN requires at least two classes in y_train")
    if train.shape[1:] != apply.shape[1:]:
        raise ValueError("X_train and X_apply must have matching channel/time dimensions")
    if n_filters < 1:
        raise ValueError("n_filters must be at least 1")

    xdawn = XdawnCovariances(nfilter=n_filters, estimator=estimator, xdawn_estimator=estimator)
    train_covariances = np.asarray(xdawn.fit_transform(train, targets), dtype=np.float64)
    apply_covariances = np.asarray(xdawn.transform(apply), dtype=np.float64)
    tangent = TangentSpace(metric=tangent_metric)
    X_train_tangent = np.asarray(tangent.fit_transform(train_covariances), dtype=np.float64)
    X_apply_tangent = np.asarray(tangent.transform(apply_covariances), dtype=np.float64)
    X_train_tangent.setflags(write=False)
    X_apply_tangent.setflags(write=False)
    return BNCI009XdawnRiemannianFeatures(
        X_train=X_train_tangent,
        X_apply=X_apply_tangent,
        train_covariances_shape=tuple(int(value) for value in train_covariances.shape),
        apply_covariances_shape=tuple(int(value) for value in apply_covariances.shape),
        n_filters=n_filters,
        estimator=estimator,
        tangent_metric=tangent_metric,
    )


def _window_mean_features(
    epochs: NDArray[np.float64],
    *,
    source_sfreq: float,
    window_bounds_seconds: tuple[tuple[float, float], ...],
    channel_names: tuple[str, ...],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    columns: list[NDArray[np.float64]] = []
    names: list[str] = []
    n_times = epochs.shape[2]
    for start_seconds, end_seconds in window_bounds_seconds:
        if start_seconds < 0 or end_seconds <= start_seconds:
            raise ValueError("Window bounds must be non-negative half-open intervals")
        start = int(round(start_seconds * source_sfreq))
        stop = int(round(end_seconds * source_sfreq))
        start = min(max(start, 0), n_times)
        stop = min(max(stop, 0), n_times)
        if stop <= start:
            raise ValueError(f"Window [{start_seconds}, {end_seconds}) contains no samples")
        means = np.mean(epochs[:, :, start:stop], axis=2)
        columns.append(means)
        names.extend(
            f"window_mean:{channel}:{start_seconds:.3f}-{end_seconds:.3f}s"
            for channel in channel_names
        )
    return np.hstack(columns), tuple(names)


def _waveform_feature_names(
    channel_names: tuple[str, ...],
    waveform_indices: tuple[int, ...],
    source_sfreq: float,
) -> tuple[str, ...]:
    return tuple(
        f"waveform:{channel}:sample_{sample_index}:t_{sample_index / source_sfreq:.6f}s"
        for channel in channel_names
        for sample_index in waveform_indices
    )


def _as_epoch_tensor(
    values: NDArray[np.floating[Any]],
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (epoch, channel, time)")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one epoch")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array
