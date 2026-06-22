from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.bnci2014_001.data import BNCIEpochMetadata, BNCISampleKey
from features.classical import (
    extract_spatial_features,
    extract_spectral_features,
    extract_time_features,
    prepare_feature_windows,
)
from features.config import FeatureExtractionConfig, load_feature_config
from features.local_patterns import extract_local_pattern_features
from features.schemas import FeatureBlock


@dataclass(frozen=True, slots=True)
class BNCIPreparedEpoch:
    metadata: BNCIEpochMetadata
    eeg: NDArray[np.floating[Any]]
    source_sfreq: float
    eeg_channels: tuple[str, ...]
    bounds_seconds: tuple[float, float]


@dataclass(frozen=True, slots=True)
class BNCIFeatureSet:
    metadata: BNCIEpochMetadata
    blocks: tuple[FeatureBlock, ...]
    window_bounds_seconds: NDArray[np.float64]
    eeg_channels: tuple[str, ...]
    analysis_sfreq: float

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("A BNCI feature set must contain at least one feature block")
        if not self.eeg_channels:
            raise ValueError("EEG channel names must not be empty")
        n_windows = self.blocks[0].values.shape[0]
        if self.window_bounds_seconds.shape != (n_windows, 2):
            raise ValueError("Window bounds must have shape (window, 2)")
        for block in self.blocks:
            if block.values.shape[0] != n_windows:
                raise ValueError("All feature blocks must use the same number of windows")
            if block.values.shape[1] != len(self.eeg_channels):
                raise ValueError("Feature block channel axis must match EEG channels")


@dataclass(frozen=True, slots=True)
class BNCIFeatureMatrix:
    X: NDArray[np.floating[Any]]
    feature_names: tuple[str, ...]
    sample_keys: tuple[BNCISampleKey, ...]
    window_indices: NDArray[np.int64]
    window_bounds_seconds: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("Feature matrix must be two-dimensional")
        if self.X.shape[0] != len(self.sample_keys):
            raise ValueError("Sample keys must match feature rows")
        if self.X.shape[0] != self.window_indices.shape[0]:
            raise ValueError("Window indices must match feature rows")
        if self.X.shape[0] != self.window_bounds_seconds.shape[0]:
            raise ValueError("Window bounds must match feature rows")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("Feature names must match feature columns")
        if not np.isfinite(self.X).all():
            raise ValueError("Feature matrix must contain only finite values")


def load_bnci_feature_config(
    *,
    overrides: dict[str, object] | None = None,
) -> FeatureExtractionConfig:
    bnci_overrides: dict[str, object] = {
        "crop_start_seconds": 0.0,
        "crop_end_seconds": 4.0,
    }
    if overrides:
        bnci_overrides.update(overrides)
    return load_feature_config(overrides=bnci_overrides)


def prepare_bnci_epoch(
    eeg: NDArray[np.floating[Any]],
    metadata: BNCIEpochMetadata,
    *,
    source_sfreq: float = 250.0,
    eeg_channels: tuple[str, ...] | None = None,
    epoch_seconds: float = 4.0,
) -> BNCIPreparedEpoch:
    signal = np.asarray(eeg)
    if signal.ndim != 2:
        raise ValueError("BNCI epoch EEG must have shape (channel, time)")
    if not np.issubdtype(signal.dtype, np.floating):
        raise TypeError("BNCI epoch EEG must have a floating-point dtype")
    if not np.isfinite(signal).all():
        raise ValueError("BNCI epoch EEG must contain only finite values")
    if not np.isfinite(source_sfreq) or source_sfreq <= 0:
        raise ValueError("source_sfreq must be finite and positive")
    expected_times = _seconds_to_samples(epoch_seconds, source_sfreq)
    if signal.shape[-1] < expected_times:
        raise ValueError(
            f"BNCI epoch has {signal.shape[-1]} samples but {expected_times} are required "
            f"for a {epoch_seconds:g} s half-open interval"
        )
    channels = eeg_channels or tuple(f"EEG_{index:02d}" for index in range(signal.shape[0]))
    if len(channels) != signal.shape[0]:
        raise ValueError("eeg_channels must match the EEG channel axis")
    return BNCIPreparedEpoch(
        metadata=metadata,
        eeg=np.asarray(signal[:, :expected_times], dtype=signal.dtype),
        source_sfreq=source_sfreq,
        eeg_channels=channels,
        bounds_seconds=(0.0, epoch_seconds),
    )


def extract_bnci_feature_set(
    eeg: NDArray[np.floating[Any]],
    metadata: BNCIEpochMetadata,
    *,
    config: FeatureExtractionConfig,
    source_sfreq: float = 250.0,
    eeg_channels: tuple[str, ...] | None = None,
) -> BNCIFeatureSet:
    prepared = prepare_bnci_epoch(
        eeg,
        metadata,
        source_sfreq=source_sfreq,
        eeg_channels=eeg_channels,
        epoch_seconds=config.crop_end_seconds - config.crop_start_seconds,
    )
    windows = prepare_feature_windows(prepared.eeg, source_sfreq=prepared.source_sfreq, config=config)
    blocks: list[FeatureBlock] = []
    for group in config.feature_groups:
        if group == "time":
            blocks.append(extract_time_features(windows, dtype=config.dtype))
        elif group == "spectral":
            blocks.append(extract_spectral_features(windows, config=config))
        elif group == "spatial":
            blocks.extend(extract_spatial_features(windows, dtype=config.dtype))
        elif group == "local_patterns":
            blocks.extend(extract_local_pattern_features(windows, config=config))
        else:
            raise ValueError(f"Unsupported feature group: {group!r}")
    return BNCIFeatureSet(
        metadata=metadata,
        blocks=tuple(blocks),
        window_bounds_seconds=windows.bounds_seconds,
        eeg_channels=prepared.eeg_channels,
        analysis_sfreq=config.analysis_sfreq,
    )


def flatten_bnci_feature_set(
    feature_set: BNCIFeatureSet,
    *,
    block_names: tuple[str, ...] | None = None,
) -> BNCIFeatureMatrix:
    blocks = _select_blocks(feature_set, block_names)
    flattened: list[NDArray[np.floating[Any]]] = []
    names: list[str] = []
    for block in blocks:
        if block.layout == "channel_matrix":
            values, block_names_out = _vectorize_symmetric_block(
                block,
                eeg_channels=feature_set.eeg_channels,
            )
        else:
            values = block.values.reshape(block.values.shape[0], -1)
            block_names_out = tuple(
                f"{block.name}:{channel}:{feature_name}"
                for channel in feature_set.eeg_channels
                for feature_name in block.feature_names
            )
        flattened.append(values)
        names.extend(block_names_out)
    X = np.concatenate(flattened, axis=1)
    n_rows = X.shape[0]
    return BNCIFeatureMatrix(
        X=X,
        feature_names=tuple(names),
        sample_keys=tuple([feature_set.metadata.sample_key] * n_rows),
        window_indices=np.arange(n_rows, dtype=np.int64),
        window_bounds_seconds=feature_set.window_bounds_seconds,
    )


def _select_blocks(
    feature_set: BNCIFeatureSet,
    block_names: tuple[str, ...] | None,
) -> tuple[FeatureBlock, ...]:
    if block_names is None:
        return feature_set.blocks
    if not block_names:
        raise ValueError("block_names must not be empty")
    available = {block.name: block for block in feature_set.blocks}
    missing = [name for name in block_names if name not in available]
    if missing:
        raise KeyError(f"Unknown feature block(s): {', '.join(missing)}")
    return tuple(available[name] for name in block_names)


def _vectorize_symmetric_block(
    block: FeatureBlock,
    *,
    eeg_channels: tuple[str, ...],
) -> tuple[NDArray[np.floating[Any]], tuple[str, ...]]:
    upper_rows, upper_columns = np.triu_indices(len(eeg_channels))
    values = block.values[:, upper_rows, upper_columns].copy()
    values[:, upper_rows != upper_columns] *= np.sqrt(2.0)
    names = tuple(
        f"{block.name}:{eeg_channels[row]}:{eeg_channels[column]}"
        for row, column in zip(upper_rows, upper_columns, strict=True)
    )
    return values, names


def _seconds_to_samples(seconds: float, sfreq: float) -> int:
    exact = seconds * sfreq
    samples = round(exact)
    if not np.isclose(exact, samples, rtol=0.0, atol=1e-12):
        raise ValueError(f"{seconds:g} s does not resolve to integer samples at {sfreq:g} Hz")
    return samples
