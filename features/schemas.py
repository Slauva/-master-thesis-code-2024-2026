from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from utils.datasets.schemas import Sample

FeatureLayout = Literal["channel_features", "channel_matrix", "channel_histogram"]


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    name: str
    layout: FeatureLayout
    values: NDArray[np.floating[Any]]
    feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature block name must not be empty")
        if self.values.ndim != 3:
            raise ValueError("Feature block values must be three-dimensional")
        if not np.issubdtype(self.values.dtype, np.floating):
            raise TypeError("Feature block values must have a floating-point dtype")
        if not np.isfinite(self.values).all():
            raise ValueError("Feature block values must contain only finite values")

        if self.layout == "channel_matrix":
            if self.values.shape[1] != self.values.shape[2]:
                raise ValueError("Channel-matrix feature blocks must have shape (window, channel, channel)")
            if self.feature_names:
                raise ValueError("Channel-matrix feature blocks derive labels from EEG channel names")
        elif len(self.feature_names) != self.values.shape[2]:
            raise ValueError("Feature names must match the final feature/code axis")


@dataclass(frozen=True, slots=True)
class FeatureSet:
    sample: Sample
    blocks: tuple[FeatureBlock, ...]
    window_bounds_seconds: NDArray[np.float64]
    eeg_channels: tuple[str, ...]
    analysis_sfreq: float

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("A feature set must contain at least one feature block")
        block_names = [block.name for block in self.blocks]
        if len(set(block_names)) != len(block_names):
            raise ValueError("Feature block names must be unique")
        if not self.eeg_channels:
            raise ValueError("EEG channel names must not be empty")
        if len(set(self.eeg_channels)) != len(self.eeg_channels):
            raise ValueError("EEG channel names must be unique")
        if not np.isfinite(self.analysis_sfreq) or self.analysis_sfreq <= 0:
            raise ValueError("Analysis sampling frequency must be finite and positive")

        n_windows = self.blocks[0].values.shape[0]
        if self.window_bounds_seconds.shape != (n_windows, 2):
            raise ValueError("Window bounds must have shape (window, 2)")
        if not np.isfinite(self.window_bounds_seconds).all():
            raise ValueError("Window bounds must be finite")
        if np.any(self.window_bounds_seconds[:, 1] <= self.window_bounds_seconds[:, 0]):
            raise ValueError("Every feature window must have a positive duration")

        n_channels = len(self.eeg_channels)
        for block in self.blocks:
            if block.values.shape[0] != n_windows:
                raise ValueError("All feature blocks must use the same number of windows")
            if block.values.shape[1] != n_channels:
                raise ValueError("Every feature block channel axis must match `eeg_channels`")
            if block.layout == "channel_matrix" and block.values.shape[2] != n_channels:
                raise ValueError("Channel matrices must match `eeg_channels` on both matrix axes")


def flatten_feature_set(
    feature_set: FeatureSet,
    *,
    block_names: tuple[str, ...] | None = None,
) -> tuple[NDArray[np.floating[Any]], tuple[str, ...]]:
    selected = _select_blocks(feature_set, block_names)
    flattened: list[NDArray[np.floating[Any]]] = []
    names: list[str] = []
    for block in selected:
        if block.layout == "channel_matrix":
            block_values, block_feature_names = _vectorize_symmetric_block(
                block,
                eeg_channels=feature_set.eeg_channels,
            )
        else:
            block_values = block.values.reshape(block.values.shape[0], -1)
            block_feature_names = tuple(
                f"{block.name}:{channel}:{feature_name}"
                for channel in feature_set.eeg_channels
                for feature_name in block.feature_names
            )
        flattened.append(block_values)
        names.extend(block_feature_names)

    output_dtype = np.result_type(*(array.dtype for array in flattened))
    matrix = np.concatenate(flattened, axis=1).astype(output_dtype, copy=False)
    return matrix, tuple(names)


def _select_blocks(
    feature_set: FeatureSet,
    block_names: tuple[str, ...] | None,
) -> tuple[FeatureBlock, ...]:
    if block_names is None:
        return feature_set.blocks
    if not block_names:
        raise ValueError("`block_names` must not be empty")
    if len(set(block_names)) != len(block_names):
        raise ValueError("Requested feature block names must be unique")

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
    off_diagonal = upper_rows != upper_columns
    values[:, off_diagonal] *= np.sqrt(2.0)
    names = tuple(
        f"{block.name}:{eeg_channels[row]}:{eeg_channels[column]}"
        for row, column in zip(upper_rows, upper_columns, strict=True)
    )
    return values, names
