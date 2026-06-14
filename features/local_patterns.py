from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from features.classical import PreparedFeatureWindows
from features.config import FeatureExtractionConfig, HistogramMode
from features.schemas import FeatureBlock


def compute_lndp_codes(
    signal: ArrayLike,
    *,
    neighbors: int = 8,
) -> NDArray[np.uint32]:
    """Compute LNDP codes following Fig. 3 of Jaiswal and Banka (2017).

    Each valid chronological window is ``P_m, ..., P_0`` with the center at
    ``P_(m/2)``. Bit zero therefore represents the rightmost difference
    ``P_1 - P_0``.
    """
    local_windows = _local_windows(signal, neighbors=neighbors)
    chronological_differences = np.diff(local_windows, axis=-1)
    bits = chronological_differences[..., ::-1] >= 0
    return _pack_bits(bits)


def compute_lgp_codes(
    signal: ArrayLike,
    *,
    neighbors: int = 8,
) -> NDArray[np.uint32]:
    """Compute 1D-LGP codes with bit zero assigned to neighbor ``P_0``."""
    local_windows = _local_windows(signal, neighbors=neighbors)
    half = neighbors // 2
    center = local_windows[..., half]
    neighbor_values = np.concatenate(
        (local_windows[..., :half], local_windows[..., half + 1 :]),
        axis=-1,
    )
    gradients = np.abs(neighbor_values - center[..., np.newaxis])
    gradient_codes = gradients - gradients.mean(axis=-1, keepdims=True)
    bits = gradient_codes[..., ::-1] >= 0
    return _pack_bits(bits)


def compute_lbp_codes(
    signal: ArrayLike,
    *,
    neighbors: int = 8,
) -> NDArray[np.uint32]:
    """Compute the paper's 1D-LBP baseline using the same neighbor order."""
    local_windows = _local_windows(signal, neighbors=neighbors)
    half = neighbors // 2
    center = local_windows[..., half]
    neighbor_values = np.concatenate(
        (local_windows[..., :half], local_windows[..., half + 1 :]),
        axis=-1,
    )
    bits = (neighbor_values - center[..., np.newaxis])[..., ::-1] >= 0
    return _pack_bits(bits)


def build_code_histograms(
    codes: ArrayLike,
    *,
    neighbors: int,
    mode: HistogramMode = "probability",
    dtype: np.dtype[Any] | str = np.float32,
) -> NDArray[np.floating[Any]]:
    validated_neighbors = _validate_neighbors(neighbors)
    code_array = np.asarray(codes)
    if code_array.ndim < 1 or code_array.shape[-1] < 1:
        raise ValueError("`codes` must have at least one code on the final axis")
    if not np.issubdtype(code_array.dtype, np.integer):
        raise TypeError("`codes` must have an integer dtype")

    bin_count = 1 << validated_neighbors
    if np.any(code_array < 0) or np.any(code_array >= bin_count):
        raise ValueError(f"`codes` must be in the range [0, {bin_count})")
    if mode not in ("count", "probability"):
        raise ValueError(f"Unsupported histogram mode: {mode!r}")

    flat_codes = code_array.reshape(-1, code_array.shape[-1])
    histograms = np.stack(
        [np.bincount(row.astype(np.int64, copy=False), minlength=bin_count) for row in flat_codes]
    ).reshape(*code_array.shape[:-1], bin_count)
    histograms = histograms.astype(np.float64, copy=False)
    if mode == "probability":
        histograms /= code_array.shape[-1]
    return histograms.astype(np.dtype(dtype), copy=False)


def extract_local_pattern_features(
    windows: PreparedFeatureWindows,
    *,
    config: FeatureExtractionConfig,
) -> tuple[FeatureBlock, FeatureBlock, FeatureBlock]:
    neighbors = config.local_pattern_neighbors
    if windows.values.shape[-1] < neighbors + 1:
        raise ValueError(
            f"Local-pattern features with m={neighbors} require at least {neighbors + 1} samples per window"
        )

    code_functions = (
        ("lndp", compute_lndp_codes),
        ("lgp", compute_lgp_codes),
        ("lbp", compute_lbp_codes),
    )
    bin_count = 1 << neighbors
    code_width = len(str(bin_count - 1))
    feature_names = tuple(f"code_{code:0{code_width}d}" for code in range(bin_count))
    blocks: list[FeatureBlock] = []
    for name, compute_codes in code_functions:
        codes = compute_codes(windows.values, neighbors=neighbors)
        histograms = build_code_histograms(
            codes,
            neighbors=neighbors,
            mode=config.histogram_mode,
            dtype=config.dtype,
        )
        blocks.append(
            FeatureBlock(
                name=name,
                layout="channel_histogram",
                values=histograms,
                feature_names=feature_names,
            )
        )
    return blocks[0], blocks[1], blocks[2]


def _local_windows(
    signal: ArrayLike,
    *,
    neighbors: int,
) -> NDArray[np.float64]:
    validated_neighbors = _validate_neighbors(neighbors)
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("`signal` must have at least one dimension")
    if not np.isfinite(values).all():
        raise ValueError("`signal` must contain only finite values")
    local_size = validated_neighbors + 1
    if values.shape[-1] < local_size:
        raise ValueError(
            f"`signal` requires at least {local_size} samples for m={validated_neighbors}, "
            f"got {values.shape[-1]}"
        )
    return sliding_window_view(values, window_shape=local_size, axis=-1)


def _validate_neighbors(neighbors: int) -> int:
    if isinstance(neighbors, bool) or not isinstance(neighbors, int):
        raise TypeError("`neighbors` must be an even integer")
    if neighbors < 2 or neighbors > 16 or neighbors % 2 != 0:
        raise ValueError("`neighbors` must be an even integer in the range [2, 16]")
    return neighbors


def _pack_bits(bits: NDArray[np.bool_]) -> NDArray[np.uint32]:
    weights = np.left_shift(np.uint32(1), np.arange(bits.shape[-1], dtype=np.uint32))
    return np.sum(bits.astype(np.uint32) * weights, axis=-1, dtype=np.uint32)
