from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from features.config import FeatureExtractionConfig


@dataclass(frozen=True, slots=True)
class CropResult:
    eeg: NDArray[np.floating[Any]]
    source_slice: slice
    bounds_seconds: tuple[float, float]


@dataclass(frozen=True, slots=True)
class WindowLayout:
    slices: tuple[slice, ...]
    bounds_seconds: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("A window layout must contain at least one window")
        if self.bounds_seconds.shape != (len(self.slices), 2):
            raise ValueError("Window bounds must have shape (window, 2)")
        if not np.isfinite(self.bounds_seconds).all():
            raise ValueError("Window bounds must be finite")
        if np.any(self.bounds_seconds[:, 1] <= self.bounds_seconds[:, 0]):
            raise ValueError("Every window must have a positive duration")


def crop_eeg(
    eeg: ArrayLike,
    *,
    source_sfreq: float,
    config: FeatureExtractionConfig,
) -> CropResult:
    signal = np.asarray(eeg)
    if signal.ndim != 2:
        raise ValueError("`eeg` must have shape (channel, time)")
    if not np.issubdtype(signal.dtype, np.floating):
        raise TypeError("`eeg` must have a floating-point dtype")
    if not np.isfinite(signal).all():
        raise ValueError("`eeg` must contain only finite values")
    if not np.isfinite(source_sfreq) or source_sfreq <= 0:
        raise ValueError("`source_sfreq` must be finite and positive")

    start = _seconds_to_samples(config.crop_start_seconds, source_sfreq, name="crop_start_seconds")
    stop = _seconds_to_samples(config.crop_end_seconds, source_sfreq, name="crop_end_seconds")
    if stop > signal.shape[-1]:
        available_seconds = signal.shape[-1] / source_sfreq
        raise ValueError(
            f"Configured crop ends at {config.crop_end_seconds:g} s, but the signal contains only "
            f"{available_seconds:g} s"
        )
    return CropResult(
        eeg=signal[:, start:stop],
        source_slice=slice(start, stop),
        bounds_seconds=(config.crop_start_seconds, config.crop_end_seconds),
    )


def build_window_layout(
    *,
    n_times: int,
    config: FeatureExtractionConfig,
) -> WindowLayout:
    if isinstance(n_times, bool) or not isinstance(n_times, int) or n_times < 1:
        raise ValueError("`n_times` must be a positive integer")

    expected_times = _seconds_to_samples(
        config.crop_end_seconds - config.crop_start_seconds,
        config.analysis_sfreq,
        name="crop duration",
    )
    if n_times != expected_times:
        raise ValueError(
            f"Analysis epoch has {n_times} samples, expected {expected_times} for the configured crop "
            f"and analysis sampling frequency"
        )

    if config.window_seconds is None:
        slices = (slice(0, n_times),)
    else:
        window_samples = _seconds_to_samples(
            config.window_seconds,
            config.analysis_sfreq,
            name="window_seconds",
        )
        stride_samples = _seconds_to_samples(
            config.window_stride_seconds,
            config.analysis_sfreq,
            name="window_stride_seconds",
        )
        starts = range(0, n_times - window_samples + 1, stride_samples)
        slices = tuple(slice(start, start + window_samples) for start in starts)

    bounds = np.asarray(
        [
            (
                config.crop_start_seconds + window.start / config.analysis_sfreq,
                config.crop_start_seconds + window.stop / config.analysis_sfreq,
            )
            for window in slices
        ],
        dtype=np.float64,
    )
    return WindowLayout(slices=slices, bounds_seconds=bounds)


def _seconds_to_samples(seconds: float | None, sfreq: float, *, name: str) -> int:
    if seconds is None:
        raise ValueError(f"`{name}` must not be None")
    exact_samples = seconds * sfreq
    samples = round(exact_samples)
    if not np.isclose(exact_samples, samples, rtol=0.0, atol=1e-12):
        raise ValueError(f"`{name}` does not resolve to an integer sample at {sfreq:g} Hz")
    return samples
