from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from preprocessors.config import SpectralScaling


@dataclass(frozen=True, slots=True)
class SpectralTransformResult:
    eeg_power: NDArray[np.floating[Any]]
    frequencies: NDArray[np.floating[Any]]
    times: NDArray[np.floating[Any]] | None
    analysis_sfreq: float
    scaling: SpectralScaling

    def __post_init__(self) -> None:
        if self.eeg_power.ndim not in (2, 3):
            raise ValueError("`eeg_power` must have shape (channel, frequency) or (channel, frequency, time)")
        if self.frequencies.ndim != 1:
            raise ValueError("`frequencies` must be one-dimensional")
        if self.eeg_power.shape[1] != self.frequencies.size:
            raise ValueError("The EEG frequency axis must match `frequencies`")
        if self.times is None and self.eeg_power.ndim != 2:
            raise ValueError("Three-dimensional EEG power requires a time axis")
        if self.times is not None:
            if self.times.ndim != 1:
                raise ValueError("`times` must be one-dimensional")
            if self.eeg_power.ndim != 3 or self.eeg_power.shape[2] != self.times.size:
                raise ValueError("The EEG time axis must match `times`")
        if self.analysis_sfreq <= 0:
            raise ValueError("`analysis_sfreq` must be positive")
        if not np.isfinite(self.eeg_power).all():
            raise ValueError("`eeg_power` must contain only finite values")
        if np.any(self.eeg_power < 0):
            raise ValueError("`eeg_power` must be non-negative")
        if not np.isfinite(self.frequencies).all() or np.any(np.diff(self.frequencies) <= 0):
            raise ValueError("`frequencies` must be finite and strictly increasing")
        if self.times is not None and (not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)):
            raise ValueError("`times` must be finite and strictly increasing")
