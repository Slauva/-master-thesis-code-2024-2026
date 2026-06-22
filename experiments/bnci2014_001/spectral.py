from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.bnci2014_001.data import BNCIEpochMetadata
from experiments.bnci2014_001.features import prepare_bnci_epoch
from preprocessors.config import (
    PreprocessingConfig,
    PreprocessingMethod,
    load_preprocessing_config,
)
from preprocessors.fft import compute_fft_psd
from preprocessors.morlet import compute_morlet_power
from preprocessors.stft import compute_stft_psd
from preprocessors.superlet import compute_superlet_power

_TRANSFORMS = {
    "fft": compute_fft_psd,
    "morlet": compute_morlet_power,
    "superlet": compute_superlet_power,
    "stft": compute_stft_psd,
}


@dataclass(frozen=True, slots=True)
class BNCISpectralSample:
    metadata: BNCIEpochMetadata
    eeg_power: NDArray[np.floating[Any]]
    frequencies: NDArray[np.floating[Any]]
    times: NDArray[np.floating[Any]] | None
    eeg_channels: tuple[str, ...]
    source_sfreq: float
    analysis_sfreq: float
    method: PreprocessingMethod
    scaling: str
    epoch_bounds_seconds: tuple[float, float]

    def __post_init__(self) -> None:
        if self.method == "fft" and self.eeg_power.ndim != 2:
            raise ValueError("FFT BNCI spectral power must have shape (channel, frequency)")
        if self.method != "fft" and self.eeg_power.ndim != 3:
            raise ValueError("Time-frequency BNCI spectral power must have shape (channel, frequency, time)")
        if self.eeg_power.shape[0] != len(self.eeg_channels):
            raise ValueError("Spectral channel axis must match EEG channels")
        if self.eeg_power.shape[1] != self.frequencies.shape[0]:
            raise ValueError("Spectral frequency axis must match frequencies")
        if self.times is not None and self.eeg_power.shape[2] != self.times.shape[0]:
            raise ValueError("Spectral time axis must match times")
        if not np.isfinite(self.eeg_power).all():
            raise ValueError("Spectral power must be finite")


def compute_bnci_spectral_sample(
    eeg: NDArray[np.floating[Any]],
    metadata: BNCIEpochMetadata,
    *,
    method: PreprocessingMethod,
    preprocessing_config: PreprocessingConfig | None = None,
    preprocessing_overrides: dict[str, object] | None = None,
    source_sfreq: float = 250.0,
    eeg_channels: tuple[str, ...] | None = None,
    epoch_seconds: float = 4.0,
) -> BNCISpectralSample:
    if preprocessing_config is not None and preprocessing_overrides is not None:
        raise ValueError("Pass either preprocessing_config or preprocessing_overrides, not both")
    config = preprocessing_config or load_preprocessing_config(method, overrides=preprocessing_overrides)
    if config.method != method:
        raise ValueError("Preprocessing method and configuration disagree")
    prepared = prepare_bnci_epoch(
        eeg,
        metadata,
        source_sfreq=source_sfreq,
        eeg_channels=eeg_channels,
        epoch_seconds=epoch_seconds,
    )
    transformed = _TRANSFORMS[method](
        prepared.eeg,
        source_sfreq=prepared.source_sfreq,
        config=config,  # type: ignore[arg-type]
    )
    return BNCISpectralSample(
        metadata=metadata,
        eeg_power=transformed.eeg_power,
        frequencies=transformed.frequencies,
        times=transformed.times,
        eeg_channels=prepared.eeg_channels,
        source_sfreq=prepared.source_sfreq,
        analysis_sfreq=transformed.analysis_sfreq,
        method=method,
        scaling=transformed.scaling,
        epoch_bounds_seconds=prepared.bounds_seconds,
    )


def spectral_payload_nbytes(sample: BNCISpectralSample) -> int:
    total = int(sample.eeg_power.nbytes + sample.frequencies.nbytes)
    if sample.times is not None:
        total += int(sample.times.nbytes)
    return total
