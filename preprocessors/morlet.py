import numpy as np
from mne.time_frequency import morlet, tfr_array_morlet
from numpy.typing import ArrayLike, NDArray

from preprocessors._signal import prepare_eeg
from preprocessors._time_frequency import trim_and_bin_power
from preprocessors.config import MorletConfig, build_frequency_grid
from preprocessors.schemas import SpectralTransformResult


def build_morlet_cycles(
    frequencies: NDArray[np.float64],
    config: MorletConfig,
) -> NDArray[np.float64]:
    return np.clip(
        frequencies / config.n_cycles_divisor,
        config.n_cycles_min,
        config.n_cycles_max,
    )


def compute_morlet_power(
    eeg: ArrayLike,
    *,
    source_sfreq: float,
    config: MorletConfig,
) -> SpectralTransformResult:
    """Compute edge-trimmed, time-binned Morlet wavelet power."""
    signal = prepare_eeg(
        eeg,
        source_sfreq=source_sfreq,
        target_sfreq=config.analysis_sfreq,
    )
    frequencies = build_frequency_grid(config)
    n_cycles = build_morlet_cycles(frequencies, config)
    edge_samples = _morlet_edge_samples(
        sfreq=config.analysis_sfreq,
        frequencies=frequencies,
        n_cycles=n_cycles,
        zero_mean=config.zero_mean,
    )
    minimum_samples = 2 * edge_samples + config.time_bin_samples
    if signal.shape[-1] < minimum_samples:
        raise ValueError(
            "Morlet preprocessing requires at least "
            f"{minimum_samples} resampled samples for edge trimming and one time bin"
        )

    power = tfr_array_morlet(
        signal[np.newaxis, ...],
        sfreq=config.analysis_sfreq,
        freqs=frequencies,
        n_cycles=n_cycles,
        zero_mean=config.zero_mean,
        use_fft=config.use_fft,
        decim=1,
        output="power",
        n_jobs=1,
        verbose="ERROR",
    )[0]
    binned_power, times = trim_and_bin_power(
        power,
        sfreq=config.analysis_sfreq,
        edge_samples=edge_samples,
        bin_samples=config.time_bin_samples,
        method="Morlet",
    )
    output_dtype = np.dtype(config.dtype)
    return SpectralTransformResult(
        eeg_power=binned_power.astype(output_dtype, copy=False),
        frequencies=frequencies.astype(output_dtype, copy=False),
        times=times.astype(output_dtype, copy=False),
        analysis_sfreq=config.analysis_sfreq,
        scaling=config.scaling,
    )


def _morlet_edge_samples(
    *,
    sfreq: float,
    frequencies: NDArray[np.float64],
    n_cycles: NDArray[np.float64],
    zero_mean: bool,
) -> int:
    wavelets = morlet(
        sfreq=sfreq,
        freqs=frequencies,
        n_cycles=n_cycles,
        zero_mean=zero_mean,
    )
    return max(len(wavelet) for wavelet in wavelets) // 2
