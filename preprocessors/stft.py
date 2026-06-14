import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import ShortTimeFFT

from preprocessors._frequency import rebin_density
from preprocessors._signal import prepare_eeg
from preprocessors.config import STFTConfig, build_frequency_grid
from preprocessors.schemas import SpectralTransformResult


def compute_stft_psd(
    eeg: ArrayLike,
    *,
    source_sfreq: float,
    config: STFTConfig,
) -> SpectralTransformResult:
    """Compute an edge-valid, one-sided STFT power spectral density."""
    signal = prepare_eeg(
        eeg,
        source_sfreq=source_sfreq,
        target_sfreq=config.analysis_sfreq,
    )
    transform = build_short_time_fft(config)
    minimum_samples = minimum_unpadded_samples(transform)
    if signal.shape[-1] < minimum_samples:
        raise ValueError(
            "STFT preprocessing requires at least "
            f"{minimum_samples} resampled samples for one unpadded time slice"
        )

    p0 = transform.lower_border_end[1]
    p1 = transform.upper_border_begin(signal.shape[-1])[1]
    power = transform.spectrogram(signal, p0=p0, p1=p1, axis=-1)
    frequencies = build_frequency_grid(config)
    rebinned_power = rebin_density(
        np.moveaxis(power, 1, -1),
        source_frequencies=transform.f,
        target_frequencies=frequencies,
        target_width=config.frequency_step,
    )
    rebinned_power = np.moveaxis(rebinned_power, -1, 1)
    times = transform.t(signal.shape[-1], p0=p0, p1=p1)

    output_dtype = np.dtype(config.dtype)
    return SpectralTransformResult(
        eeg_power=rebinned_power.astype(output_dtype, copy=False),
        frequencies=frequencies.astype(output_dtype, copy=False),
        times=times.astype(output_dtype, copy=False),
        analysis_sfreq=config.analysis_sfreq,
        scaling=config.scaling,
    )


def build_short_time_fft(config: STFTConfig) -> ShortTimeFFT:
    window_samples = round(config.window_seconds * config.analysis_sfreq)
    return ShortTimeFFT.from_window(
        config.window,
        fs=config.analysis_sfreq,
        nperseg=window_samples,
        noverlap=window_samples - config.hop_samples,
        symmetric_win=False,
        fft_mode=config.fft_mode,
        mfft=config.mfft,
        scale_to=config.scaling,
    )


def minimum_unpadded_samples(transform: ShortTimeFFT) -> int:
    p0 = transform.lower_border_end[1]
    minimum_samples = (transform.m_num + 1) // 2
    while transform.upper_border_begin(minimum_samples)[1] <= p0:
        minimum_samples += 1
    return minimum_samples
