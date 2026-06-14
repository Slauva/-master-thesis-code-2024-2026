import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window

from preprocessors._frequency import rebin_density
from preprocessors._signal import prepare_eeg
from preprocessors.config import FFTConfig, build_frequency_grid
from preprocessors.schemas import SpectralTransformResult


def compute_fft_psd(
    eeg: ArrayLike,
    *,
    source_sfreq: float,
    config: FFTConfig,
) -> SpectralTransformResult:
    """Compute a channel-wise, one-sided FFT periodogram on the configured grid."""
    signal = prepare_eeg(
        eeg,
        source_sfreq=source_sfreq,
        target_sfreq=config.analysis_sfreq,
    )
    if config.demean:
        signal = signal - signal.mean(axis=-1, keepdims=True)

    n_times = signal.shape[-1]
    window = get_window(config.window, n_times, fftbins=True)
    spectrum = rfft(signal * window, axis=-1)
    psd = np.square(np.abs(spectrum)) / (config.analysis_sfreq * np.square(window).sum())
    if n_times % 2 == 0:
        psd[..., 1:-1] *= 2.0
    else:
        psd[..., 1:] *= 2.0

    native_frequencies = rfftfreq(n_times, d=1.0 / config.analysis_sfreq)
    target_frequencies = build_frequency_grid(config)
    rebinned_psd = rebin_density(
        psd,
        source_frequencies=native_frequencies,
        target_frequencies=target_frequencies,
        target_width=config.frequency_step,
    )
    output_dtype = np.dtype(config.dtype)
    return SpectralTransformResult(
        eeg_power=rebinned_psd.astype(output_dtype, copy=False),
        frequencies=target_frequencies.astype(output_dtype, copy=False),
        times=None,
        analysis_sfreq=config.analysis_sfreq,
        scaling=config.scaling,
    )
