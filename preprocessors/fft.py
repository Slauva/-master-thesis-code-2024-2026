from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window, resample_poly

from preprocessors.config import FFTConfig, build_frequency_grid
from preprocessors.schemas import SpectralTransformResult


def compute_fft_psd(
    eeg: ArrayLike,
    *,
    source_sfreq: float,
    config: FFTConfig,
) -> SpectralTransformResult:
    """Compute a channel-wise, one-sided FFT periodogram on the configured grid."""
    signal = np.asarray(eeg, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError("`eeg` must have shape (channel, time)")
    if signal.shape[1] < 2:
        raise ValueError("FFT preprocessing requires at least two time samples")
    if not np.isfinite(signal).all():
        raise ValueError("`eeg` must contain only finite values")
    if not np.isfinite(source_sfreq) or source_sfreq <= 0:
        raise ValueError("`source_sfreq` must be finite and positive")

    signal = _resample(signal, source_sfreq=source_sfreq, target_sfreq=config.analysis_sfreq)
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
    rebinned_psd = _rebin_density(
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


def _resample(
    signal: NDArray[np.float64],
    *,
    source_sfreq: float,
    target_sfreq: float,
) -> NDArray[np.float64]:
    if np.isclose(source_sfreq, target_sfreq, rtol=0.0, atol=1e-12):
        return signal

    ratio = (Fraction(str(target_sfreq)) / Fraction(str(source_sfreq))).limit_denominator(1_000_000)
    effective_sfreq = source_sfreq * ratio.numerator / ratio.denominator
    if not np.isclose(effective_sfreq, target_sfreq, rtol=1e-12, atol=1e-12):
        raise ValueError(
            f"Could not represent resampling ratio from {source_sfreq:g} Hz to {target_sfreq:g} Hz"
        )
    return resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)


def _rebin_density(
    psd: NDArray[np.float64],
    *,
    source_frequencies: NDArray[np.float64],
    target_frequencies: NDArray[np.float64],
    target_width: float,
) -> NDArray[np.float64]:
    if source_frequencies.size < 2:
        raise ValueError("PSD rebinning requires at least two source frequency bins")

    source_edges = _frequency_edges(source_frequencies)
    target_edges = np.concatenate(
        (
            target_frequencies - target_width / 2.0,
            np.array([target_frequencies[-1] + target_width / 2.0]),
        )
    )
    overlap = np.maximum(
        0.0,
        np.minimum(target_edges[1:, np.newaxis], source_edges[np.newaxis, 1:])
        - np.maximum(target_edges[:-1, np.newaxis], source_edges[np.newaxis, :-1]),
    )
    covered_width = overlap.sum(axis=1)
    if not np.allclose(covered_width, target_width):
        raise ValueError("The native FFT grid does not fully cover the configured output frequency bins")
    return psd @ overlap.T / target_width


def _frequency_edges(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
    midpoints = (frequencies[:-1] + frequencies[1:]) / 2.0
    first = frequencies[0] - (midpoints[0] - frequencies[0])
    last = frequencies[-1] + (frequencies[-1] - midpoints[-1])
    return np.concatenate((np.array([first]), midpoints, np.array([last])))
