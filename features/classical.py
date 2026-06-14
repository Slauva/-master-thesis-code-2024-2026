from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.covariance import oas

from features.config import FeatureExtractionConfig
from features.schemas import FeatureBlock
from features.windowing import build_window_layout, crop_eeg
from preprocessors._signal import prepare_eeg
from preprocessors.config import FFTConfig
from preprocessors.fft import compute_fft_psd

TIME_FEATURE_NAMES = (
    "mean",
    "variance",
    "std",
    "rms",
    "median",
    "mad",
    "peak_to_peak",
    "skewness",
    "excess_kurtosis",
    "normalized_line_length",
    "zero_crossing_rate",
    "hjorth_mobility",
    "hjorth_complexity",
)

SPECTRAL_SUMMARY_NAMES = (
    "total_power",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_entropy",
)


@dataclass(frozen=True, slots=True)
class PreparedFeatureWindows:
    values: NDArray[np.float64]
    bounds_seconds: NDArray[np.float64]
    sfreq: float

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("Prepared feature windows must have shape (window, channel, time)")
        if self.values.shape[0] < 1 or self.values.shape[1] < 1 or self.values.shape[2] < 1:
            raise ValueError("Prepared feature windows must not contain empty axes")
        if self.values.dtype != np.dtype(np.float64):
            raise TypeError("Prepared feature windows must use float64 working precision")
        if not np.isfinite(self.values).all():
            raise ValueError("Prepared feature windows must contain only finite values")
        if self.bounds_seconds.shape != (self.values.shape[0], 2):
            raise ValueError("Prepared window bounds must have shape (window, 2)")
        if not np.isfinite(self.bounds_seconds).all():
            raise ValueError("Prepared window bounds must be finite")
        if not np.isfinite(self.sfreq) or self.sfreq <= 0:
            raise ValueError("Prepared window sampling frequency must be finite and positive")


def prepare_feature_windows(
    eeg: ArrayLike,
    *,
    source_sfreq: float,
    config: FeatureExtractionConfig,
) -> PreparedFeatureWindows:
    cropped = crop_eeg(eeg, source_sfreq=source_sfreq, config=config)
    analysis_eeg = prepare_eeg(
        cropped.eeg,
        source_sfreq=source_sfreq,
        target_sfreq=config.analysis_sfreq,
    )
    layout = build_window_layout(n_times=analysis_eeg.shape[-1], config=config)
    windows = np.stack([analysis_eeg[:, window] for window in layout.slices], axis=0)
    return PreparedFeatureWindows(
        values=windows,
        bounds_seconds=layout.bounds_seconds,
        sfreq=config.analysis_sfreq,
    )


def extract_time_features(
    windows: PreparedFeatureWindows,
    *,
    dtype: np.dtype[Any] | str = np.float32,
) -> FeatureBlock:
    signal = windows.values
    if signal.shape[-1] < 3:
        raise ValueError("Time features require at least three samples per window")

    mean = signal.mean(axis=-1)
    centered = signal - mean[..., np.newaxis]
    variance = np.mean(np.square(centered), axis=-1)
    std = np.sqrt(variance)
    rms = np.sqrt(np.mean(np.square(signal), axis=-1))
    median = np.median(signal, axis=-1)
    mad = np.median(np.abs(signal - median[..., np.newaxis]), axis=-1)
    peak_to_peak = np.ptp(signal, axis=-1)

    third_moment = np.mean(np.power(centered, 3), axis=-1)
    fourth_moment = np.mean(np.power(centered, 4), axis=-1)
    skewness = _safe_divide(third_moment, np.power(variance, 1.5))
    excess_kurtosis = _safe_divide(fourth_moment, np.square(variance)) - np.where(variance > 0, 3.0, 0.0)

    first_difference = np.diff(signal, axis=-1)
    second_difference = np.diff(first_difference, axis=-1)
    normalized_line_length = np.mean(np.abs(first_difference), axis=-1)
    zero_crossing_rate = np.mean(
        np.signbit(signal[..., 1:]) != np.signbit(signal[..., :-1]),
        axis=-1,
    )

    first_variance = np.var(first_difference, axis=-1)
    second_variance = np.var(second_difference, axis=-1)
    mobility = np.sqrt(_safe_divide(first_variance, variance))
    derivative_mobility = np.sqrt(_safe_divide(second_variance, first_variance))
    complexity = _safe_divide(derivative_mobility, mobility)

    values = np.stack(
        (
            mean,
            variance,
            std,
            rms,
            median,
            mad,
            peak_to_peak,
            skewness,
            excess_kurtosis,
            normalized_line_length,
            zero_crossing_rate,
            mobility,
            complexity,
        ),
        axis=-1,
    )
    return _channel_feature_block("time", values, TIME_FEATURE_NAMES, dtype=dtype)


def extract_spectral_features(
    windows: PreparedFeatureWindows,
    *,
    config: FeatureExtractionConfig,
) -> FeatureBlock:
    if min(band.f_min for band in config.frequency_bands) <= 0:
        raise ValueError("Spectral feature bands must start above 0 Hz")

    fft_config = FFTConfig(
        analysis_sfreq=config.analysis_sfreq,
        f_min=min(band.f_min for band in config.frequency_bands),
        f_max=max(band.f_max for band in config.frequency_bands),
        frequency_step=1.0,
        dtype=config.dtype,
        transform_eog=False,
        filter_hz=None,
        notch_hz=None,
        reference=None,
        normalization="none",
    )
    psd_results = [
        compute_fft_psd(window, source_sfreq=windows.sfreq, config=fft_config)
        for window in windows.values
    ]
    psd = np.stack([result.eeg_power.astype(np.float64, copy=False) for result in psd_results])
    frequencies = psd_results[0].frequencies.astype(np.float64, copy=False)
    bin_edges = _frequency_edges(frequencies)

    band_powers = np.stack(
        [
            _integrate_density(psd, bin_edges=bin_edges, lower=band.f_min, upper=band.f_max)
            for band in config.frequency_bands
        ],
        axis=-1,
    )
    total_lower = min(band.f_min for band in config.frequency_bands)
    total_upper = max(band.f_max for band in config.frequency_bands)
    total_overlap = _interval_overlap(bin_edges, lower=total_lower, upper=total_upper)
    spectral_mass = psd * total_overlap
    total_power = spectral_mass.sum(axis=-1)
    relative_band_powers = _safe_divide(band_powers, total_power[..., np.newaxis])

    active_bins = total_overlap > 0
    active_psd = np.where(active_bins, psd, -np.inf)
    dominant_indices = np.argmax(active_psd, axis=-1)
    dominant_frequency = frequencies[dominant_indices]
    dominant_frequency = np.where(total_power > 0, dominant_frequency, 0.0)
    spectral_centroid = _safe_divide(
        np.sum(spectral_mass * frequencies, axis=-1),
        total_power,
    )
    probabilities = _safe_divide(spectral_mass, total_power[..., np.newaxis])
    entropy_terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    entropy_terms[positive] = -probabilities[positive] * np.log(probabilities[positive])
    active_count = int(np.count_nonzero(active_bins))
    if active_count > 1:
        spectral_entropy = entropy_terms.sum(axis=-1) / np.log(active_count)
    else:
        spectral_entropy = np.zeros_like(total_power)

    values = np.concatenate(
        (
            band_powers,
            relative_band_powers,
            total_power[..., np.newaxis],
            dominant_frequency[..., np.newaxis],
            spectral_centroid[..., np.newaxis],
            spectral_entropy[..., np.newaxis],
        ),
        axis=-1,
    )
    feature_names = (
        *(f"absolute_band_power_{band.name}" for band in config.frequency_bands),
        *(f"relative_band_power_{band.name}" for band in config.frequency_bands),
        *SPECTRAL_SUMMARY_NAMES,
    )
    return _channel_feature_block("spectral", values, feature_names, dtype=config.dtype)


def extract_spatial_features(
    windows: PreparedFeatureWindows,
    *,
    dtype: np.dtype[Any] | str = np.float32,
) -> tuple[FeatureBlock, FeatureBlock, FeatureBlock]:
    if windows.values.shape[-1] < 2:
        raise ValueError("Spatial features require at least two samples per window")

    covariance_matrices: list[NDArray[np.float64]] = []
    correlation_matrices: list[NDArray[np.float64]] = []
    log_covariance_matrices: list[NDArray[np.float64]] = []
    for window in windows.values:
        centered = window - window.mean(axis=-1, keepdims=True)
        if np.all(centered == 0):
            covariance = np.zeros((window.shape[0], window.shape[0]), dtype=np.float64)
        else:
            covariance, _ = oas(window.T, assume_centered=False)
            covariance = np.asarray(covariance, dtype=np.float64)
        covariance = _symmetrize(covariance)
        correlation = _covariance_to_correlation(covariance)
        log_covariance = _symmetric_matrix_logarithm(covariance)

        covariance_matrices.append(covariance)
        correlation_matrices.append(correlation)
        log_covariance_matrices.append(log_covariance)

    output_dtype = np.dtype(dtype)
    return (
        FeatureBlock(
            name="covariance",
            layout="channel_matrix",
            values=np.stack(covariance_matrices).astype(output_dtype, copy=False),
        ),
        FeatureBlock(
            name="correlation",
            layout="channel_matrix",
            values=np.stack(correlation_matrices).astype(output_dtype, copy=False),
        ),
        FeatureBlock(
            name="log_covariance",
            layout="channel_matrix",
            values=np.stack(log_covariance_matrices).astype(output_dtype, copy=False),
        ),
    )


def extract_classical_feature_blocks(
    eeg: ArrayLike,
    *,
    source_sfreq: float,
    config: FeatureExtractionConfig,
) -> tuple[tuple[FeatureBlock, ...], NDArray[np.float64]]:
    windows = prepare_feature_windows(eeg, source_sfreq=source_sfreq, config=config)
    blocks: list[FeatureBlock] = []
    if "time" in config.feature_groups:
        blocks.append(extract_time_features(windows, dtype=config.dtype))
    if "spectral" in config.feature_groups:
        blocks.append(extract_spectral_features(windows, config=config))
    if "spatial" in config.feature_groups:
        blocks.extend(extract_spatial_features(windows, dtype=config.dtype))
    if not blocks:
        raise ValueError("No classical feature groups are enabled")
    return tuple(blocks), windows.bounds_seconds


def _channel_feature_block(
    name: str,
    values: NDArray[np.float64],
    feature_names: tuple[str, ...],
    *,
    dtype: np.dtype[Any] | str,
) -> FeatureBlock:
    if not np.isfinite(values).all():
        raise ValueError(f"Computed {name} features contain non-finite values")
    return FeatureBlock(
        name=name,
        layout="channel_features",
        values=values.astype(np.dtype(dtype), copy=False),
        feature_names=feature_names,
    )


def _safe_divide(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
) -> NDArray[np.float64]:
    output = np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape), dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def _frequency_edges(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("Spectral integration requires at least two frequency bins")
    midpoints = (frequencies[:-1] + frequencies[1:]) / 2.0
    return np.concatenate(
        (
            np.array([frequencies[0] - (midpoints[0] - frequencies[0])]),
            midpoints,
            np.array([frequencies[-1] + (frequencies[-1] - midpoints[-1])]),
        )
    )


def _interval_overlap(
    bin_edges: NDArray[np.float64],
    *,
    lower: float,
    upper: float,
) -> NDArray[np.float64]:
    return np.maximum(
        0.0,
        np.minimum(bin_edges[1:], upper) - np.maximum(bin_edges[:-1], lower),
    )


def _integrate_density(
    psd: NDArray[np.float64],
    *,
    bin_edges: NDArray[np.float64],
    lower: float,
    upper: float,
) -> NDArray[np.float64]:
    overlap = _interval_overlap(bin_edges, lower=lower, upper=upper)
    return np.sum(psd * overlap, axis=-1)


def _covariance_to_correlation(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard_deviations, standard_deviations)
    correlation = _safe_divide(covariance, denominator)
    nonzero_variance = standard_deviations > 0
    diagonal = np.diag_indices_from(correlation)
    correlation[diagonal] = nonzero_variance.astype(np.float64)
    return _symmetrize(np.clip(correlation, -1.0, 1.0))


def _symmetric_matrix_logarithm(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(np.max(eigenvalues))
    if maximum <= 0:
        return np.zeros_like(covariance)
    floor = maximum * 1e-12
    log_eigenvalues = np.log(np.maximum(eigenvalues, floor))
    return _symmetrize((eigenvectors * log_eigenvalues) @ eigenvectors.T)


def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return (matrix + matrix.T) / 2.0
