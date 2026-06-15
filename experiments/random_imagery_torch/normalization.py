from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from experiments.random_imagery_torch.schemas import (
    CropSpectralSample,
    SpectralNormalizationState,
)
from experiments.random_imagery_torch.spectral_dataset import CropSpectralDataset
from utils.datasets.base import SampleKey


def fit_spectral_normalization(
    dataset: CropSpectralDataset,
    sample_keys: Sequence[SampleKey],
) -> SpectralNormalizationState:
    keys = tuple(sample_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Normalization fit sample keys must be non-empty and unique")

    total: NDArray[np.float64] | None = None
    total_squared: NDArray[np.float64] | None = None
    observation_count = 0
    reference: CropSpectralSample | None = None
    for key in keys:
        sample = dataset[key]
        if sample.sample_key != key:
            raise ValueError("Spectral sample key does not match the requested fit key")
        if reference is None:
            reference = sample
            total = np.zeros(sample.frequencies.size, dtype=np.float64)
            total_squared = np.zeros(sample.frequencies.size, dtype=np.float64)
        else:
            _validate_sample_compatibility(reference, sample)

        log_power = _log_power(sample, epsilon=dataset.input_config.log_epsilon)
        frequency_rows = _frequency_rows(log_power)
        if total is None or total_squared is None:
            raise RuntimeError("Normalization accumulators were not initialized")
        total += frequency_rows.sum(axis=1, dtype=np.float64)
        total_squared += np.square(frequency_rows).sum(axis=1, dtype=np.float64)
        observation_count += frequency_rows.shape[1]

    if reference is None or total is None or total_squared is None:
        raise RuntimeError("Normalization fit did not process any samples")
    mean = total / observation_count
    variance = np.maximum(total_squared / observation_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    zero_variance_mask = std < dataset.input_config.std_epsilon
    scale = np.where(zero_variance_mask, 1.0, std)
    return SpectralNormalizationState(
        method=reference.method,
        scaling=reference.scaling,
        frequencies=reference.frequencies.copy(),
        eeg_channels=reference.eeg_channels,
        mean=mean,
        scale=scale,
        zero_variance_mask=zero_variance_mask,
        fit_sample_keys=keys,
        observation_count=observation_count,
        crop_bounds_seconds=reference.crop_bounds_seconds,
        log_epsilon=dataset.input_config.log_epsilon,
        std_epsilon=dataset.input_config.std_epsilon,
    )


def normalize_spectral_sample(
    sample: CropSpectralSample,
    state: SpectralNormalizationState,
) -> NDArray[np.float32]:
    _validate_state_compatibility(sample, state)
    log_power = _log_power(sample, epsilon=state.log_epsilon)
    if sample.method == "fft":
        normalized = (log_power - state.mean[np.newaxis, :]) / state.scale[np.newaxis, :]
        model_input = normalized[np.newaxis, :, :]
    else:
        normalized = (
            log_power - state.mean[np.newaxis, :, np.newaxis]
        ) / state.scale[np.newaxis, :, np.newaxis]
        model_input = np.transpose(normalized, (1, 0, 2))
    output = np.asarray(model_input, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Normalized model input contains non-finite values")
    output.setflags(write=False)
    return output


def _log_power(
    sample: CropSpectralSample,
    *,
    epsilon: float,
) -> NDArray[np.float64]:
    power = np.asarray(sample.eeg_power, dtype=np.float64)
    return np.log(np.maximum(power, epsilon))


def _frequency_rows(log_power: NDArray[np.float64]) -> NDArray[np.float64]:
    if log_power.ndim == 2:
        return np.transpose(log_power, (1, 0))
    return np.transpose(log_power, (1, 0, 2)).reshape(log_power.shape[1], -1)


def _validate_sample_compatibility(
    reference: CropSpectralSample,
    sample: CropSpectralSample,
) -> None:
    if sample.method != reference.method or sample.scaling != reference.scaling:
        raise ValueError("Normalization samples must share method and scaling")
    if sample.eeg_channels != reference.eeg_channels:
        raise ValueError("Normalization samples must share EEG channel order")
    if not np.array_equal(sample.frequencies, reference.frequencies):
        raise ValueError("Normalization samples must share the frequency grid")
    if sample.eeg_power.shape != reference.eeg_power.shape:
        raise ValueError("Normalization samples must share the exact spectral shape")
    if sample.crop_bounds_seconds != reference.crop_bounds_seconds:
        raise ValueError("Normalization samples must share crop bounds")
    if (sample.times is None) != (reference.times is None):
        raise ValueError("Normalization samples must share time-axis presence")
    if sample.times is not None and not np.array_equal(sample.times, reference.times):
        raise ValueError("Normalization samples must share the exact time axis")


def _validate_state_compatibility(
    sample: CropSpectralSample,
    state: SpectralNormalizationState,
) -> None:
    if sample.method != state.method or sample.scaling != state.scaling:
        raise ValueError("Normalization state method or scaling does not match the sample")
    if sample.eeg_channels != state.eeg_channels:
        raise ValueError("Normalization state EEG channels do not match the sample")
    if not np.array_equal(sample.frequencies, state.frequencies):
        raise ValueError("Normalization state frequency grid does not match the sample")
    if sample.crop_bounds_seconds != state.crop_bounds_seconds:
        raise ValueError("Normalization state crop bounds do not match the sample")
