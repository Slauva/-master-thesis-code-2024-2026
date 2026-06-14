"""Adaptive Superlet transform and project-specific power wrapper.

The core transform is adapted from Gregor Mönke's Python implementation in
https://github.com/tensionhead/Superlets at commit
20f6bfdf31b783b4d8254546effa8f27784118a2. That repository is a fork of the
reference Superlets repository from the Transylvanian Institute of
Neuroscience and is distributed under the MIT License. See
``preprocessors/SUPERLET_LICENSE.txt``.

Algorithm reference:
Moca et al. (2021), "Time-frequency super-resolution with superlets",
Nature Communications.

SPDX-License-Identifier: MIT
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve

from preprocessors._signal import prepare_eeg
from preprocessors._time_frequency import trim_and_bin_power
from preprocessors.config import SuperletConfig, build_frequency_grid
from preprocessors.schemas import SpectralTransformResult


@dataclass(frozen=True, slots=True)
class SuperletMorlet:
    cycles: float
    gaussian_sd: float = 5.0

    def __call__(
        self,
        time: NDArray[np.float64],
        scale: float,
    ) -> NDArray[np.complex128]:
        scaled_time = time / scale
        normalization = self.gaussian_sd / (
            scale * self.cycles * (2.0 * np.pi) ** 1.5
        )
        oscillation = np.exp(1j * scaled_time)
        envelope = np.exp(
            -0.5
            * (
                self.gaussian_sd
                * scaled_time
                / (2.0 * np.pi * self.cycles)
            )
            ** 2
        )
        return normalization * oscillation * envelope


def compute_superlet_power(
    eeg: ArrayLike,
    *,
    source_sfreq: float,
    config: SuperletConfig,
) -> SpectralTransformResult:
    """Compute edge-trimmed, time-binned adaptive Superlet power."""
    signal = prepare_eeg(
        eeg,
        source_sfreq=source_sfreq,
        target_sfreq=config.analysis_sfreq,
    )
    frequencies = build_frequency_grid(config)
    scales = scale_from_period(1.0 / frequencies)
    edge_samples = superlet_edge_samples(
        frequencies,
        sfreq=config.analysis_sfreq,
        order_min=config.order_min,
        order_max=config.order_max,
        c_1=config.c_1,
    )
    minimum_samples = 2 * edge_samples + config.time_bin_samples
    if signal.shape[-1] < minimum_samples:
        raise ValueError(
            "Superlet preprocessing requires at least "
            f"{minimum_samples} resampled samples for edge trimming and one time bin"
        )

    coefficients = superlet(
        signal.T,
        samplerate=config.analysis_sfreq,
        scales=scales,
        order_max=config.order_max,
        order_min=config.order_min,
        c_1=config.c_1,
        adaptive=config.adaptive,
    )
    power = np.square(np.abs(coefficients)).transpose(2, 0, 1)
    binned_power, times = trim_and_bin_power(
        power,
        sfreq=config.analysis_sfreq,
        edge_samples=edge_samples,
        bin_samples=config.time_bin_samples,
        method="Superlet",
    )
    output_dtype = np.dtype(config.dtype)
    return SpectralTransformResult(
        eeg_power=binned_power.astype(output_dtype, copy=False),
        frequencies=frequencies.astype(output_dtype, copy=False),
        times=times.astype(output_dtype, copy=False),
        analysis_sfreq=config.analysis_sfreq,
        scaling=config.scaling,
    )


def superlet(
    data: ArrayLike,
    samplerate: float,
    scales: ArrayLike,
    order_max: int,
    order_min: int = 1,
    c_1: int = 3,
    adaptive: bool = False,
) -> NDArray[np.complex128]:
    """Compute a multiplicative or fractional adaptive Superlet transform.

    The first input dimension is time. For input shape ``(time, channel)``,
    the output shape is ``(frequency, time, channel)``.
    """
    signal = np.asarray(data, dtype=np.float64)
    scale_array = np.asarray(scales, dtype=np.float64)
    _validate_superlet_inputs(
        signal,
        samplerate=samplerate,
        scales=scale_array,
        order_min=order_min,
        order_max=order_max,
        c_1=c_1,
    )
    if adaptive:
        return fractional_adaptive_superlet_transform(
            signal,
            samplerate=samplerate,
            scales=scale_array,
            order_min=order_min,
            order_max=order_max,
            c_1=c_1,
        )
    return multiplicative_superlet_transform(
        signal,
        samplerate=samplerate,
        scales=scale_array,
        order_min=order_min,
        order_max=order_max,
        c_1=c_1,
    )


def multiplicative_superlet_transform(
    data: NDArray[np.float64],
    *,
    samplerate: float,
    scales: NDArray[np.float64],
    order_min: int,
    order_max: int,
    c_1: int,
) -> NDArray[np.complex128]:
    dt = 1.0 / samplerate
    cycles = c_1 * np.arange(order_min, order_max + 1)
    order_count = order_max - order_min + 1
    wavelets = [SuperletMorlet(float(cycle_count)) for cycle_count in cycles]

    geometric_mean = continuous_wavelet_transform(data, wavelets[0], scales, dt)
    geometric_mean = np.power(geometric_mean, 1.0 / order_count)
    for wavelet in wavelets[1:]:
        coefficients = continuous_wavelet_transform(data, wavelet, scales, dt)
        geometric_mean *= np.power(coefficients, 1.0 / order_count)
    return geometric_mean


def fractional_adaptive_superlet_transform(
    data: NDArray[np.float64],
    *,
    samplerate: float,
    scales: NDArray[np.float64],
    order_min: int,
    order_max: int,
    c_1: int,
) -> NDArray[np.complex128]:
    dt = 1.0 / samplerate
    frequencies = 1.0 / (2.0 * np.pi * scales)
    orders = compute_adaptive_order(frequencies, order_min, order_max)
    integer_orders = np.floor(orders).astype(np.int32)
    cycles = c_1 * np.unique(integer_orders)
    wavelets = [SuperletMorlet(float(cycle_count)) for cycle_count in cycles]
    exponents = 1.0 / (orders - order_min + 1.0)
    order_jumps = np.flatnonzero(np.diff(integer_orders))
    fractions = orders - integer_orders

    geometric_mean = continuous_wavelet_transform(data, wavelets[0], scales, dt)
    geometric_mean = _frequency_power(geometric_mean, exponents)
    last_jump = 1

    for wavelet_index, jump in enumerate(order_jumps):
        remaining_scales = scales[last_jump:]
        next_coefficients = continuous_wavelet_transform(
            data,
            wavelets[wavelet_index + 1],
            remaining_scales,
            dt,
        )

        fractional_span = slice(last_jump, jump + 1)
        fractional_count = jump - last_jump + 1
        geometric_mean[fractional_span] *= _frequency_power(
            next_coefficients[:fractional_count],
            fractions[fractional_span] * exponents[fractional_span],
        )
        geometric_mean[jump + 1 :] *= _frequency_power(
            next_coefficients[fractional_count:],
            exponents[jump + 1 :],
        )
        last_jump = jump + 1
    return geometric_mean


def continuous_wavelet_transform(
    data: NDArray[np.float64],
    wavelet: SuperletMorlet,
    scales: NDArray[np.float64],
    dt: float,
) -> NDArray[np.complex128]:
    output = np.empty((scales.size, *data.shape), dtype=np.complex128)
    wavelet_axes = (slice(None),) + (None,) * (data.ndim - 1)
    for index, scale in enumerate(scales):
        support = superlet_support(float(scale), dt, wavelet.cycles)
        normalization = np.sqrt(dt) / (4.0 * np.pi)
        sampled_wavelet = normalization * wavelet(support, float(scale))
        output[index] = fftconvolve(
            data,
            sampled_wavelet[wavelet_axes],
            mode="same",
            axes=0,
        )
    return output


def compute_adaptive_order(
    frequencies: ArrayLike,
    order_min: int,
    order_max: int,
) -> NDArray[np.float64]:
    frequency_array = np.asarray(frequencies, dtype=np.float64)
    if frequency_array.ndim != 1 or frequency_array.size < 2:
        raise ValueError(
            "Adaptive Superlet frequencies must be a one-dimensional array "
            "with at least two values"
        )
    if not np.isfinite(frequency_array).all() or np.any(np.diff(frequency_array) <= 0):
        raise ValueError("Adaptive Superlet frequencies must be finite and strictly increasing")
    frequency_min = frequency_array[0]
    frequency_max = frequency_array[-1]
    scaled_order = (order_max - order_min) * (
        frequency_array - frequency_min
    ) / (frequency_max - frequency_min)
    return order_min + scaled_order


def superlet_edge_samples(
    frequencies: ArrayLike,
    *,
    sfreq: float,
    order_min: int,
    order_max: int,
    c_1: int,
) -> int:
    frequency_array = np.asarray(frequencies, dtype=np.float64)
    orders = compute_adaptive_order(frequency_array, order_min, order_max)
    scales = scale_from_period(1.0 / frequency_array)
    longest_support = max(
        superlet_support(
            float(scale),
            1.0 / sfreq,
            float(c_1 * np.ceil(order)),
        ).size
        for scale, order in zip(scales, orders, strict=True)
    )
    return longest_support // 2


def superlet_support(
    scale: float,
    dt: float,
    cycles: float,
) -> NDArray[np.float64]:
    sample_span = 10.0 * scale * cycles / dt
    return np.arange(
        (-sample_span + 1.0) / 2.0,
        (sample_span + 1.0) / 2.0,
        dtype=np.float64,
    ) * dt


def fourier_period(scale: ArrayLike) -> NDArray[np.float64]:
    return 2.0 * np.pi * np.asarray(scale, dtype=np.float64)


def scale_from_period(period: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(period, dtype=np.float64) / (2.0 * np.pi)


def _frequency_power(
    values: NDArray[np.complex128],
    exponents: NDArray[np.float64],
) -> NDArray[np.complex128]:
    exponent_shape = (exponents.size,) + (1,) * (values.ndim - 1)
    return np.power(values, exponents.reshape(exponent_shape))


def _validate_superlet_inputs(
    data: NDArray[np.float64],
    *,
    samplerate: float,
    scales: NDArray[np.float64],
    order_min: int,
    order_max: int,
    c_1: int,
) -> None:
    if data.ndim < 1 or data.shape[0] < 2:
        raise ValueError("Superlet input must have a time axis with at least two samples")
    if not np.isfinite(data).all():
        raise ValueError("Superlet input must contain only finite values")
    if not np.isfinite(samplerate) or samplerate <= 0:
        raise ValueError("Superlet samplerate must be finite and positive")
    if scales.ndim != 1 or scales.size < 2:
        raise ValueError("Superlet scales must be a one-dimensional array with at least two values")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Superlet scales must be finite and positive")
    if order_min < 1 or order_max < order_min:
        raise ValueError("Superlet orders must satisfy 1 <= order_min <= order_max")
    if c_1 < 1 or c_1 * order_min < 3:
        raise ValueError("The minimum Superlet cycle count must be at least 3")


__all__ = [
    "SuperletMorlet",
    "compute_adaptive_order",
    "compute_superlet_power",
    "continuous_wavelet_transform",
    "fourier_period",
    "fractional_adaptive_superlet_transform",
    "multiplicative_superlet_transform",
    "scale_from_period",
    "superlet",
    "superlet_edge_samples",
    "superlet_support",
]
