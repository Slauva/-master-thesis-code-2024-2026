from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import NDArray

from preprocessors.config import PreprocessingMethod, SpectralScaling
from utils.datasets.base import SampleKey
from utils.datasets.schemas import RandomSample


@dataclass(frozen=True, slots=True)
class CropSpectralSample:
    sample: RandomSample
    eeg_power: NDArray[np.floating[Any]]
    frequencies: NDArray[np.floating[Any]]
    times: NDArray[np.floating[Any]] | None
    eeg_channels: tuple[str, ...]
    source_sfreq: float
    analysis_sfreq: float
    method: PreprocessingMethod
    scaling: SpectralScaling
    crop_bounds_seconds: tuple[float, float]

    def __post_init__(self) -> None:
        expected_ndim = 2 if self.method == "fft" else 3
        if self.eeg_power.ndim != expected_ndim:
            raise ValueError(f"{self.method.upper()} power has an invalid dimensionality")
        if self.eeg_power.shape[0] != len(self.eeg_channels):
            raise ValueError("Power channel axis does not match `eeg_channels`")
        if self.frequencies.ndim != 1 or self.eeg_power.shape[1] != self.frequencies.size:
            raise ValueError("Power frequency axis does not match `frequencies`")
        if self.method == "fft":
            if self.times is not None:
                raise ValueError("FFT samples must not have a time axis")
        elif self.times is None or self.times.ndim != 1:
            raise ValueError(f"{self.method.upper()} samples require a one-dimensional time axis")
        elif self.eeg_power.shape[-1] != self.times.size:
            raise ValueError("Power time axis does not match `times`")
        if not np.issubdtype(self.eeg_power.dtype, np.floating):
            raise TypeError("Power must use a floating-point dtype")
        if self.frequencies.dtype != self.eeg_power.dtype:
            raise TypeError("Power and frequencies must use the same dtype")
        if self.times is not None and self.times.dtype != self.eeg_power.dtype:
            raise TypeError("Power and times must use the same dtype")
        if not np.isfinite(self.eeg_power).all() or np.any(self.eeg_power < 0):
            raise ValueError("Power must contain finite non-negative values")
        if not np.isfinite(self.frequencies).all() or np.any(np.diff(self.frequencies) <= 0):
            raise ValueError("Frequencies must be finite and strictly increasing")
        if self.times is not None and (
            not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("Times must be finite and strictly increasing")
        if self.source_sfreq <= 0 or self.analysis_sfreq <= 0:
            raise ValueError("Sampling frequencies must be positive")
        if self.crop_bounds_seconds[1] <= self.crop_bounds_seconds[0]:
            raise ValueError("Crop bounds must define a positive-duration interval")
        self.eeg_power.setflags(write=False)
        self.frequencies.setflags(write=False)
        if self.times is not None:
            self.times.setflags(write=False)

    @property
    def sample_key(self) -> SampleKey:
        return self.sample.subject_id, self.sample.trial_number, self.sample.block_index


@dataclass(frozen=True, slots=True)
class SpectralNormalizationState:
    method: PreprocessingMethod
    scaling: SpectralScaling
    frequencies: NDArray[np.floating[Any]]
    eeg_channels: tuple[str, ...]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    zero_variance_mask: NDArray[np.bool_]
    fit_sample_keys: tuple[SampleKey, ...]
    observation_count: int
    crop_bounds_seconds: tuple[float, float]
    log_epsilon: float
    std_epsilon: float

    def __post_init__(self) -> None:
        n_frequencies = self.frequencies.size
        if self.frequencies.ndim != 1 or n_frequencies < 1:
            raise ValueError("Normalization frequencies must be a non-empty vector")
        if not np.isfinite(self.frequencies).all() or np.any(np.diff(self.frequencies) <= 0):
            raise ValueError("Normalization frequencies must be finite and strictly increasing")
        for name, values, dtype in (
            ("mean", self.mean, np.dtype(np.float64)),
            ("scale", self.scale, np.dtype(np.float64)),
            ("zero_variance_mask", self.zero_variance_mask, np.dtype(np.bool_)),
        ):
            if values.shape != (n_frequencies,) or values.dtype != dtype:
                raise TypeError(f"`{name}` must match the frequency axis and use {dtype.name}")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("Normalization statistics must be finite")
        if np.any(self.scale <= 0):
            raise ValueError("Normalization scales must be positive")
        if not self.eeg_channels or len(set(self.eeg_channels)) != len(self.eeg_channels):
            raise ValueError("EEG channels must be non-empty and unique")
        if not self.fit_sample_keys or len(set(self.fit_sample_keys)) != len(self.fit_sample_keys):
            raise ValueError("Fit sample keys must be non-empty and unique")
        if self.observation_count < 1:
            raise ValueError("Normalization requires at least one observation")
        if self.log_epsilon <= 0 or self.std_epsilon <= 0:
            raise ValueError("Normalization epsilons must be positive")
        self.frequencies.setflags(write=False)
        self.mean.setflags(write=False)
        self.scale.setflags(write=False)
        self.zero_variance_mask.setflags(write=False)


@dataclass(frozen=True, slots=True)
class TorchSpectralInputSample:
    sample_key: SampleKey
    target_row_index: int
    model_input: torch.Tensor
    target: torch.Tensor
    frequencies: torch.Tensor
    times: torch.Tensor | None
    eeg_channels: tuple[str, ...]
    method: PreprocessingMethod
    scaling: SpectralScaling


@dataclass(frozen=True, slots=True)
class TorchSpectralInputBatch:
    sample_keys: tuple[SampleKey, ...]
    target_row_indices: torch.Tensor
    model_inputs: torch.Tensor
    targets: torch.Tensor
    frequencies: torch.Tensor
    times: torch.Tensor | None
    eeg_channels: tuple[str, ...]
    method: PreprocessingMethod
    scaling: SpectralScaling

    def pin_memory(self) -> Self:
        return replace(
            self,
            target_row_indices=self.target_row_indices.pin_memory(),
            model_inputs=self.model_inputs.pin_memory(),
            targets=self.targets.pin_memory(),
            frequencies=self.frequencies.pin_memory(),
            times=None if self.times is None else self.times.pin_memory(),
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        return replace(
            self,
            target_row_indices=self.target_row_indices.to(
                device=device,
                non_blocking=non_blocking,
            ),
            model_inputs=self.model_inputs.to(device=device, non_blocking=non_blocking),
            targets=self.targets.to(device=device, non_blocking=non_blocking),
            frequencies=self.frequencies.to(device=device, non_blocking=non_blocking),
            times=None
            if self.times is None
            else self.times.to(device=device, non_blocking=non_blocking),
        )
