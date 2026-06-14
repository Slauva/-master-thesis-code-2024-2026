from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal, Self, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class RawSample(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    subject_id: int = Field(ge=1)
    trial_number: int = Field(ge=1)
    block_index: int = Field(alias="Exec_Block_Index", ge=1)
    eeg_path: Path
    eog_path: Path
    img: list[list[int]]


class GeometricSample(RawSample):
    type: Literal["geometric"] = "geometric"
    pattern_id: int


class RandomSample(RawSample):
    type: Literal["random"] = "random"
    seed: int


Sample = Annotated[Union[GeometricSample, RandomSample], Field(discriminator="type")]


class LabelModel(BaseModel):
    blocks: list[Sample]


@dataclass(frozen=True, slots=True)
class LoadedSample:
    sample: Sample
    eeg: NDArray[np.floating[Any]]
    eog: NDArray[np.floating[Any]]
    sfreq: float
    eeg_channels: tuple[str, ...]
    eog_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpectralSample:
    sample: Sample
    eeg_power: NDArray[np.floating[Any]]
    eog: NDArray[np.floating[Any]]
    frequencies: NDArray[np.floating[Any]]
    times: NDArray[np.floating[Any]] | None
    eeg_channels: tuple[str, ...]
    eog_channels: tuple[str, ...]
    source_sfreq: float
    analysis_sfreq: float
    method: Literal["fft", "morlet", "superlet", "stft"]
    scaling: Literal["psd", "wavelet_power"]


@dataclass(frozen=True, slots=True)
class TorchSample:
    sample: Sample
    eeg: torch.Tensor
    eog: torch.Tensor
    sfreq: float
    eeg_channels: tuple[str, ...]
    eog_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TorchSpectralSample:
    sample: Sample
    eeg_power: torch.Tensor
    eog: torch.Tensor
    frequencies: torch.Tensor
    times: torch.Tensor | None
    eeg_channels: tuple[str, ...]
    eog_channels: tuple[str, ...]
    source_sfreq: float
    analysis_sfreq: float
    method: Literal["fft", "morlet", "superlet", "stft"]
    scaling: Literal["psd", "wavelet_power"]


@dataclass(frozen=True, slots=True)
class TorchSampleBatch:
    samples: tuple[Sample, ...]
    eeg: torch.Tensor
    eog: torch.Tensor
    lengths: torch.Tensor
    time_mask: torch.Tensor
    eog_finite_mask: torch.Tensor
    sfreq: float
    eeg_channels: tuple[str, ...]
    eog_channels: tuple[str, ...]

    def pin_memory(self) -> Self:
        return replace(
            self,
            eeg=self.eeg.pin_memory(),
            eog=self.eog.pin_memory(),
            lengths=self.lengths.pin_memory(),
            time_mask=self.time_mask.pin_memory(),
            eog_finite_mask=self.eog_finite_mask.pin_memory(),
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        return replace(
            self,
            eeg=self.eeg.to(device=device, non_blocking=non_blocking),
            eog=self.eog.to(device=device, non_blocking=non_blocking),
            lengths=self.lengths.to(device=device, non_blocking=non_blocking),
            time_mask=self.time_mask.to(device=device, non_blocking=non_blocking),
            eog_finite_mask=self.eog_finite_mask.to(device=device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class TorchSpectralBatch:
    samples: tuple[Sample, ...]
    eeg_power: torch.Tensor
    eog: torch.Tensor
    frequencies: torch.Tensor
    times: torch.Tensor | None
    spectral_lengths: torch.Tensor | None
    spectral_time_mask: torch.Tensor | None
    eog_lengths: torch.Tensor
    eog_time_mask: torch.Tensor
    eog_finite_mask: torch.Tensor
    eeg_channels: tuple[str, ...]
    eog_channels: tuple[str, ...]
    source_sfreq: float
    analysis_sfreq: float
    method: Literal["fft", "morlet", "superlet", "stft"]
    scaling: Literal["psd", "wavelet_power"]

    def pin_memory(self) -> Self:
        return replace(
            self,
            eeg_power=self.eeg_power.pin_memory(),
            eog=self.eog.pin_memory(),
            frequencies=self.frequencies.pin_memory(),
            times=None if self.times is None else self.times.pin_memory(),
            spectral_lengths=None if self.spectral_lengths is None else self.spectral_lengths.pin_memory(),
            spectral_time_mask=(
                None if self.spectral_time_mask is None else self.spectral_time_mask.pin_memory()
            ),
            eog_lengths=self.eog_lengths.pin_memory(),
            eog_time_mask=self.eog_time_mask.pin_memory(),
            eog_finite_mask=self.eog_finite_mask.pin_memory(),
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        return replace(
            self,
            eeg_power=self.eeg_power.to(device=device, non_blocking=non_blocking),
            eog=self.eog.to(device=device, non_blocking=non_blocking),
            frequencies=self.frequencies.to(device=device, non_blocking=non_blocking),
            times=None if self.times is None else self.times.to(device=device, non_blocking=non_blocking),
            spectral_lengths=(
                None
                if self.spectral_lengths is None
                else self.spectral_lengths.to(device=device, non_blocking=non_blocking)
            ),
            spectral_time_mask=(
                None
                if self.spectral_time_mask is None
                else self.spectral_time_mask.to(device=device, non_blocking=non_blocking)
            ),
            eog_lengths=self.eog_lengths.to(device=device, non_blocking=non_blocking),
            eog_time_mask=self.eog_time_mask.to(device=device, non_blocking=non_blocking),
            eog_finite_mask=self.eog_finite_mask.to(device=device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class CacheWarmupError:
    key: tuple[int, int, int]
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class CacheWarmupReport:
    processed: int
    cached: int
    skipped: int
    failed: int
    errors: tuple[CacheWarmupError, ...]
    max_workers: int
    duration_seconds: float

    @property
    def total(self) -> int:
        return self.processed + self.cached + self.skipped + self.failed
