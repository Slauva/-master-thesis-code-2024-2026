from collections.abc import Iterator, Sequence
from math import isclose

import torch
from torch.utils.data import Dataset

from utils.datasets.base import SampleKey, SourceMap
from utils.datasets.preprocessed import PreprocessedDataset
from utils.datasets.schemas import Sample, TorchSpectralBatch, TorchSpectralSample


class TorchPreprocessedDataset(Dataset[TorchSpectralSample]):
    def __init__(self, source_dataset: PreprocessedDataset):
        if not isinstance(source_dataset, PreprocessedDataset):
            raise TypeError("`source_dataset` must be a PreprocessedDataset")
        self.source_dataset = source_dataset

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, key: int | SampleKey) -> TorchSpectralSample:
        spectral = self.source_dataset[key]
        return TorchSpectralSample(
            sample=spectral.sample,
            eeg_power=torch.from_numpy(spectral.eeg_power),
            eog=torch.from_numpy(spectral.eog),
            frequencies=torch.from_numpy(spectral.frequencies),
            times=None if spectral.times is None else torch.from_numpy(spectral.times),
            eeg_channels=spectral.eeg_channels,
            eog_channels=spectral.eog_channels,
            source_sfreq=spectral.source_sfreq,
            analysis_sfreq=spectral.analysis_sfreq,
            method=spectral.method,
            scaling=spectral.scaling,
        )

    def __iter__(self) -> Iterator[TorchSpectralSample]:
        for index in range(len(self)):
            yield self[index]

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self.source_dataset.samples

    @property
    def source_map(self) -> SourceMap:
        return self.source_dataset.source_map


def collate_torch_spectral_samples(batch: Sequence[TorchSpectralSample]) -> TorchSpectralBatch:
    if not batch:
        raise ValueError("Cannot collate an empty spectral batch")

    reference = batch[0]
    _validate_spectral_sample(reference)
    for sample in batch[1:]:
        _validate_spectral_sample(sample)
        _validate_spectral_compatibility(reference, sample)

    eeg_power, times, spectral_lengths, spectral_time_mask = _collate_power(batch)
    eog, eog_lengths, eog_time_mask, eog_finite_mask = _collate_eog(batch)
    return TorchSpectralBatch(
        samples=tuple(sample.sample for sample in batch),
        eeg_power=eeg_power,
        eog=eog,
        frequencies=reference.frequencies,
        times=times,
        spectral_lengths=spectral_lengths,
        spectral_time_mask=spectral_time_mask,
        eog_lengths=eog_lengths,
        eog_time_mask=eog_time_mask,
        eog_finite_mask=eog_finite_mask,
        eeg_channels=reference.eeg_channels,
        eog_channels=reference.eog_channels,
        source_sfreq=reference.source_sfreq,
        analysis_sfreq=reference.analysis_sfreq,
        method=reference.method,
        scaling=reference.scaling,
    )


def _collate_power(
    batch: Sequence[TorchSpectralSample],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    reference = batch[0]
    if reference.method == "fft":
        return torch.stack([sample.eeg_power for sample in batch]), None, None, None

    spectral_lengths = torch.tensor([sample.eeg_power.shape[-1] for sample in batch], dtype=torch.int64)
    max_length = int(spectral_lengths.max().item())
    eeg_power = reference.eeg_power.new_zeros(
        (len(batch), reference.eeg_power.shape[0], reference.eeg_power.shape[1], max_length)
    )
    times = reference.times.new_zeros((len(batch), max_length)) if reference.times is not None else None
    for index, sample in enumerate(batch):
        length = sample.eeg_power.shape[-1]
        eeg_power[index, :, :, :length] = sample.eeg_power
        if times is not None and sample.times is not None:
            times[index, :length] = sample.times

    spectral_time_mask = (
        torch.arange(max_length, device=spectral_lengths.device).unsqueeze(0)
        < spectral_lengths.unsqueeze(1)
    )
    return eeg_power, times, spectral_lengths, spectral_time_mask


def _collate_eog(
    batch: Sequence[TorchSpectralSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reference = batch[0]
    eog_lengths = torch.tensor([sample.eog.shape[-1] for sample in batch], dtype=torch.int64)
    max_length = int(eog_lengths.max().item())
    eog = reference.eog.new_zeros((len(batch), reference.eog.shape[0], max_length))
    eog_finite_mask = torch.zeros(eog.shape, dtype=torch.bool, device=eog.device)
    for index, sample in enumerate(batch):
        length = sample.eog.shape[-1]
        eog[index, :, :length] = sample.eog
        eog_finite_mask[index, :, :length] = torch.isfinite(sample.eog)

    eog_time_mask = (
        torch.arange(max_length, device=eog_lengths.device).unsqueeze(0) < eog_lengths.unsqueeze(1)
    )
    return eog, eog_lengths, eog_time_mask, eog_finite_mask


def _validate_spectral_sample(sample: TorchSpectralSample) -> None:
    expected_ndim = 2 if sample.method == "fft" else 3
    if sample.eeg_power.ndim != expected_ndim:
        expected_shape = "(channel, frequency)" if expected_ndim == 2 else "(channel, frequency, time)"
        raise ValueError(f"{sample.method.upper()} power tensor must have shape {expected_shape}")
    if sample.eog.ndim != 2:
        raise ValueError("EOG tensor must have shape (channel, time)")
    if sample.frequencies.ndim != 1:
        raise ValueError("Frequency tensor must be one-dimensional")
    if sample.eeg_power.shape[0] != len(sample.eeg_channels):
        raise ValueError("Power tensor channel axis does not match `eeg_channels`")
    if sample.eog.shape[0] != len(sample.eog_channels):
        raise ValueError("EOG tensor channel axis does not match `eog_channels`")
    if sample.eeg_power.shape[1] != sample.frequencies.numel():
        raise ValueError("Power tensor frequency axis does not match `frequencies`")
    if sample.method == "fft":
        if sample.times is not None:
            raise ValueError("FFT samples must not have a time axis")
    elif sample.times is None or sample.times.ndim != 1:
        raise ValueError(f"{sample.method.upper()} samples require a one-dimensional time axis")
    elif sample.eeg_power.shape[-1] != sample.times.numel():
        raise ValueError("Power tensor time axis does not match `times`")

    tensors = [sample.eeg_power, sample.eog, sample.frequencies]
    if sample.times is not None:
        tensors.append(sample.times)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("Spectral samples must remain on CPU until after collation")
    if sample.eeg_power.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"Power tensor must use float32 or float64, got {sample.eeg_power.dtype}")
    if sample.frequencies.dtype != sample.eeg_power.dtype:
        raise TypeError("Power and frequency tensor dtypes differ")
    if sample.times is not None and sample.times.dtype != sample.eeg_power.dtype:
        raise TypeError("Power and time tensor dtypes differ")
    if sample.eog.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"EOG tensor must use float32 or float64, got {sample.eog.dtype}")
    if not torch.isfinite(sample.eeg_power).all() or torch.any(sample.eeg_power < 0):
        raise ValueError("Power tensor must contain finite non-negative values")
    if not torch.isfinite(sample.frequencies).all() or torch.any(torch.diff(sample.frequencies) <= 0):
        raise ValueError("Frequencies must be finite and strictly increasing")
    if sample.times is not None and (
        not torch.isfinite(sample.times).all() or torch.any(torch.diff(sample.times) <= 0)
    ):
        raise ValueError("Times must be finite and strictly increasing")


def _validate_spectral_compatibility(
    reference: TorchSpectralSample,
    sample: TorchSpectralSample,
) -> None:
    if sample.method != reference.method:
        raise ValueError("Cannot collate samples from different preprocessing methods")
    if sample.scaling != reference.scaling:
        raise ValueError("Cannot collate samples with different spectral scaling")
    if sample.eeg_channels != reference.eeg_channels:
        raise ValueError("Cannot collate samples with different EEG channels")
    if sample.eog_channels != reference.eog_channels:
        raise ValueError("Cannot collate samples with different EOG channels")
    if not isclose(sample.source_sfreq, reference.source_sfreq, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Cannot collate samples with different source sampling frequencies")
    if not isclose(sample.analysis_sfreq, reference.analysis_sfreq, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Cannot collate samples with different analysis sampling frequencies")
    if sample.eeg_power.dtype != reference.eeg_power.dtype or sample.eog.dtype != reference.eog.dtype:
        raise TypeError("Cannot collate samples with different tensor dtypes")
    if not torch.equal(sample.frequencies, reference.frequencies):
        raise ValueError("Cannot collate samples with different frequency grids")
