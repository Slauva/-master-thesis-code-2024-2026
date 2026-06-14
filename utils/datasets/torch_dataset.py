from collections.abc import Iterator, Sequence
from math import isclose

import torch
from torch.utils.data import Dataset

from utils.datasets.base import SampleKey, SourceMap
from utils.datasets.numpy_dataset import NumpyDataset
from utils.datasets.schemas import Sample, TorchSample, TorchSampleBatch


class TorchDataset(Dataset[TorchSample]):
    def __init__(self, source_dataset: NumpyDataset):
        if not isinstance(source_dataset, NumpyDataset):
            raise TypeError("`source_dataset` must be a NumpyDataset")
        self.source_dataset = source_dataset

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, key: int | SampleKey) -> TorchSample:
        loaded = self.source_dataset[key]
        return TorchSample(
            sample=loaded.sample,
            eeg=torch.from_numpy(loaded.eeg),
            eog=torch.from_numpy(loaded.eog),
            sfreq=loaded.sfreq,
            eeg_channels=loaded.eeg_channels,
            eog_channels=loaded.eog_channels,
        )

    def __iter__(self) -> Iterator[TorchSample]:
        for index in range(len(self)):
            yield self[index]

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self.source_dataset.samples

    @property
    def source_map(self) -> SourceMap:
        return self.source_dataset.source_map


def collate_torch_samples(batch: Sequence[TorchSample]) -> TorchSampleBatch:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    reference = batch[0]
    _validate_raw_sample(reference)
    for sample in batch[1:]:
        _validate_raw_sample(sample)
        _validate_raw_compatibility(reference, sample)

    lengths = torch.tensor([sample.eeg.shape[-1] for sample in batch], dtype=torch.int64)
    max_length = int(lengths.max().item())
    eeg = reference.eeg.new_zeros((len(batch), reference.eeg.shape[0], max_length))
    eog = reference.eog.new_zeros((len(batch), reference.eog.shape[0], max_length))
    eog_finite_mask = torch.zeros(eog.shape, dtype=torch.bool, device=eog.device)

    for index, sample in enumerate(batch):
        length = sample.eeg.shape[-1]
        eeg[index, :, :length] = sample.eeg
        eog[index, :, :length] = sample.eog
        eog_finite_mask[index, :, :length] = torch.isfinite(sample.eog)

    time_mask = torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    return TorchSampleBatch(
        samples=tuple(sample.sample for sample in batch),
        eeg=eeg,
        eog=eog,
        lengths=lengths,
        time_mask=time_mask,
        eog_finite_mask=eog_finite_mask,
        sfreq=reference.sfreq,
        eeg_channels=reference.eeg_channels,
        eog_channels=reference.eog_channels,
    )


def _validate_raw_sample(sample: TorchSample) -> None:
    if sample.eeg.ndim != 2:
        raise ValueError("EEG tensor must have shape (channel, time)")
    if sample.eog.ndim != 2:
        raise ValueError("EOG tensor must have shape (channel, time)")
    if sample.eeg.shape[0] != len(sample.eeg_channels):
        raise ValueError("EEG tensor channel axis does not match `eeg_channels`")
    if sample.eog.shape[0] != len(sample.eog_channels):
        raise ValueError("EOG tensor channel axis does not match `eog_channels`")
    if sample.eeg.shape[-1] != sample.eog.shape[-1]:
        raise ValueError("EEG and EOG tensor sample counts differ")
    if sample.eeg.device.type != "cpu" or sample.eog.device.type != "cpu":
        raise ValueError("Raw samples must remain on CPU until after collation")
    if sample.eeg.device != sample.eog.device:
        raise ValueError("EEG and EOG tensors are stored on different devices")
    if sample.eeg.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"EEG tensor must use float32 or float64, got {sample.eeg.dtype}")
    if sample.eog.dtype != sample.eeg.dtype:
        raise TypeError("EEG and EOG tensor dtypes differ")
    if not torch.isfinite(sample.eeg).all():
        raise ValueError("EEG tensor contains non-finite values")


def _validate_raw_compatibility(reference: TorchSample, sample: TorchSample) -> None:
    if sample.eeg_channels != reference.eeg_channels:
        raise ValueError("Cannot collate samples with different EEG channels")
    if sample.eog_channels != reference.eog_channels:
        raise ValueError("Cannot collate samples with different EOG channels")
    if not isclose(sample.sfreq, reference.sfreq, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Cannot collate samples with different sampling frequencies")
    if sample.eeg.dtype != reference.eeg.dtype:
        raise TypeError("Cannot collate samples with different tensor dtypes")
    if sample.eeg.device != reference.eeg.device or sample.eog.device != reference.eog.device:
        raise ValueError("Cannot collate samples stored on different devices")
