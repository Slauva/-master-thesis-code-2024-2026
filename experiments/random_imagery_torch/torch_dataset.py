from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.random_imagery.shared import PixelTargetDataset
from experiments.random_imagery_torch.normalization import normalize_spectral_sample
from experiments.random_imagery_torch.schemas import (
    SpectralNormalizationState,
    TorchSpectralInputBatch,
    TorchSpectralInputSample,
)
from experiments.random_imagery_torch.spectral_dataset import CropSpectralDataset
from preprocessors.config import build_frequency_grid


class TorchSpectralInputDataset(Dataset[TorchSpectralInputSample]):
    def __init__(
        self,
        source_dataset: CropSpectralDataset,
        targets: PixelTargetDataset,
        target_row_indices: np.ndarray,
        normalization: SpectralNormalizationState,
    ) -> None:
        rows = np.asarray(target_row_indices)
        if rows.ndim != 1 or rows.dtype != np.dtype(np.int64):
            raise TypeError("Target row indices must be a one-dimensional int64 array")
        if rows.size < 1 or np.any(rows < 0) or np.any(rows >= targets.y.shape[0]):
            raise ValueError("Target row indices must contain valid target rows")
        if not np.array_equal(rows, np.unique(rows)):
            raise ValueError("Target row indices must be sorted and unique")
        if normalization.method != source_dataset.method:
            raise ValueError("Normalization method does not match the spectral dataset")
        if normalization.scaling != source_dataset.preprocessing_config.scaling:
            raise ValueError("Normalization scaling does not match the spectral dataset")
        if normalization.crop_bounds_seconds != source_dataset.input_config.crop_bounds_seconds:
            raise ValueError("Normalization crop bounds do not match the spectral dataset")
        expected_frequencies = build_frequency_grid(source_dataset.preprocessing_config)
        if not np.array_equal(normalization.frequencies, expected_frequencies):
            raise ValueError("Normalization frequency grid does not match the spectral dataset")

        source_by_key = {
            (sample.subject_id, sample.trial_number, sample.block_index): sample
            for sample in source_dataset.samples
        }
        for row in rows:
            row_index = int(row)
            key = targets.sample_keys[row_index]
            try:
                sample = source_by_key[key]
            except KeyError as error:
                raise ValueError(f"Target sample key is absent from the spectral dataset: {key}") from error
            target = targets.y[row_index]
            if np.asarray(sample.img, dtype=np.int8).reshape(-1).shape != target.shape or not np.array_equal(
                np.asarray(sample.img, dtype=np.int8).reshape(-1),
                target,
            ):
                raise ValueError(f"Target payload does not match source sample metadata for key {key}")

        self.source_dataset = source_dataset
        self.targets = targets
        self.target_row_indices = rows.copy()
        self.target_row_indices.setflags(write=False)
        self.normalization = normalization

    def __len__(self) -> int:
        return self.target_row_indices.size

    def __getitem__(self, index: int) -> TorchSpectralInputSample:
        target_row_index = int(self.target_row_indices[index])
        sample_key = self.targets.sample_keys[target_row_index]
        spectral = self.source_dataset[sample_key]
        if spectral.sample_key != sample_key:
            raise ValueError("Loaded spectral sample does not match the aligned target row")
        model_input = normalize_spectral_sample(spectral, self.normalization)
        return TorchSpectralInputSample(
            sample_key=sample_key,
            target_row_index=target_row_index,
            model_input=torch.from_numpy(model_input.copy()),
            target=torch.from_numpy(
                self.targets.y[target_row_index].astype(np.float32, copy=True)
            ),
            frequencies=torch.from_numpy(spectral.frequencies.copy()),
            times=None if spectral.times is None else torch.from_numpy(spectral.times.copy()),
            eeg_channels=spectral.eeg_channels,
            method=spectral.method,
            scaling=spectral.scaling,
        )

    def __iter__(self) -> Iterator[TorchSpectralInputSample]:
        for index in range(len(self)):
            yield self[index]


def collate_spectral_inputs(
    samples: Sequence[TorchSpectralInputSample],
) -> TorchSpectralInputBatch:
    if not samples:
        raise ValueError("Cannot collate an empty spectral input batch")
    reference = samples[0]
    _validate_torch_sample(reference)
    for sample in samples[1:]:
        _validate_torch_sample(sample)
        if sample.method != reference.method or sample.scaling != reference.scaling:
            raise ValueError("Spectral input samples must share method and scaling")
        if sample.eeg_channels != reference.eeg_channels:
            raise ValueError("Spectral input samples must share EEG channel order")
        if sample.model_input.shape != reference.model_input.shape:
            raise ValueError("Spectral input samples must share the exact model-input shape")
        if not torch.equal(sample.frequencies, reference.frequencies):
            raise ValueError("Spectral input samples must share the frequency grid")
        if (sample.times is None) != (reference.times is None):
            raise ValueError("Spectral input samples must share time-axis presence")
        if sample.times is not None and not torch.equal(sample.times, reference.times):
            raise ValueError("Spectral input samples must share the exact time axis")

    return TorchSpectralInputBatch(
        sample_keys=tuple(sample.sample_key for sample in samples),
        target_row_indices=torch.tensor(
            [sample.target_row_index for sample in samples],
            dtype=torch.int64,
        ),
        model_inputs=torch.stack([sample.model_input for sample in samples]),
        targets=torch.stack([sample.target for sample in samples]),
        frequencies=reference.frequencies,
        times=reference.times,
        eeg_channels=reference.eeg_channels,
        method=reference.method,
        scaling=reference.scaling,
    )


def _validate_torch_sample(sample: TorchSpectralInputSample) -> None:
    expected_ndim = 3
    if sample.model_input.ndim != expected_ndim:
        raise ValueError("Per-sample model input must be three-dimensional")
    if sample.method == "fft":
        if sample.model_input.shape[0] != 1 or sample.times is not None:
            raise ValueError("FFT model input must have shape (1, electrode, frequency)")
        if sample.model_input.shape[1] != len(sample.eeg_channels):
            raise ValueError("FFT model-input electrode axis is inconsistent")
        if sample.model_input.shape[2] != sample.frequencies.numel():
            raise ValueError("FFT model-input frequency axis is inconsistent")
    else:
        if sample.times is None:
            raise ValueError("Time-frequency model inputs require a time axis")
        if sample.model_input.shape[0] != sample.frequencies.numel():
            raise ValueError("Time-frequency model-input frequency axis is inconsistent")
        if sample.model_input.shape[1] != len(sample.eeg_channels):
            raise ValueError("Time-frequency model-input electrode axis is inconsistent")
        if sample.model_input.shape[2] != sample.times.numel():
            raise ValueError("Time-frequency model-input time axis is inconsistent")
    if sample.target.shape != (36,) or sample.target.dtype != torch.float32:
        raise TypeError("Random-imagery targets must be float32 vectors with 36 pixels")
    if sample.model_input.dtype != torch.float32:
        raise TypeError("Model inputs must use float32")
    if not torch.isfinite(sample.model_input).all():
        raise ValueError("Model inputs must be finite")
    if not torch.isin(sample.target, torch.tensor([0.0, 1.0])).all():
        raise ValueError("Targets must be binary")
