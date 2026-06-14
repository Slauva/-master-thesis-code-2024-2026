from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from preprocessors import FFTConfig, SpectralTransformResult
from utils.datasets import (
    GeometricSample,
    PreprocessedDataset,
    SpectralSample,
    TorchPreprocessedDataset,
    TorchSpectralBatch,
    TorchSpectralSample,
    collate_torch_spectral_samples,
)
from utils.datasets.schemas import LoadedSample

SpectralMethod = Literal["fft", "morlet", "superlet", "stft"]


class StubPreprocessedDataset(PreprocessedDataset):
    METHOD = "fft"
    CONFIG_TYPE = FFTConfig

    def __init__(self, items: list[SpectralSample]) -> None:
        self._items = items
        self._by_key = {
            (item.sample.subject_id, item.sample.trial_number, item.sample.block_index): item
            for item in items
        }

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: int | tuple[int, int, int]) -> SpectralSample:
        return self._items[key] if isinstance(key, int) else self._by_key[key]

    @property
    def samples(self) -> tuple[GeometricSample, ...]:
        return tuple(item.sample for item in self._items)  # type: ignore[return-value]

    @property
    def source_map(self) -> dict[int, dict[int, dict[int, GeometricSample]]]:
        result: dict[int, dict[int, dict[int, GeometricSample]]] = {}
        for item in self._items:
            result.setdefault(item.sample.subject_id, {}).setdefault(item.sample.trial_number, {})[
                item.sample.block_index
            ] = item.sample  # type: ignore[assignment]
        return result

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        raise NotImplementedError


def _metadata(index: int) -> GeometricSample:
    return GeometricSample(
        subject_id=1,
        trial_number=1,
        Exec_Block_Index=index,
        eeg_path=Path(f"eeg-{index}.fif"),
        eog_path=Path(f"eog-{index}.fif"),
        img=[[0, 1], [1, 0]],
        pattern_id=index,
    )


def _numpy_spectral_sample(
    index: int,
    *,
    method: SpectralMethod,
    spectral_length: int = 4,
    eog_length: int = 6,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> SpectralSample:
    frequencies = np.arange(2.0, 5.0, dtype=dtype)
    if method == "fft":
        power = np.arange(6, dtype=dtype).reshape(2, 3)
        times = None
        scaling = "psd"
    else:
        power = np.arange(2 * 3 * spectral_length, dtype=dtype).reshape(2, 3, spectral_length)
        times = np.arange(spectral_length, dtype=dtype) * dtype.type(0.25)
        scaling = "psd" if method == "stft" else "wavelet_power"
    eog = np.arange(eog_length, dtype=dtype).reshape(1, eog_length)
    return SpectralSample(
        sample=_metadata(index),
        eeg_power=power,
        eog=eog,
        frequencies=frequencies,
        times=times,
        eeg_channels=("Fz", "Cz"),
        eog_channels=("VEOG",),
        source_sfreq=100.0,
        analysis_sfreq=125.0,
        method=method,
        scaling=scaling,
    )


def _torch_spectral_sample(
    index: int,
    *,
    method: SpectralMethod,
    spectral_length: int = 4,
    eog_length: int = 6,
    dtype: torch.dtype = torch.float32,
    frequencies: torch.Tensor | None = None,
    source_sfreq: float = 100.0,
    analysis_sfreq: float = 125.0,
    eeg_channels: tuple[str, ...] = ("Fz", "Cz"),
    eog_channels: tuple[str, ...] = ("VEOG",),
    scaling: Literal["psd", "wavelet_power"] | None = None,
) -> TorchSpectralSample:
    frequency_axis = frequencies if frequencies is not None else torch.arange(2.0, 5.0, dtype=dtype)
    if method == "fft":
        power = torch.arange(len(eeg_channels) * frequency_axis.numel(), dtype=dtype).reshape(
            len(eeg_channels), frequency_axis.numel()
        )
        times = None
        resolved_scaling = scaling or "psd"
    else:
        power = torch.arange(
            len(eeg_channels) * frequency_axis.numel() * spectral_length,
            dtype=dtype,
        ).reshape(len(eeg_channels), frequency_axis.numel(), spectral_length)
        times = torch.arange(spectral_length, dtype=dtype) * 0.25
        resolved_scaling = scaling or ("psd" if method == "stft" else "wavelet_power")
    eog = torch.arange(len(eog_channels) * eog_length, dtype=dtype).reshape(
        len(eog_channels), eog_length
    )
    return TorchSpectralSample(
        sample=_metadata(index),
        eeg_power=power,
        eog=eog,
        frequencies=frequency_axis,
        times=times,
        eeg_channels=eeg_channels,
        eog_channels=eog_channels,
        source_sfreq=source_sfreq,
        analysis_sfreq=analysis_sfreq,
        method=method,
        scaling=resolved_scaling,
    )


@pytest.mark.parametrize("method", ["fft", "morlet", "superlet", "stft"])
def test_wraps_all_preprocessed_methods_without_copying(method: SpectralMethod) -> None:
    source_sample = _numpy_spectral_sample(1, method=method)
    source = StubPreprocessedDataset([source_sample])
    dataset = TorchPreprocessedDataset(source)

    sample = dataset[0]
    by_key = dataset[1, 1, 1]

    assert sample.method == method
    assert sample.eeg_power.data_ptr() == source_sample.eeg_power.__array_interface__["data"][0]
    assert sample.eog.data_ptr() == source_sample.eog.__array_interface__["data"][0]
    assert sample.frequencies.data_ptr() == source_sample.frequencies.__array_interface__["data"][0]
    if source_sample.times is None:
        assert sample.times is None
    else:
        assert sample.times is not None
        assert sample.times.data_ptr() == source_sample.times.__array_interface__["data"][0]
    torch.testing.assert_close(by_key.eeg_power, sample.eeg_power)
    assert dataset.samples == source.samples
    assert dataset.source_map == source.source_map


def test_rejects_non_preprocessed_source_dataset() -> None:
    with pytest.raises(TypeError, match="must be a PreprocessedDataset"):
        TorchPreprocessedDataset(object())  # type: ignore[arg-type]


def test_collates_fft_without_spectral_time_metadata() -> None:
    first = _torch_spectral_sample(1, method="fft", eog_length=4)
    second = _torch_spectral_sample(2, method="fft", eog_length=6)
    first.eog[0, 1] = torch.nan

    batch = collate_torch_spectral_samples([first, second])

    assert isinstance(batch, TorchSpectralBatch)
    assert batch.eeg_power.shape == (2, 2, 3)
    assert batch.times is None
    assert batch.spectral_lengths is None
    assert batch.spectral_time_mask is None
    assert batch.eog.shape == (2, 1, 6)
    assert batch.eog_lengths.tolist() == [4, 6]
    assert batch.eog_time_mask.sum(dim=1).tolist() == [4, 6]
    assert torch.equal(batch.eog[0, :, 4:], torch.zeros(1, 2))
    assert batch.eog_finite_mask[0, 0].tolist() == [True, False, True, True, False, False]


@pytest.mark.parametrize("method", ["morlet", "superlet", "stft"])
def test_collates_time_frequency_methods_with_independent_padding(method: SpectralMethod) -> None:
    first = _torch_spectral_sample(1, method=method, spectral_length=3, eog_length=5)
    second = _torch_spectral_sample(2, method=method, spectral_length=5, eog_length=7)

    batch = collate_torch_spectral_samples([first, second])

    assert batch.eeg_power.shape == (2, 2, 3, 5)
    assert batch.times is not None
    assert batch.times.shape == (2, 5)
    assert batch.spectral_lengths is not None
    assert batch.spectral_lengths.tolist() == [3, 5]
    assert batch.spectral_time_mask is not None
    assert batch.spectral_time_mask.sum(dim=1).tolist() == [3, 5]
    assert torch.equal(batch.eeg_power[0, :, :, 3:], torch.zeros(2, 3, 2))
    assert torch.equal(batch.times[0, 3:], torch.zeros(2))
    assert batch.eog_lengths.tolist() == [5, 7]
    assert batch.eog_time_mask.sum(dim=1).tolist() == [5, 7]
    assert batch.method == method


def test_spectral_batch_to_moves_only_tensor_fields() -> None:
    batch = collate_torch_spectral_samples(
        [
            _torch_spectral_sample(1, method="morlet", spectral_length=3),
            _torch_spectral_sample(2, method="morlet", spectral_length=5),
        ]
    )

    moved = batch.to("cpu", non_blocking=True)

    assert moved.samples is batch.samples
    assert moved.eeg_channels is batch.eeg_channels
    assert moved.eeg_power.device.type == "cpu"
    assert moved.frequencies.device.type == "cpu"
    assert moved.times is not None and moved.times.device.type == "cpu"
    assert moved.spectral_lengths is not None and moved.spectral_lengths.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pinned-memory verification requires CUDA")
def test_dataloader_pins_custom_spectral_batch_tensors() -> None:
    samples = [
        _torch_spectral_sample(1, method="stft", spectral_length=3),
        _torch_spectral_sample(2, method="stft", spectral_length=5),
    ]
    loader = DataLoader(
        samples,
        batch_size=2,
        collate_fn=collate_torch_spectral_samples,
        pin_memory=True,
    )

    batch = next(iter(loader))

    assert batch.eeg_power.is_pinned()
    assert batch.eog.is_pinned()
    assert batch.frequencies.is_pinned()
    assert batch.times is not None and batch.times.is_pinned()
    assert batch.spectral_lengths is not None and batch.spectral_lengths.is_pinned()
    assert batch.spectral_time_mask is not None and batch.spectral_time_mask.is_pinned()
    assert batch.eog_lengths.is_pinned()
    assert batch.eog_time_mask.is_pinned()
    assert batch.eog_finite_mask.is_pinned()


def test_rejects_empty_spectral_batch() -> None:
    with pytest.raises(ValueError, match="empty spectral batch"):
        collate_torch_spectral_samples([])


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"method": "stft"}, "different preprocessing methods"),
        ({"scaling": "psd"}, "different spectral scaling"),
        ({"eeg_channels": ("Fp1", "Cz")}, "different EEG channels"),
        ({"eog_channels": ("EOG_x",)}, "different EOG channels"),
        ({"source_sfreq": 200.0}, "different source sampling frequencies"),
        ({"analysis_sfreq": 100.0}, "different analysis sampling frequencies"),
        ({"dtype": torch.float64}, "different tensor dtypes"),
        ({"frequencies": torch.tensor([2.0, 3.0, 5.0])}, "different frequency grids"),
    ],
)
def test_rejects_incompatible_spectral_samples(replacement: dict[str, object], message: str) -> None:
    first = _torch_spectral_sample(1, method="morlet")
    second = _torch_spectral_sample(
        2,
        method=replacement.get("method", "morlet"),  # type: ignore[arg-type]
        dtype=replacement.get("dtype", torch.float32),  # type: ignore[arg-type]
        frequencies=replacement.get("frequencies"),  # type: ignore[arg-type]
        source_sfreq=replacement.get("source_sfreq", 100.0),  # type: ignore[arg-type]
        analysis_sfreq=replacement.get("analysis_sfreq", 125.0),  # type: ignore[arg-type]
        eeg_channels=replacement.get("eeg_channels", ("Fz", "Cz")),  # type: ignore[arg-type]
        eog_channels=replacement.get("eog_channels", ("VEOG",)),  # type: ignore[arg-type]
        scaling=replacement.get("scaling"),  # type: ignore[arg-type]
    )

    with pytest.raises((TypeError, ValueError), match=message):
        collate_torch_spectral_samples([first, second])


def test_rejects_invalid_spectral_values_and_time_axis() -> None:
    invalid_power = _torch_spectral_sample(1, method="fft")
    invalid_power.eeg_power[0, 0] = -1.0
    with pytest.raises(ValueError, match="finite non-negative"):
        collate_torch_spectral_samples([invalid_power])

    invalid_times = _torch_spectral_sample(2, method="morlet")
    assert invalid_times.times is not None
    invalid_times.times[2] = invalid_times.times[1]
    with pytest.raises(ValueError, match="strictly increasing"):
        collate_torch_spectral_samples([invalid_times])
