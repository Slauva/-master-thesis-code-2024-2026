import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from utils.datasets import (
    GeometricSample,
    NumpyDataset,
    TorchDataset,
    TorchSample,
    TorchSampleBatch,
    collate_torch_samples,
)


def _save_raw(
    path: Path,
    *,
    data: np.ndarray,
    channel_names: list[str],
    channel_type: str,
    sfreq: float = 100.0,
) -> None:
    info = mne.create_info(channel_names, sfreq=sfreq, ch_types=[channel_type] * len(channel_names))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw.save(path, overwrite=True, verbose="ERROR")


def _write_trial(dataset_dir: Path) -> None:
    trial_dir = dataset_dir / "S_1" / "Trial_1"
    trial_dir.mkdir(parents=True)
    blocks = [
        {
            "Exec_Block_Index": 1,
            "type": "geometric",
            "pattern_id": 3,
            "img": [[0, 1], [1, 0]],
        },
        {
            "Exec_Block_Index": 2,
            "type": "geometric",
            "pattern_id": 4,
            "img": [[1, 0], [0, 1]],
        },
    ]
    (trial_dir / "labels.json").write_text(json.dumps({"blocks": blocks}), encoding="utf-8")
    for block in blocks:
        index = block["Exec_Block_Index"]
        eeg = np.arange(16, dtype=np.float64).reshape(2, 8) + index
        eog = np.arange(8, dtype=np.float64).reshape(1, 8) + index
        _save_raw(
            trial_dir / f"exec_EEG_{index}.fif",
            data=eeg,
            channel_names=["Fz", "Cz"],
            channel_type="eeg",
        )
        _save_raw(
            trial_dir / f"exec_EOG_{index}.fif",
            data=eog,
            channel_names=["VEOG"],
            channel_type="eog",
        )


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


def _torch_sample(
    index: int,
    *,
    length: int,
    dtype: torch.dtype = torch.float32,
    sfreq: float = 100.0,
    eeg_channels: tuple[str, ...] = ("Fz", "Cz"),
    eog_channels: tuple[str, ...] = ("VEOG",),
) -> TorchSample:
    eeg = torch.arange(len(eeg_channels) * length, dtype=dtype).reshape(len(eeg_channels), length)
    eog = torch.arange(len(eog_channels) * length, dtype=dtype).reshape(len(eog_channels), length)
    return TorchSample(
        sample=_metadata(index),
        eeg=eeg,
        eog=eog,
        sfreq=sfreq,
        eeg_channels=eeg_channels,
        eog_channels=eog_channels,
    )


@pytest.mark.parametrize(("dtype", "torch_dtype"), [(np.float32, torch.float32), (np.float64, torch.float64)])
def test_wraps_numpy_dataset_without_copying_arrays(
    tmp_path: Path,
    dtype: type[np.float32] | type[np.float64],
    torch_dtype: torch.dtype,
) -> None:
    _write_trial(tmp_path)
    source = NumpyDataset(tmp_path, dtype=dtype, cache_policy="memory")
    loaded = source[0]
    dataset = TorchDataset(source)

    sample = dataset[0]
    by_key = dataset[1, 1, 1]

    assert sample.eeg.dtype == torch_dtype
    assert sample.eog.dtype == torch_dtype
    assert sample.eeg.data_ptr() == loaded.eeg.__array_interface__["data"][0]
    assert sample.eog.data_ptr() == loaded.eog.__array_interface__["data"][0]
    assert by_key.sample == sample.sample
    torch.testing.assert_close(by_key.eeg, sample.eeg)
    assert dataset.samples is source.samples
    assert dataset.source_map is source.source_map
    assert [item.sample.block_index for item in dataset] == [1, 2]


def test_rejects_non_numpy_source_dataset() -> None:
    with pytest.raises(TypeError, match="must be a NumpyDataset"):
        TorchDataset(object())  # type: ignore[arg-type]


def test_collates_variable_lengths_with_zero_padding_and_masks() -> None:
    first = _torch_sample(1, length=3)
    second = _torch_sample(2, length=5)
    first.eog[0, 1] = torch.nan

    batch = collate_torch_samples([first, second])

    assert isinstance(batch, TorchSampleBatch)
    assert batch.eeg.shape == (2, 2, 5)
    assert batch.eog.shape == (2, 1, 5)
    assert batch.lengths.tolist() == [3, 5]
    assert batch.time_mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert torch.equal(batch.eeg[0, :, 3:], torch.zeros(2, 2))
    assert torch.equal(batch.eog[0, :, 3:], torch.zeros(1, 2))
    assert torch.isnan(batch.eog[0, 0, 1])
    assert batch.eog_finite_mask[0, 0].tolist() == [True, False, True, False, False]
    assert batch.eog_finite_mask[1].all()
    assert batch.samples == (first.sample, second.sample)


def test_batch_to_moves_only_tensor_fields() -> None:
    batch = collate_torch_samples([_torch_sample(1, length=3), _torch_sample(2, length=5)])

    moved = batch.to("cpu", non_blocking=True)

    assert moved.samples is batch.samples
    assert moved.eeg_channels is batch.eeg_channels
    assert moved.eog_channels is batch.eog_channels
    assert moved.eeg.device.type == "cpu"
    assert moved.lengths.device.type == "cpu"
    assert moved.time_mask.dtype == torch.bool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pinned-memory verification requires CUDA")
def test_dataloader_pins_custom_batch_tensors() -> None:
    samples = [_torch_sample(1, length=3), _torch_sample(2, length=5)]
    loader = DataLoader(
        samples,
        batch_size=2,
        collate_fn=collate_torch_samples,
        pin_memory=True,
    )

    batch = next(iter(loader))

    assert batch.eeg.is_pinned()
    assert batch.eog.is_pinned()
    assert batch.lengths.is_pinned()
    assert batch.time_mask.is_pinned()
    assert batch.eog_finite_mask.is_pinned()


def test_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_torch_samples([])


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"eeg_channels": ("Fp1", "Cz")}, "different EEG channels"),
        ({"eog_channels": ("EOG_x",)}, "different EOG channels"),
        ({"sfreq": 125.0}, "different sampling frequencies"),
        ({"dtype": torch.float64}, "different tensor dtypes"),
    ],
)
def test_rejects_incompatible_samples(replacement: dict[str, object], message: str) -> None:
    first = _torch_sample(1, length=3)
    second = _torch_sample(
        2,
        length=5,
        dtype=replacement.get("dtype", torch.float32),  # type: ignore[arg-type]
        sfreq=replacement.get("sfreq", 100.0),  # type: ignore[arg-type]
        eeg_channels=replacement.get("eeg_channels", ("Fz", "Cz")),  # type: ignore[arg-type]
        eog_channels=replacement.get("eog_channels", ("VEOG",)),  # type: ignore[arg-type]
    )

    with pytest.raises((TypeError, ValueError), match=message):
        collate_torch_samples([first, second])


def test_rejects_invalid_tensor_shapes_and_values() -> None:
    valid = _torch_sample(1, length=3)
    invalid_shape = TorchSample(
        sample=valid.sample,
        eeg=valid.eeg.unsqueeze(0),
        eog=valid.eog,
        sfreq=valid.sfreq,
        eeg_channels=valid.eeg_channels,
        eog_channels=valid.eog_channels,
    )
    with pytest.raises(ValueError, match="EEG tensor must have shape"):
        collate_torch_samples([invalid_shape])

    invalid_eeg = _torch_sample(2, length=3)
    invalid_eeg.eeg[0, 0] = torch.nan
    with pytest.raises(ValueError, match="EEG tensor contains non-finite"):
        collate_torch_samples([invalid_eeg])
