import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pytest

from utils.datasets import LoadedSample, NumpyDataset


def _save_raw(
    path: Path,
    *,
    channel_names: list[str],
    channel_type: str,
    n_times: int = 8,
    sfreq: float = 100.0,
    offset: float = 0.0,
) -> None:
    info = mne.create_info(channel_names, sfreq=sfreq, ch_types=[channel_type] * len(channel_names))
    data = np.arange(len(channel_names) * n_times, dtype=np.float64).reshape(len(channel_names), n_times)
    raw = mne.io.RawArray(data + offset, info, verbose="ERROR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw.save(path, overwrite=True, verbose="ERROR")


def _write_numpy_trial(
    dataset_dir: Path,
    *,
    blocks: list[dict] | None = None,
    eeg_n_times: int = 8,
    eog_n_times: int = 8,
    eeg_sfreq: float = 100.0,
    eog_sfreq: float = 100.0,
) -> Path:
    trial_dir = dataset_dir / "S_1" / "Trial_1"
    trial_dir.mkdir(parents=True)
    trial_blocks = blocks or [
        {
            "Exec_Block_Index": 1,
            "type": "geometric",
            "pattern_id": 3,
            "img": [[0, 1], [1, 0]],
        },
        {
            "Exec_Block_Index": 2,
            "type": "random",
            "seed": 42,
            "img": [[1, 0], [0, 1]],
        },
    ]
    (trial_dir / "labels.json").write_text(json.dumps({"blocks": trial_blocks}), encoding="utf-8")

    for block in trial_blocks:
        index = block["Exec_Block_Index"]
        _save_raw(
            trial_dir / f"exec_EEG_{index}.fif",
            channel_names=["Fz", "Cz"],
            channel_type="eeg",
            n_times=eeg_n_times,
            sfreq=eeg_sfreq,
            offset=float(index),
        )
        _save_raw(
            trial_dir / f"exec_EOG_{index}.fif",
            channel_names=["VEOG"],
            channel_type="eog",
            n_times=eog_n_times,
            sfreq=eog_sfreq,
            offset=float(index * 10),
        )

    return trial_dir


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [(np.float32, np.dtype("float32")), (np.float64, np.dtype("float64"))],
)
def test_loads_fif_as_numpy_arrays(
    tmp_path: Path,
    dtype: type[np.float32] | type[np.float64],
    expected_dtype: np.dtype,
) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, dtype=dtype)

    loaded = dataset[0]

    assert isinstance(loaded, LoadedSample)
    assert loaded.sample.block_index == 1
    assert loaded.eeg.shape == (2, 8)
    assert loaded.eog.shape == (1, 8)
    assert loaded.eeg.dtype == expected_dtype
    assert loaded.eog.dtype == expected_dtype
    assert loaded.sfreq == 100.0
    assert loaded.eeg_channels == ("Fz", "Cz")
    assert loaded.eog_channels == ("VEOG",)


def test_integer_and_tuple_indexing_load_same_sample(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path)

    by_position = dataset[1]
    by_key = dataset[1, 1, 2]

    assert by_position.sample == by_key.sample
    np.testing.assert_array_equal(by_position.eeg, by_key.eeg)
    np.testing.assert_array_equal(by_position.eog, by_key.eog)


@pytest.mark.parametrize(("pattern_type", "expected_block"), [("geometric", 1), ("random", 2)])
def test_filters_all_public_indexes_by_pattern_type(
    tmp_path: Path,
    pattern_type: str,
    expected_block: int,
) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, dataset_pattern_type=pattern_type)  # type: ignore[arg-type]

    assert len(dataset) == 1
    assert dataset[0].sample.block_index == expected_block
    assert set(dataset.source_map[1][1]) == {expected_block}

    missing_block = 2 if expected_block == 1 else 1
    with pytest.raises(KeyError, match=f"block={missing_block}"):
        dataset[1, 1, missing_block]


def test_iteration_loads_samples_in_stable_order(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path)

    assert [loaded.sample.block_index for loaded in dataset] == [1, 2]


@pytest.mark.parametrize("dtype", [np.int16, np.complex64, "invalid"])
def test_rejects_unsupported_dtype(tmp_path: Path, dtype: object) -> None:
    _write_numpy_trial(tmp_path)

    with pytest.raises((TypeError, ValueError), match="dtype|not understood"):
        NumpyDataset(tmp_path, dtype=dtype)  # type: ignore[arg-type]


def test_rejects_different_sample_counts(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path, eeg_n_times=8, eog_n_times=7)
    dataset = NumpyDataset(tmp_path)

    with pytest.raises(ValueError, match="sample counts differ"):
        dataset[0]


def test_rejects_different_sampling_frequencies(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path, eeg_sfreq=100.0, eog_sfreq=200.0)
    dataset = NumpyDataset(tmp_path)

    with pytest.raises(ValueError, match="sampling frequencies differ"):
        dataset[0]


def test_rejects_invalid_key_type(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path)

    with pytest.raises(TypeError, match="must be an integer"):
        dataset[True]
