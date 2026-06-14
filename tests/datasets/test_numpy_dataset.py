import json
import os
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
    dataset = NumpyDataset(tmp_path, dtype=dtype, cache_policy=None)

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
    dataset = NumpyDataset(tmp_path, cache_policy=None)

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
    dataset = NumpyDataset(tmp_path, dataset_pattern_type=pattern_type, cache_policy=None)  # type: ignore[arg-type]

    assert len(dataset) == 1
    assert dataset[0].sample.block_index == expected_block
    assert set(dataset.source_map[1][1]) == {expected_block}

    missing_block = 2 if expected_block == 1 else 1
    with pytest.raises(KeyError, match=f"block={missing_block}"):
        dataset[1, 1, missing_block]


def test_iteration_loads_samples_in_stable_order(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, cache_policy=None)

    assert [loaded.sample.block_index for loaded in dataset] == [1, 2]


@pytest.mark.parametrize("dtype", [np.int16, np.complex64, "invalid"])
def test_rejects_unsupported_dtype(tmp_path: Path, dtype: object) -> None:
    _write_numpy_trial(tmp_path)

    with pytest.raises((TypeError, ValueError), match="dtype|not understood"):
        NumpyDataset(tmp_path, dtype=dtype, cache_policy=None)  # type: ignore[arg-type]


def test_rejects_different_sample_counts(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path, eeg_n_times=8, eog_n_times=7)
    dataset = NumpyDataset(tmp_path, cache_policy=None)

    with pytest.raises(ValueError, match="sample counts differ"):
        dataset[0]


def test_rejects_different_sampling_frequencies(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path, eeg_sfreq=100.0, eog_sfreq=200.0)
    dataset = NumpyDataset(tmp_path, cache_policy=None)

    with pytest.raises(ValueError, match="sampling frequencies differ"):
        dataset[0]


def test_rejects_invalid_key_type(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, cache_policy=None)

    with pytest.raises(TypeError, match="must be an integer"):
        dataset[True]


def test_disk_cache_writes_manifest_and_reuses_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _write_numpy_trial(dataset_dir)
    dataset = NumpyDataset(dataset_dir, cache_policy="disk", cache_dir=cache_dir)

    first = dataset[0]
    entry_dir = dataset.get_cache_entry_path(0)

    assert (entry_dir / "eeg.npy").is_file()
    assert (entry_dir / "eog.npy").is_file()
    assert (entry_dir / "manifest.json").is_file()

    def fail_if_fif_is_opened(*args: object, **kwargs: object) -> None:
        raise AssertionError("FIF should not be opened on a cache hit")

    monkeypatch.setattr(mne.io, "read_raw_fif", fail_if_fif_is_opened)
    second = dataset[0]

    np.testing.assert_array_equal(second.eeg, first.eeg)
    np.testing.assert_array_equal(second.eog, first.eog)
    assert second.sample == first.sample


def test_disk_cache_is_invalidated_when_source_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _write_numpy_trial(dataset_dir)
    dataset = NumpyDataset(dataset_dir, cache_policy="disk", cache_dir=cache_dir)
    dataset[0]

    sample = dataset.samples[0]
    stat = sample.eeg_path.stat()
    os.utime(sample.eeg_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    original_reader = mne.io.read_raw_fif
    opened_paths: list[Path] = []

    def tracking_reader(path: Path, *args: object, **kwargs: object) -> mne.io.BaseRaw:
        opened_paths.append(Path(path))
        return original_reader(path, *args, **kwargs)

    monkeypatch.setattr(mne.io, "read_raw_fif", tracking_reader)
    dataset[0]

    assert opened_paths == [sample.eeg_path, sample.eog_path]


def test_incomplete_or_corrupt_cache_is_rebuilt(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _write_numpy_trial(dataset_dir)
    dataset = NumpyDataset(dataset_dir, cache_policy="disk", cache_dir=cache_dir)
    expected = dataset[0]
    entry_dir = dataset.get_cache_entry_path(0)

    (entry_dir / "manifest.json").unlink()
    (entry_dir / "eeg.npy").write_bytes(b"not-a-valid-npy")

    rebuilt = dataset[0]

    np.testing.assert_array_equal(rebuilt.eeg, expected.eeg)
    np.testing.assert_array_equal(rebuilt.eog, expected.eog)
    with (entry_dir / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest["schema_version"] == NumpyDataset.CACHE_SCHEMA_VERSION


def test_cache_manifest_contains_source_and_array_metadata(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _write_numpy_trial(dataset_dir)
    dataset = NumpyDataset(dataset_dir, cache_policy="disk", cache_dir=cache_dir)
    loaded = dataset[0]

    with (dataset.get_cache_entry_path(0) / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["key"] == {"subject_id": 1, "trial_number": 1, "block_index": 1}
    assert manifest["dtype"] == "float32"
    assert manifest["arrays"]["eeg"] == {"shape": list(loaded.eeg.shape), "dtype": "float32"}
    assert manifest["sources"]["eeg"]["size"] == loaded.sample.eeg_path.stat().st_size
    assert manifest["sources"]["eog"]["mtime_ns"] == loaded.sample.eog_path.stat().st_mtime_ns


def test_default_cache_path_is_inside_project_artifacts() -> None:
    dataset = NumpyDataset(Path("data/Data_Train"))

    assert dataset.cache_dir == (
        Path.cwd() / "artifacts" / "cache" / "Data_Train" / "exec" / "float32"
    ).resolve()
    assert not dataset.get_cache_entry_path(0).exists()


def test_memory_cache_reuses_loaded_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, cache_policy="memory")

    first = dataset[0]

    def fail_if_fif_is_opened(*args: object, **kwargs: object) -> None:
        raise AssertionError("FIF should not be opened on a memory cache hit")

    monkeypatch.setattr(dataset, "_load_sample", fail_if_fif_is_opened)
    second = dataset[0]

    assert second is first
    assert dataset.memory_cache_items == 1
    assert dataset.memory_cache_current_bytes == first.eeg.nbytes + first.eog.nbytes
    assert dataset.memory_cache_keys == ((1, 1, 1),)


def test_memory_cache_uses_lru_eviction_by_bytes(tmp_path: Path) -> None:
    blocks = [
        {
            "Exec_Block_Index": index,
            "type": "geometric",
            "pattern_id": index,
            "img": [[0, 1], [1, 0]],
        }
        for index in (1, 2, 3)
    ]
    _write_numpy_trial(tmp_path, blocks=blocks)
    sample_bytes = (2 * 8 + 1 * 8) * np.dtype(np.float32).itemsize
    dataset = NumpyDataset(
        tmp_path,
        cache_policy="memory",
        memory_cache_bytes=sample_bytes * 2,
    )

    dataset[0]
    dataset[1]
    dataset[0]
    dataset[2]

    assert dataset.memory_cache_keys == ((1, 1, 1), (1, 1, 3))
    assert dataset.memory_cache_items == 2
    assert dataset.memory_cache_current_bytes == sample_bytes * 2


def test_oversized_sample_is_not_stored_in_memory_cache(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, cache_policy="memory", memory_cache_bytes=1)

    dataset[0]

    assert dataset.memory_cache_items == 0
    assert dataset.memory_cache_current_bytes == 0
    assert dataset.memory_cache_keys == ()


def test_both_policy_prefers_memory_over_disk_and_fif(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_numpy_trial(dataset_dir)
    dataset = NumpyDataset(
        dataset_dir,
        cache_policy="both",
        cache_dir=tmp_path / "cache",
    )
    first = dataset[0]

    def fail_if_lower_cache_is_used(*args: object, **kwargs: object) -> None:
        raise AssertionError("Memory cache should be checked before disk and FIF")

    monkeypatch.setattr(dataset, "_load_disk_cache", fail_if_lower_cache_is_used)
    monkeypatch.setattr(dataset, "_load_sample", fail_if_lower_cache_is_used)

    assert dataset[0] is first


def test_memory_cache_invalidates_entry_when_source_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, cache_policy="memory")
    dataset[0]

    sample = dataset.samples[0]
    stat = sample.eeg_path.stat()
    os.utime(sample.eeg_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    original_loader = dataset._load_sample
    load_count = 0

    def tracking_loader(current_sample: object) -> object:
        nonlocal load_count
        load_count += 1
        return original_loader(current_sample)  # type: ignore[arg-type]

    monkeypatch.setattr(dataset, "_load_sample", tracking_loader)
    dataset[0]

    assert load_count == 1
    assert dataset.memory_cache_items == 1


def test_clear_memory_cache_resets_usage(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, cache_policy="memory")
    dataset[0]

    dataset.clear_memory_cache()

    assert dataset.memory_cache_items == 0
    assert dataset.memory_cache_current_bytes == 0
    assert dataset.memory_cache_keys == ()


@pytest.mark.parametrize("memory_cache_bytes", [-1, 1.5, True])
def test_rejects_invalid_memory_cache_limit(tmp_path: Path, memory_cache_bytes: object) -> None:
    _write_numpy_trial(tmp_path)

    with pytest.raises(ValueError, match="non-negative integer"):
        NumpyDataset(
            tmp_path,
            cache_policy="memory",
            memory_cache_bytes=memory_cache_bytes,  # type: ignore[arg-type]
        )


def test_warm_cache_sequential_builds_then_reuses_disk_entries(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _write_numpy_trial(dataset_dir)
    dataset = NumpyDataset(
        dataset_dir,
        cache_policy="both",
        cache_dir=cache_dir,
    )

    first_report = dataset.warm_cache(max_workers=1)
    second_report = dataset.warm_cache(max_workers=1)

    assert first_report.processed == 2
    assert first_report.cached == 0
    assert first_report.skipped == 0
    assert first_report.failed == 0
    assert first_report.total == 2
    assert first_report.max_workers == 1
    assert not first_report.errors
    assert second_report.processed == 0
    assert second_report.cached == 2
    assert second_report.total == 2
    assert dataset.memory_cache_items == 0


def test_warm_cache_multiprocessing_builds_entries(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    _write_numpy_trial(dataset_dir)
    dataset = NumpyDataset(dataset_dir, cache_policy="disk", cache_dir=cache_dir)

    report = dataset.warm_cache(max_workers=2)

    assert report.processed == 2
    assert report.cached == 0
    assert report.skipped == 0
    assert report.failed == 0
    assert report.total == 2
    assert report.max_workers == 2
    assert all((dataset.get_cache_entry_path(index) / "manifest.json").is_file() for index in range(2))


def test_warm_cache_collects_errors_and_continues(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_numpy_trial(dataset_dir)
    (dataset_dir / "S_1" / "Trial_1" / "exec_EEG_1.fif").write_bytes(b"invalid-fif")
    dataset = NumpyDataset(dataset_dir, cache_policy="disk", cache_dir=tmp_path / "cache")

    report = dataset.warm_cache(max_workers=1, fail_fast=False)

    assert report.processed == 1
    assert report.cached == 0
    assert report.skipped == 0
    assert report.failed == 1
    assert report.total == 2
    assert report.errors[0].key == (1, 1, 1)
    assert report.errors[0].error_type
    assert report.errors[0].message


def test_warm_cache_fail_fast_marks_remaining_samples_skipped(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_numpy_trial(dataset_dir)
    (dataset_dir / "S_1" / "Trial_1" / "exec_EEG_1.fif").write_bytes(b"invalid-fif")
    dataset = NumpyDataset(dataset_dir, cache_policy="disk", cache_dir=tmp_path / "cache")

    report = dataset.warm_cache(max_workers=1, fail_fast=True)

    assert report.processed == 0
    assert report.cached == 0
    assert report.skipped == 1
    assert report.failed == 1
    assert report.total == 2


def test_warm_cache_requires_disk_policy(tmp_path: Path) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, cache_policy="memory")

    with pytest.raises(ValueError, match="requires cache_policy"):
        dataset.warm_cache()


@pytest.mark.parametrize("max_workers", [0, -1, 1.5, True])
def test_warm_cache_rejects_invalid_worker_count(tmp_path: Path, max_workers: object) -> None:
    _write_numpy_trial(tmp_path)
    dataset = NumpyDataset(tmp_path, cache_policy="disk", cache_dir=tmp_path / "cache")

    with pytest.raises(ValueError, match="positive integer"):
        dataset.warm_cache(max_workers=max_workers)  # type: ignore[arg-type]
