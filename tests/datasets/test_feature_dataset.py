import json
import os
import warnings
from pathlib import Path

import mne
import numpy as np
import pytest

from features import FeatureMatrix, build_feature_matrix
from utils.datasets import FeatureDataset
from utils.datasets.schemas import LoadedSample


def _save_raw(
    path: Path,
    *,
    data: np.ndarray,
    channel_names: list[str],
    channel_type: str,
    sfreq: float,
) -> None:
    info = mne.create_info(channel_names, sfreq=sfreq, ch_types=[channel_type] * len(channel_names))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw.save(path, overwrite=True, verbose="ERROR")


def _write_feature_trial(
    dataset_dir: Path,
    *,
    dataset_step_type: str,
    block_count: int = 2,
    sfreq: float = 125.0,
) -> None:
    trial_dir = dataset_dir / "S_1" / "Trial_1"
    trial_dir.mkdir(parents=True)
    blocks = [
        {
            "Exec_Block_Index": block_index,
            "type": "geometric",
            "pattern_id": block_index,
            "img": [[0, 1], [1, 0]],
        }
        for block_index in range(1, block_count + 1)
    ]
    (trial_dir / "labels.json").write_text(json.dumps({"blocks": blocks}), encoding="utf-8")

    times = np.arange(251, dtype=np.float64) / sfreq
    for block in blocks:
        block_index = block["Exec_Block_Index"]
        eeg = np.stack(
            (
                np.sin(2 * np.pi * (8 + block_index) * times),
                np.sin(2 * np.pi * (18 + block_index) * times),
            )
        )
        eog = np.zeros((1, times.size), dtype=np.float64)
        _save_raw(
            trial_dir / f"{dataset_step_type}_EEG_{block_index}.fif",
            data=eeg,
            channel_names=["Fz", "Cz"],
            channel_type="eeg",
            sfreq=sfreq,
        )
        _save_raw(
            trial_dir / f"{dataset_step_type}_EOG_{block_index}.fif",
            data=eog,
            channel_names=["EOG_x"],
            channel_type="eog",
            sfreq=sfreq,
        )


def _feature_overrides() -> dict[str, object]:
    return {
        "crop_start_seconds": 0.0,
        "crop_end_seconds": 2.0,
        "window_seconds": 1.0,
        "window_stride_seconds": 1.0,
        "feature_groups": ["time", "local_patterns"],
    }


def test_feature_dataset_supports_indexing_metadata_and_modular_cache(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "Data_Train"
    cache_root = tmp_path / "feature-cache"
    _write_feature_trial(dataset_dir, dataset_step_type="exec")
    dataset = FeatureDataset(
        dataset_dir,
        dataset_step_type="exec",
        config_overrides=_feature_overrides(),
        cache_dir=cache_root,
        source_cache_policy=None,
    )

    by_index = dataset[0]
    by_key = dataset[1, 1, 1]
    entry_dir = dataset.get_cache_entry_path(0)

    assert by_index.sample == by_key.sample
    assert tuple(block.name for block in by_index.blocks) == ("time", "lndp", "lgp", "lbp")
    assert by_index.blocks[0].values.shape == (2, 2, 13)
    assert all(block.values.shape == (2, 2, 256) for block in by_index.blocks[1:])
    assert dataset.samples[0].block_index == 1
    assert dataset.source_map[1][1][1].block_index == 1
    assert dataset.dataset_step_type == "exec"
    assert dataset.cache_dir == (
        cache_root / "Data_Train" / "exec" / "float32" / dataset.config_hash
    )
    assert (entry_dir / "window_bounds_seconds.npy").is_file()
    assert (entry_dir / "manifest.json").is_file()
    assert all((entry_dir / f"{name}.npy").is_file() for name in ("time", "lndp", "lgp", "lbp"))


def test_feature_cache_reuses_arrays_without_running_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "Data_Train"
    cache_root = tmp_path / "feature-cache"
    _write_feature_trial(dataset_dir, dataset_step_type="exec", block_count=1)
    dataset = FeatureDataset(
        dataset_dir,
        config_overrides=_feature_overrides(),
        cache_dir=cache_root,
        source_cache_policy=None,
    )
    expected = dataset[0]

    def fail_extraction(loaded: LoadedSample, *, config: object) -> object:
        raise AssertionError("Feature extraction should not run on a valid cache hit")

    monkeypatch.setattr("utils.datasets.feature_dataset.extract_feature_set", fail_extraction)
    cached_dataset = FeatureDataset(
        dataset_dir,
        config_overrides=_feature_overrides(),
        cache_dir=cache_root,
        source_cache_policy=None,
    )

    def fail_source_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("Source arrays should not load on a valid feature-cache hit")

    monkeypatch.setattr(cached_dataset.source_dataset, "_load_sample", fail_source_load)
    cached = cached_dataset[0]

    for actual_block, expected_block in zip(cached.blocks, expected.blocks, strict=True):
        np.testing.assert_array_equal(actual_block.values, expected_block.values)


def test_feature_cache_invalidates_when_source_signature_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "Data_Train"
    _write_feature_trial(dataset_dir, dataset_step_type="exec", block_count=1)
    dataset = FeatureDataset(
        dataset_dir,
        config_overrides=_feature_overrides(),
        cache_dir=tmp_path / "feature-cache",
        source_cache_policy=None,
    )
    dataset[0]

    sample = dataset.samples[0]
    stat = sample.eeg_path.stat()
    os.utime(sample.eeg_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    from utils.datasets import feature_dataset as feature_dataset_module

    original_extractor = feature_dataset_module.extract_feature_set
    extraction_calls = 0

    def tracking_extractor(loaded: LoadedSample, *, config: object) -> object:
        nonlocal extraction_calls
        extraction_calls += 1
        return original_extractor(loaded, config=config)  # type: ignore[arg-type]

    monkeypatch.setattr(feature_dataset_module, "extract_feature_set", tracking_extractor)
    dataset[0]

    assert extraction_calls == 1


def test_corrupt_feature_block_is_rebuilt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = tmp_path / "Data_Train"
    _write_feature_trial(dataset_dir, dataset_step_type="exec", block_count=1)
    dataset = FeatureDataset(
        dataset_dir,
        config_overrides=_feature_overrides(),
        cache_dir=tmp_path / "feature-cache",
        source_cache_policy=None,
    )
    expected = dataset[0]
    entry_dir = dataset.get_cache_entry_path(0)
    (entry_dir / "lndp.npy").write_bytes(b"not-a-valid-npy")

    from utils.datasets import feature_dataset as feature_dataset_module

    original_extractor = feature_dataset_module.extract_feature_set
    extraction_calls = 0

    def tracking_extractor(loaded: LoadedSample, *, config: object) -> object:
        nonlocal extraction_calls
        extraction_calls += 1
        return original_extractor(loaded, config=config)  # type: ignore[arg-type]

    monkeypatch.setattr(feature_dataset_module, "extract_feature_set", tracking_extractor)
    rebuilt = dataset[0]

    assert extraction_calls == 1
    np.testing.assert_array_equal(
        next(block for block in rebuilt.blocks if block.name == "lndp").values,
        next(block for block in expected.blocks if block.name == "lndp").values,
    )
    np.load(entry_dir / "lndp.npy", allow_pickle=False)


def test_cache_identity_separates_configs_versions_and_recording_families(tmp_path: Path) -> None:
    exec_dir = tmp_path / "Data_Train"
    patt_dir = tmp_path / "Data_Pattern"
    cache_root = tmp_path / "feature-cache"
    _write_feature_trial(exec_dir, dataset_step_type="exec", block_count=1)
    _write_feature_trial(patt_dir, dataset_step_type="patt", block_count=1)

    exec_dataset = FeatureDataset(
        exec_dir,
        dataset_step_type="exec",
        config_overrides={**_feature_overrides(), "feature_groups": ["time"]},
        cache_dir=cache_root,
        source_cache_policy=None,
    )
    alternate_config = FeatureDataset(
        exec_dir,
        dataset_step_type="exec",
        config_overrides={**_feature_overrides(), "feature_groups": ["local_patterns"]},
        cache_dir=cache_root,
        source_cache_policy=None,
    )

    class VersionTwoFeatureDataset(FeatureDataset):
        EXTRACTOR_VERSION = 2

    alternate_version = VersionTwoFeatureDataset(
        exec_dir,
        dataset_step_type="exec",
        config_overrides={**_feature_overrides(), "feature_groups": ["time"]},
        cache_dir=cache_root,
        source_cache_policy=None,
    )
    patt_dataset = FeatureDataset(
        patt_dir,
        dataset_step_type="patt",
        config_overrides={**_feature_overrides(), "feature_groups": ["time"]},
        cache_dir=cache_root,
        source_cache_policy=None,
    )

    assert len(
        {
            exec_dataset.cache_dir,
            alternate_config.cache_dir,
            alternate_version.cache_dir,
            patt_dataset.cache_dir,
        }
    ) == 4


def test_build_feature_matrix_preserves_parent_keys_windows_and_bounds(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "Data_Pattern"
    _write_feature_trial(dataset_dir, dataset_step_type="patt")
    dataset = FeatureDataset(
        dataset_dir,
        dataset_step_type="patt",
        config_overrides=_feature_overrides(),
        cache_policy=None,
        source_cache_policy=None,
    )

    exported = build_feature_matrix(dataset, block_names=("time",))

    assert isinstance(exported, FeatureMatrix)
    assert exported.X.shape == (4, 26)
    assert len(exported.feature_names) == 26
    assert exported.feature_names[0] == "time:Fz:mean"
    assert exported.feature_names[-1] == "time:Cz:hjorth_complexity"
    assert exported.sample_keys == (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 2),
    )
    np.testing.assert_array_equal(exported.window_indices, [0, 1, 0, 1])
    np.testing.assert_array_equal(
        exported.window_bounds_seconds,
        [[0.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 2.0]],
    )
    assert exported.recording_family == "patt"


def test_manifest_records_source_config_and_modular_array_metadata(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "Data_Train"
    _write_feature_trial(dataset_dir, dataset_step_type="exec", block_count=1)
    dataset = FeatureDataset(
        dataset_dir,
        config_overrides=_feature_overrides(),
        cache_dir=tmp_path / "feature-cache",
        source_cache_policy=None,
    )
    feature_set = dataset[0]

    with (dataset.get_cache_entry_path(0) / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["config_hash"] == dataset.config_hash
    assert manifest["extractor_version"] == FeatureDataset.EXTRACTOR_VERSION
    assert manifest["dataset_step_type"] == "exec"
    assert manifest["sources"]["eeg"]["size"] == feature_set.sample.eeg_path.stat().st_size
    assert manifest["sources"]["eog"]["mtime_ns"] == feature_set.sample.eog_path.stat().st_mtime_ns
    assert [block["name"] for block in manifest["blocks"]] == ["time", "lndp", "lgp", "lbp"]
    assert manifest["blocks"][0]["array"] == {"shape": [2, 2, 13], "dtype": "float32"}
