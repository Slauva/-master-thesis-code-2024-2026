import json
import os
import warnings
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from preprocessors import FFTConfig, SpectralTransformResult, load_preprocessing_config
from utils.datasets import (
    FFTDataset,
    MorletDataset,
    PreprocessedDataset,
    SpectralSample,
    STFTDataset,
    SuperletDataset,
)
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


def _write_trial(
    dataset_dir: Path,
    *,
    eeg: np.ndarray | None = None,
    sfreq: float = 100.0,
) -> None:
    trial_dir = dataset_dir / "S_1" / "Trial_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "labels.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "Exec_Block_Index": 1,
                        "type": "geometric",
                        "pattern_id": 3,
                        "img": [[0, 1], [1, 0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    eeg_data = eeg if eeg is not None else np.arange(32, dtype=np.float64).reshape(2, 16)
    eog_data = np.resize(np.array([0.0, np.nan]), eeg_data.shape[1])[np.newaxis, :]
    _save_raw(
        trial_dir / "exec_EEG_1.fif",
        data=eeg_data,
        channel_names=["Fz", "Cz"],
        channel_type="eeg",
        sfreq=sfreq,
    )
    _save_raw(
        trial_dir / "exec_EOG_1.fif",
        data=eog_data,
        channel_names=["EOG_x"],
        channel_type="eog",
        sfreq=sfreq,
    )


class SyntheticFFTDataset(PreprocessedDataset):
    METHOD = "fft"
    CONFIG_TYPE = FFTConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.transform_calls = 0
        super().__init__(*args, **kwargs)

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        self.transform_calls += 1
        frequencies = np.array([2.0, 3.0, 4.0])
        power = np.abs(loaded.eeg[:, :3])
        return SpectralTransformResult(
            eeg_power=power,
            frequencies=frequencies,
            times=None,
            analysis_sfreq=self.config.analysis_sfreq,
            scaling=self.config.scaling,
        )


class SyntheticTimeFrequencyDataset(SyntheticFFTDataset):
    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        self.transform_calls += 1
        frequencies = np.array([2.0, 3.0, 4.0])
        times = np.array([0.0, 0.25])
        power = np.repeat(np.abs(loaded.eeg[:, :3, np.newaxis]), times.size, axis=2)
        return SpectralTransformResult(
            eeg_power=power,
            frequencies=frequencies,
            times=times,
            analysis_sfreq=self.config.analysis_sfreq,
            scaling=self.config.scaling,
        )


def test_builds_spectral_sample_and_preserves_original_eog(tmp_path: Path) -> None:
    _write_trial(tmp_path)
    dataset = SyntheticFFTDataset(
        tmp_path,
        config_overrides={"f_max": 4.0},
        cache_policy=None,
        source_cache_policy=None,
    )

    transformed = dataset[0]

    assert isinstance(transformed, SpectralSample)
    assert transformed.sample.block_index == 1
    assert transformed.eeg_power.shape == (2, 3)
    assert transformed.eeg_power.dtype == np.float32
    assert transformed.frequencies.dtype == np.float32
    assert transformed.times is None
    assert transformed.method == "fft"
    assert transformed.scaling == "psd"
    assert transformed.source_sfreq == 100.0
    assert transformed.analysis_sfreq == 125.0
    assert transformed.eeg_channels == ("Fz", "Cz")
    assert transformed.eog_channels == ("EOG_x",)
    assert np.isnan(transformed.eog).any()


def test_supports_integer_tuple_iteration_and_index_proxies(tmp_path: Path) -> None:
    _write_trial(tmp_path)
    dataset = SyntheticFFTDataset(
        tmp_path,
        config_overrides={"f_max": 4.0},
        cache_policy=None,
        source_cache_policy=None,
    )

    by_index = dataset[0]
    by_key = dataset[1, 1, 1]

    np.testing.assert_array_equal(by_index.eeg_power, by_key.eeg_power)
    assert [sample.sample.block_index for sample in dataset] == [1]
    assert dataset.samples[0].block_index == 1
    assert dataset.source_map[1][1][1].block_index == 1


def test_rejects_non_finite_source_eeg(tmp_path: Path) -> None:
    eeg = np.arange(32, dtype=np.float64).reshape(2, 16)
    eeg[0, 3] = np.nan
    _write_trial(tmp_path, eeg=eeg)
    dataset = SyntheticFFTDataset(
        tmp_path,
        config_overrides={"f_max": 4.0},
        cache_policy=None,
        source_cache_policy=None,
    )

    with pytest.raises(ValueError, match="Source EEG contains non-finite"):
        dataset[0]


def test_rejects_wrong_config_type(tmp_path: Path) -> None:
    _write_trial(tmp_path)
    morlet_config = load_preprocessing_config("morlet")

    with pytest.raises(TypeError, match="requires FFTConfig"):
        SyntheticFFTDataset(tmp_path, config=morlet_config, cache_policy=None, source_cache_policy=None)


def test_rejects_config_and_loading_options_together(tmp_path: Path) -> None:
    _write_trial(tmp_path)
    config = load_preprocessing_config("fft")

    with pytest.raises(ValueError, match="either `config`"):
        SyntheticFFTDataset(
            tmp_path,
            config=config,
            config_overrides={"dtype": "float64"},
            cache_policy=None,
            source_cache_policy=None,
        )


def test_fft_dataset_supports_indexing_dtype_shape_and_cache_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    sfreq = 100.0
    time = np.arange(400, dtype=np.float64) / sfreq
    eeg = np.stack(
        (
            np.sin(2.0 * np.pi * 10.0 * time),
            np.sin(2.0 * np.pi * 20.0 * time),
        )
    )
    _write_trial(dataset_dir, eeg=eeg, sfreq=sfreq)
    dataset = FFTDataset(
        dataset_dir,
        config_overrides={"analysis_sfreq": sfreq},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )

    by_index = dataset[0]
    by_key = dataset[1, 1, 1]

    assert by_index.eeg_power.shape == (2, 39)
    assert by_index.eeg_power.dtype == np.float32
    assert by_index.frequencies.dtype == np.float32
    assert by_index.times is None
    assert by_index.method == "fft"
    assert by_index.scaling == "psd"
    np.testing.assert_array_equal(by_key.eeg_power, by_index.eeg_power)
    assert by_index.frequencies[np.argmax(by_index.eeg_power[0])] == 10.0
    assert by_index.frequencies[np.argmax(by_index.eeg_power[1])] == 20.0

    def fail_transform(self: FFTDataset, loaded: LoadedSample) -> SpectralTransformResult:
        raise AssertionError("FFT transform should not run when the disk cache is valid")

    monkeypatch.setattr(FFTDataset, "_transform", fail_transform)
    cached_dataset = FFTDataset(
        dataset_dir,
        config_overrides={"analysis_sfreq": sfreq},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    cached = cached_dataset[0]

    np.testing.assert_array_equal(cached.eeg_power, by_index.eeg_power)


def test_morlet_dataset_supports_time_axis_and_cache_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    sfreq = 100.0
    time = np.arange(800, dtype=np.float64) / sfreq
    eeg = np.stack(
        (
            np.sin(2.0 * np.pi * 10.0 * time),
            np.sin(2.0 * np.pi * 20.0 * time),
        )
    )
    _write_trial(dataset_dir, eeg=eeg, sfreq=sfreq)
    dataset = MorletDataset(
        dataset_dir,
        config_overrides={"analysis_sfreq": sfreq},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )

    by_index = dataset[0]
    by_key = dataset[1, 1, 1]

    assert by_index.eeg_power.shape == (2, 39, 17)
    assert by_index.eeg_power.dtype == np.float32
    assert by_index.frequencies.dtype == np.float32
    assert by_index.times is not None
    assert by_index.times.dtype == np.float32
    assert by_index.method == "morlet"
    assert by_index.scaling == "wavelet_power"
    np.testing.assert_array_equal(by_key.eeg_power, by_index.eeg_power)
    np.testing.assert_array_equal(by_key.times, by_index.times)

    def fail_transform(self: MorletDataset, loaded: LoadedSample) -> SpectralTransformResult:
        raise AssertionError("Morlet transform should not run when the disk cache is valid")

    monkeypatch.setattr(MorletDataset, "_transform", fail_transform)
    cached_dataset = MorletDataset(
        dataset_dir,
        config_overrides={"analysis_sfreq": sfreq},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    cached = cached_dataset[0]

    np.testing.assert_array_equal(cached.eeg_power, by_index.eeg_power)
    np.testing.assert_array_equal(cached.times, by_index.times)


def test_superlet_dataset_supports_time_axis_and_cache_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    sfreq = 100.0
    time = np.arange(800, dtype=np.float64) / sfreq
    eeg = np.stack(
        (
            np.sin(2.0 * np.pi * 10.0 * time),
            np.sin(2.0 * np.pi * 20.0 * time),
        )
    )
    _write_trial(dataset_dir, eeg=eeg)
    dataset = SuperletDataset(
        dataset_dir,
        config_overrides={"analysis_sfreq": sfreq},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )

    by_index = dataset[0]
    by_key = dataset[1, 1, 1]

    assert by_index.eeg_power.shape == (2, 39, 15)
    assert by_index.eeg_power.dtype == np.float32
    assert by_index.frequencies.dtype == np.float32
    assert by_index.times is not None
    assert by_index.times.dtype == np.float32
    assert by_index.method == "superlet"
    assert by_index.scaling == "wavelet_power"
    np.testing.assert_array_equal(by_key.eeg_power, by_index.eeg_power)
    np.testing.assert_array_equal(by_key.times, by_index.times)

    def fail_transform(self: SuperletDataset, loaded: LoadedSample) -> SpectralTransformResult:
        raise AssertionError("Superlet transform should not run when the disk cache is valid")

    monkeypatch.setattr(SuperletDataset, "_transform", fail_transform)
    cached_dataset = SuperletDataset(
        dataset_dir,
        config_overrides={"analysis_sfreq": sfreq},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    cached = cached_dataset[0]

    np.testing.assert_array_equal(cached.eeg_power, by_index.eeg_power)
    np.testing.assert_array_equal(cached.times, by_index.times)


def test_stft_dataset_supports_time_axis_and_cache_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    sfreq = 125.0
    time = np.arange(1_000, dtype=np.float64) / sfreq
    eeg = np.stack(
        (
            np.sin(2.0 * np.pi * 10.0 * time),
            np.sin(2.0 * np.pi * 20.0 * time),
        )
    )
    _write_trial(dataset_dir, eeg=eeg, sfreq=sfreq)
    dataset = STFTDataset(
        dataset_dir,
        cache_dir=cache_dir,
        source_cache_policy=None,
    )

    by_index = dataset[0]
    by_key = dataset[1, 1, 1]

    assert by_index.eeg_power.shape == (2, 39, 24)
    assert by_index.eeg_power.dtype == np.float32
    assert by_index.frequencies.dtype == np.float32
    assert by_index.times is not None
    assert by_index.times.dtype == np.float32
    assert by_index.method == "stft"
    assert by_index.scaling == "psd"
    np.testing.assert_array_equal(by_key.eeg_power, by_index.eeg_power)
    np.testing.assert_array_equal(by_key.times, by_index.times)

    def fail_transform(self: STFTDataset, loaded: LoadedSample) -> SpectralTransformResult:
        raise AssertionError("STFT transform should not run when the disk cache is valid")

    monkeypatch.setattr(STFTDataset, "_transform", fail_transform)
    cached_dataset = STFTDataset(
        dataset_dir,
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    cached = cached_dataset[0]

    np.testing.assert_array_equal(cached.eeg_power, by_index.eeg_power)
    np.testing.assert_array_equal(cached.times, by_index.times)


def test_disk_cache_writes_only_spectral_arrays_and_reuses_transform(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    _write_trial(dataset_dir)
    dataset = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_policy="disk",
        cache_dir=cache_dir,
        source_cache_policy=None,
    )

    first = dataset[0]
    entry_dir = dataset.get_cache_entry_path(0)
    second_dataset = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_policy="disk",
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    second = second_dataset[0]

    assert dataset.transform_calls == 1
    assert second_dataset.transform_calls == 0
    assert (entry_dir / "eeg_power.npy").is_file()
    assert (entry_dir / "frequencies.npy").is_file()
    assert (entry_dir / "manifest.json").is_file()
    assert not (entry_dir / "times.npy").exists()
    assert not (entry_dir / "eog.npy").exists()
    np.testing.assert_array_equal(second.eeg_power, first.eeg_power)
    np.testing.assert_array_equal(second.eog, first.eog)


def test_disk_cache_round_trips_time_axis(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    _write_trial(dataset_dir)
    dataset = SyntheticTimeFrequencyDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )

    first = dataset[0]
    second = dataset[0]

    assert dataset.transform_calls == 1
    assert (dataset.get_cache_entry_path(0) / "times.npy").is_file()
    np.testing.assert_array_equal(second.times, first.times)
    np.testing.assert_array_equal(second.eeg_power, first.eeg_power)


@pytest.mark.parametrize("source_name", ["eeg", "eog"])
def test_disk_cache_is_invalidated_when_source_changes(tmp_path: Path, source_name: str) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    _write_trial(dataset_dir)
    dataset = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    dataset[0]

    source_path = getattr(dataset.samples[0], f"{source_name}_path")
    stat = source_path.stat()
    os.utime(source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    dataset[0]

    assert dataset.transform_calls == 2


@pytest.mark.parametrize("filename", ["manifest.json", "eeg_power.npy", "frequencies.npy"])
def test_incomplete_or_corrupt_disk_cache_is_rebuilt(tmp_path: Path, filename: str) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    _write_trial(dataset_dir)
    dataset = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    dataset[0]
    cache_path = dataset.get_cache_entry_path(0) / filename
    if filename == "manifest.json":
        cache_path.unlink()
    else:
        cache_path.write_bytes(b"not-a-valid-npy")

    rebuilt = dataset[0]

    assert dataset.transform_calls == 2
    assert rebuilt.eeg_power.shape == (2, 3)


def test_manifest_contains_config_source_and_array_metadata(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    _write_trial(dataset_dir)
    dataset = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    spectral_sample = dataset[0]

    with (dataset.get_cache_entry_path(0) / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["schema_version"] == dataset.CACHE_SCHEMA_VERSION
    assert manifest["transform_version"] == dataset.TRANSFORM_VERSION
    assert manifest["transform_class"].endswith(".SyntheticFFTDataset")
    assert manifest["config_hash"] == dataset.config_hash
    assert manifest["config"] == dataset.config.model_dump(mode="json")
    assert manifest["key"] == {"subject_id": 1, "trial_number": 1, "block_index": 1}
    assert manifest["method"] == "fft"
    assert manifest["scaling"] == "psd"
    assert manifest["arrays"]["eeg_power"] == {
        "shape": list(spectral_sample.eeg_power.shape),
        "dtype": "float32",
    }
    assert manifest["arrays"]["times"] is None
    assert manifest["sources"]["eeg"]["size"] == dataset.samples[0].eeg_path.stat().st_size
    assert manifest["sources"]["eog"]["mtime_ns"] == dataset.samples[0].eog_path.stat().st_mtime_ns


def test_config_and_transform_version_change_cache_identity(tmp_path: Path) -> None:
    _write_trial(tmp_path)

    default = SyntheticFFTDataset(
        tmp_path,
        config_overrides={"f_max": 4.0},
        cache_policy=None,
        source_cache_policy=None,
    )
    float64 = SyntheticFFTDataset(
        tmp_path,
        config_overrides={"f_max": 4.0, "dtype": "float64"},
        cache_policy=None,
        source_cache_policy=None,
    )

    class VersionedSyntheticDataset(SyntheticFFTDataset):
        TRANSFORM_VERSION = 2

    versioned = VersionedSyntheticDataset(
        tmp_path,
        config_overrides={"f_max": 4.0},
        cache_policy=None,
        source_cache_policy=None,
    )

    assert len(default.config_hash) == 16
    assert default.config_hash != float64.config_hash
    assert default.config_hash != versioned.config_hash


def test_transform_version_invalidates_shared_custom_cache(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    _write_trial(dataset_dir)
    original = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    original[0]

    class VersionedSyntheticDataset(SyntheticFFTDataset):
        TRANSFORM_VERSION = 2

    versioned = VersionedSyntheticDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    versioned[0]

    assert original.transform_calls == 1
    assert versioned.transform_calls == 1


def test_config_change_invalidates_shared_custom_cache(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    _write_trial(dataset_dir)
    original = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    original[0]

    changed = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0, "dtype": "float64"},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    transformed = changed[0]

    assert original.transform_calls == 1
    assert changed.transform_calls == 1
    assert transformed.eeg_power.dtype == np.float64


def test_schema_version_change_in_manifest_invalidates_cache(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    cache_dir = tmp_path / "spectral-cache"
    _write_trial(dataset_dir)
    dataset = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        cache_dir=cache_dir,
        source_cache_policy=None,
    )
    dataset[0]
    manifest_path = dataset.get_cache_entry_path(0) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset[0]

    assert dataset.transform_calls == 2


def test_default_cache_path_uses_artifact_hierarchy(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    dataset_dir = project_dir / "data" / "Example"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").touch()
    _write_trial(dataset_dir)

    dataset = SyntheticFFTDataset(
        dataset_dir,
        config_overrides={"f_max": 4.0},
        source_cache_policy=None,
    )

    assert dataset.cache_dir == (
        project_dir
        / "artifacts"
        / "preprocessed"
        / "Example"
        / "exec"
        / "fft"
        / dataset.config_hash
    )


def test_rejects_unsupported_spectral_cache_policy(tmp_path: Path) -> None:
    _write_trial(tmp_path)

    with pytest.raises(ValueError, match="Unsupported spectral cache policy"):
        SyntheticFFTDataset(
            tmp_path,
            config_overrides={"f_max": 4.0},
            cache_policy="memory",  # type: ignore[arg-type]
            source_cache_policy=None,
        )
