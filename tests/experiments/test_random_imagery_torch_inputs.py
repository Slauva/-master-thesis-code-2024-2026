import json
import os
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import mne
import numpy as np
import pytest
import torch

from experiments.random_imagery import build_random_imagery_targets
from experiments.random_imagery_torch import (
    CropSpectralDataset,
    CropSpectralSample,
    SpectralInputConfig,
    TorchSpectralInputDataset,
    collate_spectral_inputs,
    fit_spectral_normalization,
    normalize_spectral_sample,
)
from preprocessors import SpectralTransformResult, load_preprocessing_config
from utils.datasets import NumpyDataset, RandomSample

SpectralMethod = Literal["fft", "morlet", "superlet", "stft"]


def _save_raw(
    path: Path,
    *,
    data: np.ndarray,
    channel_names: list[str],
    channel_type: str,
    sfreq: float,
) -> None:
    info = mne.create_info(
        channel_names,
        sfreq=sfreq,
        ch_types=[channel_type] * len(channel_names),
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw.save(path, overwrite=True, verbose="ERROR")


def _write_random_trial(
    dataset_dir: Path,
    *,
    sfreq: float = 100.0,
    duration_seconds: float = 16.0,
) -> None:
    trial_dir = dataset_dir / "S_1" / "Trial_1"
    trial_dir.mkdir(parents=True)
    image = (np.arange(36).reshape(6, 6) % 2).tolist()
    (trial_dir / "labels.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "Exec_Block_Index": 1,
                        "type": "random",
                        "seed": 123,
                        "img": image,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    n_times = round(sfreq * duration_seconds)
    time = np.arange(n_times, dtype=np.float64) / sfreq
    eeg = np.stack(
        (
            np.sin(2.0 * np.pi * 10.0 * time),
            np.cos(2.0 * np.pi * 20.0 * time),
        )
    )
    eog = np.zeros((1, n_times), dtype=np.float64)
    _save_raw(
        trial_dir / "patt_EEG_1.fif",
        data=eeg,
        channel_names=["Fz", "Cz"],
        channel_type="eeg",
        sfreq=sfreq,
    )
    _save_raw(
        trial_dir / "patt_EOG_1.fif",
        data=eog,
        channel_names=["VEOG"],
        channel_type="eog",
        sfreq=sfreq,
    )


def _source_dataset(dataset_dir: Path) -> NumpyDataset:
    return NumpyDataset(
        dataset_dir,
        dataset_step_type="patt",
        dataset_pattern_type="random",
        cache_policy=None,
    )


def _fake_fft_transform(
    eeg: np.ndarray,
    *,
    source_sfreq: float,
    config: Any,
) -> SpectralTransformResult:
    del source_sfreq
    frequencies = np.arange(2.0, 41.0, dtype=np.float32)
    power = np.repeat(np.mean(np.square(eeg), axis=1, keepdims=True), 39, axis=1)
    return SpectralTransformResult(
        eeg_power=power.astype(np.float32),
        frequencies=frequencies,
        times=None,
        analysis_sfreq=config.analysis_sfreq,
        scaling=config.scaling,
    )


def test_crop_is_applied_before_transform_and_full_recording_cache_is_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "Data_Pattern"
    _write_random_trial(dataset_dir)
    observed_shapes: list[tuple[int, ...]] = []

    def recording_transform(
        eeg: np.ndarray,
        *,
        source_sfreq: float,
        config: Any,
    ) -> SpectralTransformResult:
        observed_shapes.append(eeg.shape)
        return _fake_fft_transform(
            eeg,
            source_sfreq=source_sfreq,
            config=config,
        )

    monkeypatch.setitem(
        __import__(
            "experiments.random_imagery_torch.spectral_dataset",
            fromlist=["_TRANSFORMS"],
        )._TRANSFORMS,
        "fft",
        recording_transform,
    )
    dataset = CropSpectralDataset(
        _source_dataset(dataset_dir),
        method="fft",
        cache_dir=tmp_path / "crop-cache",
    )

    sample = dataset[0]

    assert observed_shapes == [(2, 1_500)]
    assert sample.crop_bounds_seconds == (0.5, 15.5)
    assert sample.eeg_power.shape == (2, 39)
    assert "preprocessed-imagery" not in str(tmp_path / "crop-cache")
    assert dataset.config_hash


def test_crop_cache_hits_before_source_loading_and_rebuilds_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "Data_Pattern"
    _write_random_trial(dataset_dir)
    source = _source_dataset(dataset_dir)
    dataset = CropSpectralDataset(
        source,
        method="fft",
        cache_dir=tmp_path / "crop-cache",
    )
    module = __import__(
        "experiments.random_imagery_torch.spectral_dataset",
        fromlist=["_TRANSFORMS"],
    )
    calls = 0

    def counting_transform(
        eeg: np.ndarray,
        *,
        source_sfreq: float,
        config: Any,
    ) -> SpectralTransformResult:
        nonlocal calls
        calls += 1
        return _fake_fft_transform(
            eeg,
            source_sfreq=source_sfreq,
            config=config,
        )

    monkeypatch.setitem(module._TRANSFORMS, "fft", counting_transform)
    first = dataset[0]

    def fail_source_load(sample: RandomSample) -> None:
        raise AssertionError(f"Source arrays should not load on a valid crop-cache hit: {sample}")

    monkeypatch.setattr(source, "_load_sample", fail_source_load)
    cached = dataset[0]
    assert calls == 1
    np.testing.assert_array_equal(cached.eeg_power, first.eeg_power)

    (dataset.get_cache_entry_path(0) / "eeg_power.npy").write_bytes(b"corrupt")
    monkeypatch.setattr(source, "_load_sample", NumpyDataset._load_sample.__get__(source))
    rebuilt = dataset[0]
    assert calls == 2
    np.testing.assert_array_equal(rebuilt.eeg_power, first.eeg_power)


def test_crop_cache_invalidates_on_eeg_source_signature_and_crop_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "Data_Pattern"
    _write_random_trial(dataset_dir)
    source = _source_dataset(dataset_dir)
    cache_dir = tmp_path / "crop-cache"
    dataset = CropSpectralDataset(source, method="fft", cache_dir=cache_dir)
    module = __import__(
        "experiments.random_imagery_torch.spectral_dataset",
        fromlist=["_TRANSFORMS"],
    )
    calls = 0

    def counting_transform(
        eeg: np.ndarray,
        *,
        source_sfreq: float,
        config: Any,
    ) -> SpectralTransformResult:
        nonlocal calls
        calls += 1
        return _fake_fft_transform(
            eeg,
            source_sfreq=source_sfreq,
            config=config,
        )

    monkeypatch.setitem(module._TRANSFORMS, "fft", counting_transform)
    dataset[0]
    source_path = source.samples[0].eeg_path
    stat = source_path.stat()
    os.utime(source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    dataset[0]

    changed_crop = CropSpectralDataset(
        source,
        method="fft",
        input_config=SpectralInputConfig(
            crop_start_seconds=0.0,
            crop_end_seconds=15.0,
        ),
        cache_dir=cache_dir,
    )
    changed_crop[0]

    assert calls == 3
    assert dataset.config_hash != changed_crop.config_hash
    manifest = json.loads(
        (changed_crop.get_cache_entry_path(0) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["crop_bounds_seconds"] == [0.0, 15.0]
    assert manifest["time_reference"] == "source_recording_seconds"


def test_crop_config_requires_exact_source_sample_boundaries() -> None:
    config = SpectralInputConfig()

    with pytest.raises(ValueError, match="integer sample"):
        config.source_slice(127.0, n_times=2_000)


class StubCropDataset:
    def __init__(self, samples: list[CropSpectralSample]) -> None:
        self._by_key = {sample.sample_key: sample for sample in samples}
        self.samples = tuple(sample.sample for sample in samples)
        self.input_config = SpectralInputConfig()
        self.method = samples[0].method
        self.preprocessing_config = load_preprocessing_config(
            self.method,
            overrides={"f_max": 4.0},
        )
        self.requested_keys: list[tuple[int, int, int]] = []

    def __getitem__(self, key: tuple[int, int, int]) -> CropSpectralSample:
        self.requested_keys.append(key)
        return self._by_key[key]


def _random_metadata(index: int) -> RandomSample:
    image = ((np.arange(36) + index) % 2).reshape(6, 6).tolist()
    return RandomSample(
        subject_id=index,
        trial_number=1,
        Exec_Block_Index=1,
        eeg_path=Path(f"eeg-{index}.fif"),
        eog_path=Path(f"eog-{index}.fif"),
        img=image,
        seed=100 + index,
    )


def _crop_sample(
    index: int,
    *,
    method: SpectralMethod,
    offset: float,
) -> CropSpectralSample:
    metadata = _random_metadata(index)
    frequencies = np.arange(2.0, 5.0, dtype=np.float32)
    if method == "fft":
        power = np.arange(1, 7, dtype=np.float32).reshape(2, 3) + offset
        times = None
        scaling = "psd"
    else:
        power = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4) + offset
        times = np.arange(4, dtype=np.float32) * 0.256 + 1.0
        scaling = "psd" if method == "stft" else "wavelet_power"
    return CropSpectralSample(
        sample=metadata,
        eeg_power=power,
        frequencies=frequencies,
        times=times,
        eeg_channels=("Fz", "Cz"),
        source_sfreq=1_000.0,
        analysis_sfreq=125.0,
        method=method,
        scaling=scaling,
        crop_bounds_seconds=(0.5, 15.5),
    )


@pytest.mark.parametrize(
    ("method", "expected_shape"),
    [
        ("fft", (1, 2, 3)),
        ("morlet", (3, 2, 4)),
        ("superlet", (3, 2, 4)),
        ("stft", (3, 2, 4)),
    ],
)
def test_train_only_normalization_and_model_input_geometry(
    method: SpectralMethod,
    expected_shape: tuple[int, ...],
) -> None:
    samples = [
        _crop_sample(1, method=method, offset=0.0),
        _crop_sample(2, method=method, offset=1.0),
        _crop_sample(3, method=method, offset=2.0),
    ]
    dataset = StubCropDataset(samples)
    train_keys = (samples[0].sample_key, samples[1].sample_key)

    state = fit_spectral_normalization(dataset, train_keys)  # type: ignore[arg-type]
    model_input = normalize_spectral_sample(samples[2], state)

    assert dataset.requested_keys == list(train_keys)
    assert state.fit_sample_keys == train_keys
    assert state.mean.shape == (3,)
    assert state.scale.shape == (3,)
    assert state.observation_count == (
        4 if method == "fft" else 16
    )
    assert model_input.shape == expected_shape
    assert model_input.dtype == np.float32
    assert np.isfinite(model_input).all()
    assert not state.mean.flags.writeable
    assert not state.scale.flags.writeable


def test_aligned_torch_dataset_delays_test_access_and_preserves_targets() -> None:
    samples = [
        _crop_sample(1, method="fft", offset=0.0),
        _crop_sample(2, method="fft", offset=1.0),
        _crop_sample(3, method="fft", offset=2.0),
    ]
    dataset = StubCropDataset(samples)
    targets = build_random_imagery_targets(dataset.samples)
    train_keys = (samples[0].sample_key, samples[1].sample_key)
    state = fit_spectral_normalization(dataset, train_keys)  # type: ignore[arg-type]
    assert dataset.requested_keys == list(train_keys)

    test_dataset = TorchSpectralInputDataset(
        dataset,  # type: ignore[arg-type]
        targets,
        np.asarray([2], dtype=np.int64),
        state,
    )
    assert dataset.requested_keys == list(train_keys)

    item = test_dataset[0]
    assert dataset.requested_keys == [*train_keys, samples[2].sample_key]
    assert item.sample_key == targets.sample_keys[2]
    torch.testing.assert_close(item.target, torch.from_numpy(targets.y[2].astype(np.float32)))
    assert item.model_input.shape == (1, 2, 3)

    batch = collate_spectral_inputs([item])
    assert batch.model_inputs.shape == (1, 1, 2, 3)
    assert batch.targets.shape == (1, 36)
    assert batch.sample_keys == (samples[2].sample_key,)


def test_aligned_torch_dataset_rejects_target_payload_mismatch() -> None:
    sample = _crop_sample(1, method="fft", offset=0.0)
    dataset = StubCropDataset([sample])
    targets = build_random_imagery_targets(dataset.samples)
    mismatched_y = targets.y.copy()
    mismatched_y[0, 0] = np.int8(1 - mismatched_y[0, 0])
    targets = replace(targets, y=mismatched_y)
    state = fit_spectral_normalization(dataset, (sample.sample_key,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Target payload"):
        TorchSpectralInputDataset(
            dataset,  # type: ignore[arg-type]
            targets,
            np.asarray([0], dtype=np.int64),
            state,
        )
