import json
import warnings
from pathlib import Path

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


def _write_trial(dataset_dir: Path, *, eeg: np.ndarray | None = None) -> None:
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
    eog_data = np.array([[0.0, np.nan] * 8])
    _save_raw(
        trial_dir / "exec_EEG_1.fif",
        data=eeg_data,
        channel_names=["Fz", "Cz"],
        channel_type="eeg",
        sfreq=100.0,
    )
    _save_raw(
        trial_dir / "exec_EOG_1.fif",
        data=eog_data,
        channel_names=["EOG_x"],
        channel_type="eog",
        sfreq=100.0,
    )


class SyntheticFFTDataset(PreprocessedDataset):
    METHOD = "fft"
    CONFIG_TYPE = FFTConfig

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        frequencies = np.array([2.0, 3.0, 4.0])
        power = np.abs(loaded.eeg[:, :3])
        return SpectralTransformResult(
            eeg_power=power,
            frequencies=frequencies,
            times=None,
            analysis_sfreq=self.config.analysis_sfreq,
            scaling=self.config.scaling,
        )


def test_builds_spectral_sample_and_preserves_original_eog(tmp_path: Path) -> None:
    _write_trial(tmp_path)
    dataset = SyntheticFFTDataset(
        tmp_path,
        config_overrides={"f_max": 4.0},
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
        source_cache_policy=None,
    )

    with pytest.raises(ValueError, match="Source EEG contains non-finite"):
        dataset[0]


def test_rejects_wrong_config_type(tmp_path: Path) -> None:
    _write_trial(tmp_path)
    morlet_config = load_preprocessing_config("morlet")

    with pytest.raises(TypeError, match="requires FFTConfig"):
        SyntheticFFTDataset(tmp_path, config=morlet_config, source_cache_policy=None)


def test_rejects_config_and_loading_options_together(tmp_path: Path) -> None:
    _write_trial(tmp_path)
    config = load_preprocessing_config("fft")

    with pytest.raises(ValueError, match="either `config`"):
        SyntheticFFTDataset(
            tmp_path,
            config=config,
            config_overrides={"dtype": "float64"},
            source_cache_policy=None,
        )


@pytest.mark.parametrize(
    ("dataset_type", "method", "checkpoint"),
    [
        (FFTDataset, "fft", 4),
        (MorletDataset, "morlet", 5),
        (SuperletDataset, "superlet", 6),
        (STFTDataset, "stft", 7),
    ],
)
def test_public_dataset_loads_its_config_and_marks_future_transform(
    tmp_path: Path,
    dataset_type: type[PreprocessedDataset],
    method: str,
    checkpoint: int,
) -> None:
    _write_trial(tmp_path)
    dataset = dataset_type(tmp_path, source_cache_policy=None)

    assert dataset.config.method == method
    with pytest.raises(NotImplementedError, match=f"checkpoint {checkpoint}"):
        dataset[0]
