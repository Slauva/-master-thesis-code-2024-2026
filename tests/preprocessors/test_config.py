from pathlib import Path

import pytest
from pydantic import ValidationError

from preprocessors import (
    FFTConfig,
    MorletConfig,
    STFTConfig,
    SuperletConfig,
    build_frequency_grid,
    load_preprocessing_config,
)


@pytest.mark.parametrize(
    ("method", "expected_type", "expected_scaling"),
    [
        ("fft", FFTConfig, "psd"),
        ("morlet", MorletConfig, "wavelet_power"),
        ("superlet", SuperletConfig, "wavelet_power"),
        ("stft", STFTConfig, "psd"),
    ],
)
def test_loads_default_method_config(method: str, expected_type: type, expected_scaling: str) -> None:
    config = load_preprocessing_config(method)  # type: ignore[arg-type]

    assert isinstance(config, expected_type)
    assert config.method == method
    assert config.scaling == expected_scaling
    assert config.analysis_sfreq == 125.0
    assert config.f_min == 2.0
    assert config.f_max == 40.0
    assert config.dtype == "float32"
    assert config.transform_eog is False
    assert config.filter_hz is None
    assert config.notch_hz is None
    assert config.reference is None
    assert config.normalization == "none"


def test_overrides_are_merged_last_and_validated() -> None:
    config = load_preprocessing_config(
        "fft",
        overrides={"analysis_sfreq": 100.0, "f_max": 45.0, "dtype": "float64"},
    )

    assert config.analysis_sfreq == 100.0
    assert config.f_max == 45.0
    assert config.dtype == "float64"


def test_builds_inclusive_frequency_grid() -> None:
    config = load_preprocessing_config(
        "fft",
        overrides={"f_min": 2.0, "f_max": 4.0, "frequency_step": 0.5},
    )

    assert build_frequency_grid(config).tolist() == [2.0, 2.5, 3.0, 3.5, 4.0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"f_min": 40.0, "f_max": 2.0},
        {"analysis_sfreq": 60.0, "f_max": 40.0},
        {"f_min": 2.0, "f_max": 4.0, "frequency_step": 0.7},
        {"transform_eog": True},
        {"normalization": "zscore"},
        {"unknown_field": 1},
    ],
)
def test_rejects_invalid_or_unsupported_configuration(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        load_preprocessing_config("fft", overrides=overrides)


def test_rejects_missing_config_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="common.yaml"):
        load_preprocessing_config("fft", config_dir=tmp_path)


def test_rejects_stft_configuration_with_fractional_window_samples() -> None:
    with pytest.raises(ValidationError):
        load_preprocessing_config("stft", overrides={"window_seconds": 2.001})
