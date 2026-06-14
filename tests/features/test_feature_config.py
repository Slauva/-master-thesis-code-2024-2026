from pathlib import Path

import pytest
from pydantic import ValidationError

from features import FeatureExtractionConfig, build_feature_config_hash, load_feature_config


def test_loads_default_feature_configuration() -> None:
    config = load_feature_config()

    assert config.analysis_sfreq == 125.0
    assert (config.crop_start_seconds, config.crop_end_seconds) == (0.5, 15.5)
    assert config.window_seconds is None
    assert config.window_stride_seconds is None
    assert config.dtype == "float32"
    assert config.feature_groups == ("time", "spectral", "spatial", "local_patterns")
    assert tuple(band.name for band in config.frequency_bands) == (
        "delta",
        "theta",
        "alpha",
        "beta",
        "low_gamma",
    )
    assert config.histogram_mode == "probability"
    assert config.local_pattern_neighbors == 8


def test_loads_runtime_overrides_without_mutating_default() -> None:
    configured = load_feature_config(
        overrides={
            "window_seconds": 4.0,
            "window_stride_seconds": 2.0,
            "feature_groups": ["time", "spectral"],
            "histogram_mode": "count",
        }
    )
    default = load_feature_config()

    assert configured.window_seconds == 4.0
    assert configured.window_stride_seconds == 2.0
    assert configured.feature_groups == ("time", "spectral")
    assert configured.histogram_mode == "count"
    assert default.window_seconds is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"crop_start_seconds": 15.5},
        {"window_seconds": 4.0},
        {"window_stride_seconds": 2.0},
        {"window_seconds": 16.0, "window_stride_seconds": 1.0},
        {"window_seconds": 4.0, "window_stride_seconds": 2.5},
        {"feature_groups": []},
        {"feature_groups": ["time", "time"]},
        {"local_pattern_neighbors": 7},
        {"local_pattern_neighbors": 18},
    ],
)
def test_rejects_invalid_feature_configuration(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        load_feature_config(overrides=overrides)


def test_rejects_overlapping_or_out_of_nyquist_bands() -> None:
    with pytest.raises(ValidationError, match="non-overlapping"):
        load_feature_config(
            overrides={
                "frequency_bands": [
                    {"name": "first", "f_min": 2.0, "f_max": 10.0},
                    {"name": "second", "f_min": 8.0, "f_max": 12.0},
                ]
            }
        )

    with pytest.raises(ValidationError, match="Nyquist"):
        load_feature_config(
            overrides={
                "frequency_bands": [
                    {"name": "invalid", "f_min": 40.0, "f_max": 70.0},
                ]
            }
        )


def test_config_hash_is_stable_and_versioned() -> None:
    config = load_feature_config()

    assert build_feature_config_hash(config) == build_feature_config_hash(config)
    assert build_feature_config_hash(config) != build_feature_config_hash(config, extractor_version=2)
    assert build_feature_config_hash(config) != build_feature_config_hash(
        load_feature_config(overrides={"dtype": "float64"})
    )


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_feature_config(config_path=tmp_path / "missing.yaml")


def test_config_rejects_unknown_fields() -> None:
    payload = load_feature_config().model_dump(mode="python")
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        FeatureExtractionConfig.model_validate(payload)
