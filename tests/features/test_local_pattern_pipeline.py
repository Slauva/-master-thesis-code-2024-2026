import numpy as np

from features import (
    extract_local_pattern_features,
    load_feature_config,
    prepare_feature_windows,
)


def test_local_pattern_pipeline_returns_named_modular_histograms() -> None:
    source_sfreq = 1_000.0
    times = np.arange(16_001, dtype=np.float64) / source_sfreq
    eeg = np.stack(
        (
            np.sin(2 * np.pi * 10 * times),
            np.cos(2 * np.pi * 20 * times),
        )
    )
    config = load_feature_config(
        overrides={
            "feature_groups": ["local_patterns"],
            "window_seconds": 5.0,
            "window_stride_seconds": 2.0,
        }
    )
    windows = prepare_feature_windows(eeg, source_sfreq=source_sfreq, config=config)

    blocks = extract_local_pattern_features(windows, config=config)

    assert tuple(block.name for block in blocks) == ("lndp", "lgp", "lbp")
    assert all(block.layout == "channel_histogram" for block in blocks)
    assert all(block.values.shape == (6, 2, 256) for block in blocks)
    assert all(block.values.dtype == np.dtype(np.float32) for block in blocks)
    assert blocks[0].feature_names[0] == "code_000"
    assert blocks[0].feature_names[-1] == "code_255"
    for block in blocks:
        np.testing.assert_allclose(block.values.sum(axis=-1), np.ones((6, 2)), atol=1e-6)


def test_count_mode_retains_number_of_valid_centers() -> None:
    config = load_feature_config(
        overrides={
            "feature_groups": ["local_patterns"],
            "histogram_mode": "count",
        }
    )
    windows = prepare_feature_windows(
        np.zeros((1, 16_001), dtype=np.float32),
        source_sfreq=1_000.0,
        config=config,
    )

    blocks = extract_local_pattern_features(windows, config=config)

    expected_codes = windows.values.shape[-1] - config.local_pattern_neighbors
    for block in blocks:
        assert block.values.sum() == expected_codes
